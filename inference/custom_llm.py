import os
import json
import itertools
from typing import List, Union, Tuple
import torch
import safetensors
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput
from vllm.sampling_params import BeamSearchParams
from vllm.beam_search import (
    BeamSearchOutput, 
    BeamSearchSequence, 
    BeamSearchInstance, 
    get_beam_search_score, 
)
from vllm.inputs import TokensPrompt
from vllm.sequence import Logprob
from vllm.lora.request import LoRARequest


class CustomLLM(LLM):
    """
    1. Improve the implementation of `beam_search`. 
    2. Implement a buffer to restore querys to run. 
    """

    def __init__(self, model, tokenizer = None, tokenizer_mode = "auto", skip_tokenizer_init = False, trust_remote_code = False, allowed_local_media_path = "", tensor_parallel_size = 1, dtype = "auto", quantization = None, revision = None, tokenizer_revision = None, seed = 0, gpu_memory_utilization = 0.9, swap_space = 4, cpu_offload_gb = 0, enforce_eager = None, max_seq_len_to_capture = 8192, disable_custom_all_reduce = False, disable_async_output_proc = False, hf_overrides = None, mm_processor_kwargs = None, task = "auto", override_pooler_config = None, compilation_config = None, **kwargs):
        super().__init__(model, tokenizer, tokenizer_mode, skip_tokenizer_init, trust_remote_code, allowed_local_media_path, tensor_parallel_size, dtype, quantization, revision, tokenizer_revision, seed, gpu_memory_utilization, swap_space, cpu_offload_gb, enforce_eager, max_seq_len_to_capture, disable_custom_all_reduce, disable_async_output_proc, hf_overrides, mm_processor_kwargs, task, override_pooler_config, compilation_config, **kwargs)

        self.query_buffer = []  # query buffer to parallel inference
        self.stats = {
            '#LLM Call': 0, 
            '#Total Input Words': 0, 
            '#Total Input Tokens': 0, 
            '#Total Output Words': 0, 
            '#Total Output Tokens': 0, 
        }

    def generate(self, prompts, sampling_params, lora_request, use_tqdm):
        outputs = super().generate(
            prompts, 
            sampling_params=sampling_params, 
            lora_request=lora_request, 
            use_tqdm=use_tqdm, 
        )
    
        tokenizer = self.get_tokenizer()
        # Do stats!
        if isinstance(prompts, str): 
            out_text = outputs[0].outputs[0].text
            self.stats['#LLM Call'] += 1
            self.stats['#Total Input Words'] += len(prompts.split(' '))
            self.stats['#Total Input Tokens'] += len(tokenizer.tokenize(prompts))
            self.stats['#Total Output Words'] += len(out_text.split(' '))
            self.stats['#Total Output Tokens'] += len(tokenizer.tokenize(out_text))
        else: 
            assert isinstance(prompts, list)
            for i, prompt in enumerate(prompts): 
                out_text = outputs[i].outputs[0].text
                self.stats['#LLM Call'] += 1
                self.stats['#Total Input Words'] += len(prompt.split(' '))
                self.stats['#Total Input Tokens'] += len(tokenizer.tokenize(prompt))
                self.stats['#Total Output Words'] += len(out_text.split(' '))
                self.stats['#Total Output Tokens'] += len(tokenizer.tokenize(out_text))

        return outputs

    def beam_search(
        self,
        prompts: str | List[int] | List[str] | List[List[int]],
        params: BeamSearchParams,
        lora_request: LoRARequest = None, 
        skip_special_tokens: bool = False, 
    ) -> List[BeamSearchOutput]:
        """
        Generate sequences using beam search.
        modifications: 
            * support lora_request
            * remove prompt from return values
            * bos token will not be automatically added to prompt
            * support single prompt input

        Args:
            prompts: Single prompt or a list of prompts. Each prompt
                can be a string or a list of token IDs.
            params: The beam search parameters.
        """

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty

        def sort_beams_key(x: BeamSearchSequence) -> float:
            return get_beam_search_score(x.tokens, x.cum_logprob,
                                         tokenizer.eos_token_id,
                                         length_penalty)

        tokenizer = self.get_tokenizer()
        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: List[BeamSearchInstance] = []

        if isinstance(prompts, str): 
            prompts = [ prompts ]
        elif isinstance(prompts, list):
            if not prompts:
                raise ValueError(f"`prompts` param is an empty list")
            if all(isinstance(prompt, int) for prompt in prompts):  # list of token ids
                prompts = [ prompts ]
            elif not any(isinstance(prompt, (str, list)) for prompt in prompts):
                raise ValueError(f"list element of `prompts` param must be `str/list`, not [{', '.join([type(prompt) for prompt in prompts[:50]])}...]")
        else:
            raise ValueError(f"`prompts` param must be `str/list`, not {type(prompts)}")

        for prompt in prompts:
            prompt_tokens = (
                prompt if isinstance(prompt, list) 
                else tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
            )
            instances.append(BeamSearchInstance(prompt_tokens))

        for _ in range(max_tokens):
            all_beams: List[BeamSearchSequence] = list(
                sum((instance.beams for instance in instances), [])
            )
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in instances
                )
            )
            instance_start_and_end: List[Tuple[int, int]] = list(zip(pos[:-1], pos[1:]))

            if len(all_beams) == 0:
                break

            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens)
                for beam in all_beams
            ]

            # only runs for one step
            # we don't need to use tqdm here
            output = self.generate(prompts_batch,
                                   sampling_params=beam_search_params,
                                   use_tqdm=False, 
                                   lora_request=lora_request)

            for (start, end), instance in zip(instance_start_and_end,
                                              instances):
                instance_new_beams = []
                for i in range(start, end):
                    current_beam = all_beams[i]
                    result = output[i]

                    if result.outputs[0].logprobs is not None:
                        # if `result.outputs[0].logprobs` is None, it means
                        # the sequence is completed because of the max-model-len
                        # or abortion. we don't need to add it to the new beams.
                        logprobs = result.outputs[0].logprobs[0]
                        for token_id, logprob_obj in logprobs.items():
                            new_beam = BeamSearchSequence(
                                tokens=current_beam.tokens + [token_id],
                                logprobs=current_beam.logprobs + [logprobs],
                                cum_logprob=current_beam.cum_logprob + logprob_obj.logprob)

                            if token_id == tokenizer.eos_token_id and not ignore_eos:
                                instance.completed.append(new_beam)
                            else:
                                instance_new_beams.append(new_beam)
                sorted_beams = sorted(instance_new_beams,
                                      key=sort_beams_key,
                                      reverse=True)
                instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed,
                                      key=sort_beams_key,
                                      reverse=True)
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens[-len(beam.logprobs):], skip_special_tokens=skip_special_tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def register_query(self, prompt, sampling_params, lora_request):
        if not hasattr(self, "query_buffer"):
            self.query_buffer = []
        self.query_buffer.append({
            'prompt': prompt, 
            'sampling_params': sampling_params, 
            'lora_request': lora_request, 
        })
    
    def run_all_queries(self, ): 
        if not hasattr(self, "query_buffer"):
            self.query_buffer = []
        prompts = []
        sampling_params, lora_request = None, None
        for query in self.query_buffer:
            prompts.append(query['prompt'])
            if sampling_params is None and (sp := query.get('sampling_params', None)): 
                sampling_params = sp
            if lora_request is None and (lr := query.get('lora_request', None)): 
                lora_request = lr
        outputs = self.generate(prompts, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)
        self.query_buffer = []
        return outputs


class HFCustomLLM:

    def __init__(self, model_name_or_path, lora_adapter_path = None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            output_scores=True,  # ✅ 必须开启，否则不返回 scores
            return_dict_in_generate=True,  # ✅ 使 generate 返回字典
        )
        if lora_adapter_path: 
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=None): 
        """ Follow the interface of vllm.LLM.generate
        Implement the most navie version of LLM decoding. 
        * `prompts` can be a single str or a list of str. Batched generation haven't been tested, so do not use it. 
        * `lora_request` and `use_tqdm` will not be used
        * `temperature` and `top_p` will not be used
        """
        temperature = sampling_params.temperature
        top_p = sampling_params.top_p
        max_tokens = sampling_params.max_tokens
        skip_special_tokens = sampling_params.skip_special_tokens
        num_logprobs = sampling_params.logprobs

        model = self.model
        tokenizer = self.tokenizer
        device = model.device

        model.eval()

        if isinstance(prompts, str): 
            prompts = [ prompts ]
        assert len(prompts) == 1

        input_ids = torch.tensor(
            [ tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt)) for prompt in prompts ]
        ).to(device)  # (bsz, input_len)
        attention_mask = input_ids.new_ones(input_ids.shape)
        inputs = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
        }
        bsz, input_len = input_ids.shape

        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False, 
                return_dict_in_generate=True,
                output_scores=True, 
                eos_token_id=tokenizer.eos_token_id,
            )

        scores = gen_output.scores  # List[tensor], length == new tokens, each tensor shape: (batch, vocab_size)
        scores_t = torch.cat(list( sc.unsqueeze(1) for sc in scores ), dim=1)  # (bsz, gen_len, vocab_size) 这里假定了每个instance生成token的长度均相同
        logprobs_t = torch.log_softmax(scores_t, dim=-1)
        logprobs_dict_list, cumulative_logprob_list = [], []
        for seq, seq_logprobs in zip(gen_output.sequences, logprobs_t): 
            sampled_logprobs = []
            cumulative_logprob = 0.
            gen_seq = seq[input_len:]
            for gen_token_id, logprobs in zip(gen_seq, seq_logprobs):
                logprobs_with_token_id = [ (token_id, logprob) for token_id, logprob in enumerate(logprobs.tolist()) ]
                logprobs_with_token_id = sorted(logprobs_with_token_id, key=lambda x: x[1], reverse=True)[:num_logprobs]
                sampled_logprobs.append({
                    token_id: Logprob(logprob=logprob, rank=idx+1, decoded_token=tokenizer.decode(token_id))
                    for idx, (token_id, logprob) in enumerate(logprobs_with_token_id)
                })
                
                cumulative_logprob += logprobs[gen_token_id]
            logprobs_dict_list.append(sampled_logprobs)
            cumulative_logprob_list.append(cumulative_logprob)

        return [
            RequestOutput(
                request_id=None, 
                prompt=prompt, 
                prompt_token_ids=seq[:input_len],
                prompt_logprobs=None, 
                finished=True, 
                outputs=[
                    CompletionOutput(  # we only return one generated seq for each prompt
                        index=0, 
                        text=tokenizer.decode(seq[input_len:], skip_special_tokens=skip_special_tokens), 
                        token_ids=seq[input_len:], 
                        cumulative_logprob=cumulative_logprob, 
                        logprobs=logprobs, 
                    )
                ],
            ) for prompt, seq, logprobs, cumulative_logprob in zip(prompts, gen_output.sequences, logprobs_dict_list, cumulative_logprob_list)
        ]

