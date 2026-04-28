import itertools
from typing import List, Union, Tuple
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.beam_search import (
    BeamSearchOutput, 
    BeamSearchSequence, 
    BeamSearchInstance, 
    get_beam_search_score, 
)
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest


class CustomLLM(LLM):
    """
    Improve the implementation of `beam_search`. 
    """

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
