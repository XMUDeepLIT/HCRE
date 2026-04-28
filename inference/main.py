import os
import re
import json
import time
import math
import random
import logging
import warnings
import multiprocessing
from copy import deepcopy
from typing import Optional, List, Union, Tuple, Dict, Any
from argparse import ArgumentParser
from functools import reduce, partial
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams, BeamSearchParams

from utils import *
from prompts import *
from label_tree import LabelTree
from custom_llm import CustomLLM, HFCustomLLM
from trie import PredictTrie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False
BS_OOD_CNT = 0
MARKER = {}

#### Util Functions ####

def debug_print(s: str):
    global DEBUG
    if DEBUG:
        print(s)


def get_option_score_dict(args, output, serial_number_type, options):
    # import pdb; pdb.set_trace()
    snum2score = {}
    first_logprob_dict = output.logprobs[0]
    for token_id, logprob in first_logprob_dict.items(): 
        snum = logprob.decoded_token.strip()
        snum2score[snum] = logprob.logprob
    option_score_dict = {}
    for index, opt in enumerate(options):
        if serial_number_type == 'uni_special_token':
            assert MARKER
            snum = f"<|{MARKER[opt]}|>"
        else:
            snum = SERIAL_NUMS[args.serial_num_type][index]
        option_score_dict[opt] = snum2score.get(snum, -math.inf)
    return option_score_dict


def serialize_opt(index, option):
    if args.serial_num_type is None:
        return option
    
    if args.serial_num_type == 'uni_special_token':
        assert MARKER
        return f'<|{MARKER[option]}|>{option}'
    
    serial_num = SERIAL_NUMS[args.serial_num_type][index]
    if args.serial_num_type in ['number', 'letter', 'roman']:
        return f'{serial_num}. {option}'
    elif args.serial_num_type in ['special_token']:
        return f'{serial_num}{option}'
    else:
        raise NotImplementedError(f'Invalid serial_num_type = {args.serial_num_type}')


def get_prompt(args, prompt_type, eh, et, context, options, prev_nodes):
    if args.original_prompt:
        plain_prompt = PLAIN_PROMPT_ORIGIN
        ms_prompt = MS_PROMPT_NO_PREV_ORIGIN if args.no_prev else MS_PROMPT_ORIGIN
    else:
        plain_prompt = PLAIN_PROMPT
        ms_prompt = MS_PROMPT_NO_PREV if args.no_prev else MS_PROMPT
    prompt_templates = { 'plain': plain_prompt, 'ms': ms_prompt }
    serialized_options = [ serialize_opt(idx, opt) for idx, opt in enumerate(options)]
    
    return prompt_templates[prompt_type].replace('[HEAD_ENTITY]', eh)\
                               .replace('[TAIL_ENTITY]', et)\
                               .replace('[OPTIONS]', ', '.join(serialized_options))\
                               .replace('[PREV_NODES]', ' > '.join(prev_nodes))\
                               .replace('[CONTEXT]', context)


def llm_predict(
    llm: CustomLLM, 
    prompt_type: str, 
    reason_chain: List[str], 
    options: List[str], 
    input_params: Dict[str, Union[str, SamplingParams, LoRARequest, AutoTokenizer]],
) -> Tuple[str, float]:
    input_prompt = get_prompt(
        args, prompt_type, 
        input_params['eh'], 
        input_params['et'], 
        input_params['text'], 
        options, 
        reason_chain, 
    )
    debug_print(f"Prompt: \n{input_prompt}")
    output = llm.generate(
        input_params['template_prefix'] + input_prompt + input_params['template_suffix'], 
        sampling_params=input_params['sampling_params'], 
        lora_request=input_params['lora_request'], 
        use_tqdm=False, 
    )[0].outputs[0]
    generated_sequence = output.text.strip()
    confidence = output.cumulative_logprob / len(output.logprobs)
    return generated_sequence, confidence


def register_v_predict(
    llm: CustomLLM, 
    prompt_type: str, 
    reason_chain: List[str], 
    options: List[str], 
    input_params: Dict[str, Union[str, SamplingParams, LoRARequest, AutoTokenizer]],
) -> Tuple[str, float]:
    input_prompt = get_prompt(
        args, prompt_type, 
        input_params['eh'], 
        input_params['et'], 
        input_params['text'], 
        options, 
        reason_chain, 
    )
    llm.register_query(
        input_params['template_prefix'] + input_prompt + input_params['template_suffix'], 
        sampling_params=input_params['sampling_params'], 
        lora_request=input_params['lora_request'], 
    )


def llm_trie_predict(
    llm: CustomLLM, 
    prompt_type: str, 
    reason_chain: List[str], 
    options: List[str], 
    input_params: Dict[str, Union[str, SamplingParams, LoRARequest, AutoTokenizer]],
) -> Dict[str, float]: 
    input_prompt = get_prompt(
        args, prompt_type, 
        input_params['eh'], 
        input_params['et'], 
        input_params['text'], 
        options, 
        reason_chain, 
    )
    trie = PredictTrie(input_params['tokenizer'])
    results = trie.predict(
        llm=llm, 
        prompt=input_params['template_prefix'] + input_prompt + input_params['template_suffix'], 
        labels=options, 
        method='trie', 
        llm_generate_config={
            'sampling_params': input_params['sampling_params'],
            'lora_request': input_params['lora_request'], 
        }
    )
    return results

#### PLAIN ####

def direct_predict(args, llm, tokenizer, instance, sampling_params, lora_request):
    template = NAME2TEMPLATE[args.template]
    input_params = {
        "eh": instance['eh'], 
        "et": instance['et'], 
        "text": instance['context'], 
        'template_prefix': template['prefix'], 
        'template_suffix': template['suffix'], 
        'sampling_params': sampling_params, 
        'lora_request': lora_request, 
    }

    # get options
    ALL_POSSIBLE_NA = [
        ('no valid relation', 'valid relations'), 
        ('not available', 'available'), 
        ('invalid relation', 'valid relation'), 
        ('no', 'yes'), 
    ]
    all_relations = (
        json.load(open('data/rawdata/CodRED/labels.json'))
        if not args.docred
        else list(map(lambda x: x['rel_name'], json.load(open('data/rawdata/relations_with_desc-docred.json'))))
    )
    options = [ ALL_POSSIBLE_NA[args.na_type][0], *all_relations[1:] ]
    if args.num_lrd_options != -1:
        assert not args.docred  # not implemented
        assert args.num_options is None
        options = list(json.load(open('data/rawdata/rel_sample_priority.json')).keys())[:args.num_lrd_options]
    if args.num_options and args.num_options < 277:
        assert not args.docred  # not implemented
        assert args.num_lrd_options == -1

        rel_objs = json.load(open('data/rawdata/relations_with_desc.json'))
        rel_rank  = { ALL_POSSIBLE_NA[args.na_type][0]: 300 }
        pid2label = { 'n/a': ALL_POSSIBLE_NA[args.na_type][0] }
        for idx, rel in enumerate(rel_objs):
            if rel['rel_id'] == 'n/a': continue
            pid2label[rel['rel_id']] = rel['rel_name']
            rel_rank[rel['rel_name']] = idx
        label = pid2label[instance['label']]  # get linearized label string

        # sample options
        label_idx = options.index(label)
        options = random.sample(options[:label_idx] + options[label_idx+1:], k=args.num_options-1) + [ label ]
        options = sorted(options, key=lambda r: rel_rank[r])
        # print(f'{options = }\n{label = }')
        # random.shuffle(options)
    
    generated, conf = llm_predict(llm, 'plain', [], options, input_params)

    agg_rel_dist = None
    debug_print(f"Response: {generated}")
    return generated, conf, agg_rel_dist

#### MS ####

def prelimarily_predict(
    llm: CustomLLM, 
    labeltree: LabelTree, 
    reason_chain: List[str], 
    input_params: Dict[str, Union[str, SamplingParams, LoRARequest, AutoTokenizer]],
    use_bs: bool, 
) -> Tuple[str, str, float, float]:
    options = labeltree.children_of(reason_chain)
    if not use_bs: 
        best_node, best_conf = llm_predict(llm, 'ms', reason_chain, options, input_params)
        return best_node, None, best_conf, None

    input_prompt = get_prompt(
        args, 'ms', 
        input_params['eh'], 
        input_params['et'], 
        input_params['text'], 
        options, 
        reason_chain, 
    )
    bs_params = BeamSearchParams(
        beam_width=5,  # large beam size to avoid ood predictions
        max_tokens=input_params['sampling_params'].max_tokens, 
        temperature=input_params['sampling_params'].temperature, 
    )
    outputs = llm.beam_search(
        input_params['template_prefix'] + input_prompt + input_params['template_suffix'], 
        params=bs_params, 
        lora_request=input_params['lora_request'], 
        skip_special_tokens=True, 
    )[0].sequences
    generated_nodes, confs = [], []
    for output in outputs:
        if output.text in options:
            generated_nodes.append(output.text)
            confs.append(output.cum_logprob / len(output.logprobs))
    if len(generated_nodes) < 2:
        msg = f"Options: {options}\nValid nodes: {generated_nodes}\nAll output sequences: "
        for i, output in enumerate(outputs):
            msg += f"[{i}]: text={repr(output.text)} tokens={output.tokens[-len(output.logprobs):]}\n"
        warnings.warn(f"Hell no! Unable to get two nodes!\n{msg}")
        global BS_OOD_CNT; BS_OOD_CNT += 1
        for output in outputs:
            if output.text not in generated_nodes:
                generated_nodes.append(output.text)
                confs.append(output.cum_logprob / len(output.logprobs))
    return *generated_nodes[:2], *confs[:2]


def verification_predict(
    llm: CustomLLM, 
    labeltree: LabelTree, 
    reason_chain: List[str], 
    nodes_to_be_replaced: List[str], 
    original_options: List[str], 
    input_params: Dict[str, Union[str, SamplingParams, LoRARequest, AutoTokenizer]], 
):
    new_options = original_options.copy()
    for node in nodes_to_be_replaced:
        sub_nodes = labeltree.children_of(reason_chain + [node])
        if sub_nodes and node in new_options:
            node_idx = new_options.index(node)
            new_options = new_options[:node_idx] + sub_nodes + new_options[node_idx+1:]
    register_v_predict(llm, 'ms', reason_chain, new_options, input_params)


class VerifyMethod:

    def __init__(self, llm, labeltree, reason_chain, best_node, suboptimal_node, input_params):
        self.llm = llm
        self.labeltree = labeltree
        self.reason_chain = reason_chain
        self.best_node = best_node
        self.suboptimal_node = suboptimal_node
        self.input_params = input_params

    def register_verification(self, ):
        raise NotImplementedError("Implement the method in subclasses. ")
    
    def verify_pred(self, v_pred):
        raise NotImplementedError("Implement the method in subclasses. ")


class BestVerify(VerifyMethod):

    def register_verification(self, ):
        labeltree = self.labeltree
        reason_chain = self.reason_chain
        best_node = self.best_node
        original_options = labeltree.children_of(reason_chain)
        if best_node not in original_options:  # OOD
            debug_print(f"best_verify: OOD\n\t{best_node = }\n\t{original_options = }")
            return False
        sub = labeltree.children_of(reason_chain + [best_node])
        if sub == []: 
            return True
        verification_predict(self.llm, labeltree, reason_chain, [best_node], original_options, self.input_params)
        self.sub = sub
    
    def verify_pred(self, v_pred):
        return v_pred in self.sub


class SuboptimalVerify(VerifyMethod):

    def register_verification(self, ):
        labeltree = self.labeltree
        reason_chain = self.reason_chain
        best_node = self.best_node
        suboptimal_node = self.suboptimal_node
        original_options = labeltree.children_of(reason_chain)
        if best_node not in original_options:  # OOD
            debug_print(f"suboptimal_verify: OOD\n\t{best_node = }\n\t{suboptimal_node = }\n\t{original_options = }")
            return False
        if suboptimal_node not in original_options: 
            return True
        sub = labeltree.children_of(reason_chain + [suboptimal_node])
        if sub == []: 
            return True
        verification_predict(self.llm, labeltree, reason_chain, [suboptimal_node], original_options, self.input_params)
    
    def verify_pred(self, v_pred):
        return v_pred == self.best_node
    

class DoubleVerify(VerifyMethod):

    def register_verification(self, ):
        labeltree = self.labeltree
        reason_chain = self.reason_chain
        best_node = self.best_node
        suboptimal_node = self.suboptimal_node
        original_options = labeltree.children_of(reason_chain)
        if best_node not in original_options:  # OOD
            debug_print(f"double_verify: OOD\n\t{best_node = }\n\t{suboptimal_node = }\n\t{original_options = }")
            return False
        bsub = labeltree.children_of(reason_chain + [best_node])
        ssub = labeltree.children_of(reason_chain + [suboptimal_node])
        if bsub == [] and ssub == []:
            return True
        verification_predict(self.llm, labeltree, reason_chain, [best_node, suboptimal_node], original_options, self.input_params)
        self.bsub = bsub
        
    def verify_pred(self, v_pred):
        return v_pred in self.bsub


def do_verify(args, llm, labeltree, reason_chain, best_node, suboptimal_node, input_params) -> bool:
    if not args.verification_methods:
        return True
    
    VERIFICATION_METHODS: Dict[str, VerifyMethod] = {
        'best': BestVerify, 
        'suboptimal': SuboptimalVerify, 
        'double': DoubleVerify, 
    }
    v_methods, v_results = [], []
    for v_method_id in args.verification_methods:
        if v_method_id not in VERIFICATION_METHODS:
            raise ValueError(f"Invalid Verfication Func ID. Only support: {list(VERIFICATION_METHODS.keys())}")
        v_method = VERIFICATION_METHODS[v_method_id](llm, labeltree, reason_chain, best_node, suboptimal_node, input_params)
        v_results.append(v_method.register_verification())
        v_methods.append(v_method)

    outputs = llm.run_all_queries()
    j = 0
    for idx, (v_method, v_res) in enumerate(zip(v_methods, v_results)):
        if v_res is None: 
            v_pred = outputs[j].outputs[0].text.strip()
            v_results[idx] = v_method.verify_pred(v_pred)
            j += 1
    
    assert len(v_results) % 2 == 1
    return sum(v_results) > len(v_results) / 2


def multi_step_predict(args, llm, tokenizer, instance, sampling_params, lora_request):

    labeltree = LabelTree(args.tree_instance, internal_node_strategy='echo')
    template = NAME2TEMPLATE[args.template]
    input_params = {
        "eh": instance['eh'], 
        "et": instance['et'], 
        "text": instance['context'], 
        'template_prefix': template['prefix'], 
        'template_suffix': template['suffix'], 
        'sampling_params': sampling_params, 
        'lora_request': lora_request, 
        'tokenizer': tokenizer, 
    }

    reason_chain, reason_confs = [], []
    total_steps = labeltree.max_depth
    prompts_for_cls = []
    for step in range(total_steps):  # 4 steps in total
        options = labeltree.children_of(reason_chain)
        original_options = options.copy()

        v_pass = False
        ptv_cnt = 0
        while not v_pass:
            ptv_cnt += 1

            if len(options) == 1:
                best_node = options[0]
                break

            best_node, secondbest_node, best_node_conf, secondbest_node_conf = prelimarily_predict(
                llm, labeltree, reason_chain, input_params, 
                use_bs=(args.suboptimal_node_type == 'bs' and args.verification_methods != ['best'])
            )

            if args.suboptimal_node_type == 'bs':
                suboptimal_node = secondbest_node
            elif args.suboptimal_node_type == 'random':
                suboptimal_node = random.choice([node for node in options if node != best_node])
            elif args.suboptimal_node_type == 'edit':
                suboptimal_node = min([node for node in options if node != best_node], key=lambda node: edit_distance(node, best_node))
            elif args.suboptimal_node_type is None: 
                suboptimal_node = None
            else:
                raise ValueError(f"Invalid suboptimal_node_type={args.suboptimal_node_type}, only support bs/random/edit/None")
            
            v_pass = do_verify(args, llm, labeltree, reason_chain, best_node, suboptimal_node, input_params)
            if ptv_cnt > 3: 
                break
            if not v_pass and best_node in options:
                options.remove(best_node)

        reason_chain.append(best_node)
        reason_confs.append(best_node_conf)

        if best_node not in original_options:  # OOD node
            break

        if best_node == labeltree.na_node:
            break
    
    agg_rel_dist = None
    return ' -> '.join(reason_chain), mean(reason_confs), agg_rel_dist


def worker(kwargs):
    try:
        # args
        args = kwargs['args']
        data = kwargs['data']
        device = kwargs['device']
        if DEBUG:
            if args.docred: 
                data = random.sample(
                    [d for d in kwargs['data'] if d['label'] != ['no valid relation']], 
                    k=50
                ) + random.sample(
                    [d for d in kwargs['data'] if d['label'] == ['no valid relation']], 
                    k=50
                )
            else: 
                data = random.sample(
                    [d for d in kwargs['data'] if d['label'] != 'n/a'], 
                    k=50
                ) + random.sample(
                    [d for d in kwargs['data'] if d['label'] == 'n/a'], 
                    k=50
                )
        
        # envs
        os.environ['CUDA_VISIBLE_DEVICES'] = device

        # prepare model and generation config
        if args.model_backend == 'vllm': 
            additional_kwargs = {}
            llm = CustomLLM(
                seed=args.seed,
                model=args.model_name_or_path, 
                trust_remote_code=True, 
                tensor_parallel_size=1, 
                gpu_memory_utilization=0.8, 
                max_model_len=args.cutoff_len, 
                enforce_eager=False,
                enable_lora=bool(args.lora_adapter_path),
                max_lora_rank=64,
                enable_prefix_caching=True, 
                **additional_kwargs, 
            )
        elif args.model_backend == 'hf': 
            llm = HFCustomLLM(
                model_name_or_path=args.model_name_or_path, 
                lora_adapter_path=args.lora_adapter_path, 
            )
        else:
            raise NotImplementedError(f"Invalid model_backend = {args.modelbackend}, must be one of hf/vllm")
        sampling_params = SamplingParams(
            seed=args.seed, 
            temperature=args.temperature, 
            top_p=args.top_p, 
            max_tokens=args.max_new_tokens, 
            logprobs=20, 
            skip_special_tokens=True, 
        )
        lora_request = LoRARequest('lora', 1, lora_path=args.lora_adapter_path) if args.lora_adapter_path else None
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

        # prepare inference method
        inference_one_instance = partial(
            multi_step_predict if args.multistep_gen else direct_predict, 
            args=args, llm=llm, 
            tokenizer=tokenizer, 
            sampling_params=sampling_params, 
            lora_request=lora_request, 
        )

        # do generate!
        preds = []
        dump_idx = 0
        for i, entry in enumerate(tqdm(data, desc=f'Running inference on gpu {device}')):
            t1 = time.time()
            pred, logprob, agg_rel_dist = inference_one_instance(instance=entry)
            # if DEBUG: import pdb; pdb.set_trace()
            t2 = time.time()
            preds.append({
                **entry,  # id, eh, et, context, label
                "predict": pred, 
                "confidence": logprob, 
                "h_dist": [ agg_rel_dist ], 
                "latency": t2 - t1, 
            })
            if args.debug:
                input()
            if len(preds) > len(data) // 10:
                logger.info(f"Dumping temp predictions to disks to avoid OOM...")
                json.dump(preds, open(f"{args.output_dir}/{device}.temp_preds.{dump_idx}.json", 'w'))
                dump_idx += 1
                del preds
                preds = []
        torch.cuda.empty_cache()
        json.dump(llm.stats, open(f"{args.output_dir}/{device}.stat{'-debug' if args.debug else ''}.json", 'w'))
        return preds
    except:
        import traceback
        err_msg = traceback.format_exc()
        with open(f"{args.output_dir}/eval.GPU{device}{'-debug' if args.debug else ''}.err", 'w', encoding='utf-8') as f:
            f.write(err_msg)


def main(args):
    data = build_dataset(args)
    random.shuffle(data)
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    device_cnt = len(devices)
    data_per_device_len = math.ceil(len(data) / device_cnt)
    with multiprocessing.Pool(device_cnt) as pool:
        worker_arguments = [
            {
                "args": args, 
                "data": data[
                    i * data_per_device_len: 
                    min((i+1) * data_per_device_len, len(data))
                ], 
                "device": device
            } for i, device in enumerate(devices)
        ]
        all_preds = (
            pool.map(worker, worker_arguments)
            if not DEBUG else [ worker(worker_arguments[0]) ]
        )

    has_error = False
    try: 
        for f in os.listdir(args.output_dir):
            if bool(re.match(r'^\d+\.temp_preds\.\d+\.json$', f)):
                tmp = json.load(open(os.path.join(args.output_dir, f)))
                all_preds.append(tmp)
                del tmp
    except Exception as e:
        import traceback; traceback.print_exc()
        has_error = True

    all_preds = sorted(reduce(lambda x, y: x+y, all_preds, []), key=lambda p: p['id'])
    logger.debug(f'{len(all_preds) = }')
    with open(f"{args.output_dir}/generated_predictions{'-debug' if args.debug else ''}.jsonl", "w", encoding="utf-8") as f_pred:
        for entry in all_preds:
            f_pred.write(json.dumps(entry) + "\n")

    if not has_error:
        for f in os.listdir(args.output_dir): 
            if bool(re.match(r'^\d+\.temp_preds\.\d+\.json$', f)):
                os.remove(os.path.join(args.output_dir, f))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    # args checking
    if args.tree_instance:
        assert str(args.na_type) == re.search(r'NA(\d+)/', args.tree_instance).group(1)
    if args.num_options:
        assert not args.multistep_gen, f'option reduction experiments must be done in directly predict setting. '
        assert args.num_lrd_options == -1, f'label reduction data and label reduction cannot be applied at the same time. '
    assert not (bool(args.verification_methods) ^ bool(args.suboptimal_node_type)), \
        "verification_methods and suboptimal_node_type must be set or unset at the same time"
    
    # args post-processing
    args.output_dir = os.path.realpath(args.output_dir)
    args.dataset_dir = os.path.realpath(args.dataset_dir)
    args.tree_instance = os.path.realpath(args.tree_instance) if args.tree_instance else None
    args.lora_adapter_path = os.path.realpath(args.lora_adapter_path) if args.lora_adapter_path else None
    args.model_name_or_path = os.path.realpath(args.model_name_or_path)
    args.verification_methods = args.verification_methods.split('.') if args.verification_methods else []
    if len(args.verification_methods) > 0 and len(args.verification_methods) % 2 == 0:
        raise ValueError("The verification num must be an odd number to vote. ")

    # log exp settings
    DEBUG = args.debug
    if DEBUG:
        logger.info('\n' + '\n'.join([ '!!!!!  DEBUGGING DEBUGGING DEBUGGING  !!!!!' ] * 3))
    logger.info(f'=========== Exp Arguments ============')
    for k, v in args.__dict__.items():
        logger.info(f'{k}: {str(v)}')
    logger.info(f'======================================')
    
    # prepare serial numbers
    if args.marker_name:
        dataset_info = json.load(open('data/dataset_info.json'))
        MARKER = json.load(open(dataset_info[args.marker_name]['file_name']))

    # prepare checkpoints if new special tokens are added
    if args.serial_num_type and 'special_token' in args.serial_num_type:
        logger.info('Start parsing checkpoints before model loading...')
        args.model_name_or_path, args.lora_adapter_path = parse_checkpoint(
            pretrained_model_dir=args.model_name_or_path, 
            adapter_dir=args.lora_adapter_path, 
            target_dir=os.path.join(args.output_dir, 'merged_checkpoints'), 
        )

    set_seed(args.seed)

    start_time = time.time()
    main(args)
    end_time = time.time()
    logger.info(f'Consuming time: {end_time - start_time}s')

    print(f"{BS_OOD_CNT = }")
