import os
import json
import math
import shutil
import string
import logging
from typing import List, Literal, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import AnyTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NAME2TEMPLATE = {
    "empty": {
        "prefix": "", 
        "suffix": "", 
    }, 
    "llama3": {
        "prefix": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", 
        "suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
    }, 
    "qwen": {
        "prefix": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n", 
        "suffix": "<|im_end|>\n<|im_start|>assistant\n", 
    },
    "gemma": {
        "prefix": "<bos><start_of_turn>user\n", 
        "suffix": "<end_of_turn>\n<start_of_turn>model\n", 
    }
}


def generate_letter_serial_number():
    res = list(string.ascii_uppercase)
    for ch1 in string.ascii_uppercase:
        for ch2 in string.ascii_uppercase:
            res.append(ch1+ch2)
    return res


def int2roman(n):
    roman_numerals = {
        1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X',
        40: 'XL', 50: 'L', 90: 'XC', 100: 'C',
        400: 'CD', 500: 'D'
    }
    result = ""
    for value in sorted(roman_numerals.keys(), reverse=True):
        while n >= value:
            result += roman_numerals[value]
            n -= value
    return result


SERIAL_NUMS = {
    'letter': generate_letter_serial_number(),  # 26 + 26 * 26
    'number': list(map(str, range(1, 500))), 
    'roman': [ int2roman(i) for i in range(1, 500) ], 
    'special_token': [f'<|{i}|>' for i in range(1, 500)], 
}


def mean(lst):
    return sum(lst) / len(lst)


def edit_distance(word1: str, word2: str) -> int:
    """
    Author: Leetcode
    Source：https://leetcode.cn/problems/edit-distance/solutions/188223/bian-ji-ju-chi-by-leetcode-solution/
    """
    n = len(word1)
    m = len(word2)
    
    if n * m == 0:
        return n + m
    
    D = [ [0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1] 
            if word1[i - 1] != word2[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)
    
    return D[n][m]


def argmax(x, y, key=None):
    if key is None:
        key = lambda z: z
    return x if key(x) > key(y) else y


def add_args(parser):
    # vllm config
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--preprocessing_num_workers', type=int, required=True)
    parser.add_argument('--lora_adapter_path', type=str, default=None)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--eval_dataset', type=str, required=True)
    parser.add_argument('--cutoff_len', type=int, required=True)
    parser.add_argument('--max_new_tokens', type=int, required=True)
    parser.add_argument('--top_p', type=float, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--debug', action="store_true", default=False)
    # exp settings
    ## tree inference exp
    parser.add_argument('--multistep_gen', action='store_true', default=False)
    parser.add_argument('--tree_instance', type=str, default=None)
    parser.add_argument('--original_prompt', action="store_true", default=False)
    parser.add_argument('--no_prev', action="store_true", default=False)
    parser.add_argument('--na_type', type=int, required=True)
    ## verification exp (only works when --multistep_gen is set)
    parser.add_argument('--verification_methods', type=str, default=None)
    parser.add_argument('--suboptimal_node_type', type=str, default="random")
    ## label reduction exp
    parser.add_argument('--num_options', type=int, default=None)
    parser.add_argument('--num_lrd_options', type=int, default=-1)
    ## serial_num exp
    parser.add_argument('--serial_num_type', type=str, choices=['letter', 'number', 'roman', 'special_token', 'uni_special_token'], default=None)
    parser.add_argument('--marker_name', type=str, default=None)  # required only if serial_num_type == 'uni_special_token'
    return parser


def build_dataset(args) -> List[Dict[str, str | int | None]]:
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset_info = json.load(open(os.path.join(args.dataset_dir, 'dataset_info.json')))
    eval_dataset_file = dataset_info[args.eval_dataset]['file_name']
    with open(eval_dataset_file) as f:
        raw_data = json.load(f)
    return raw_data

