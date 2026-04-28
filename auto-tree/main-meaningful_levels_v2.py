import os
import time
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
# from retry import retry
from constants_new import *
from utils import extract_code


# os.environ['OPENAI_API_KEY'] = "sk-E474QKtgQrb0hAsz1281Ca46FbB04fA4BbA414Fd9b40A5E4"
# os.environ['OPENAI_BASE_URL'] = "https://api.yesapikey.com/v1"

os.environ['OPENAI_API_KEY'] = "sk-7OAa13ruMqB6EFyu060c35CaBf4c4524BbCc18642c0e6d0d"
os.environ['OPENAI_BASE_URL'] = "https://api.lqqq.cc/v1"


MAX_TREE_DEPTH = 3
# MODEL_NAME = 'o1-2024-12-17'
MODEL_NAME_LIST = [ 'gpt-4o-2024-11-20' ]
MODEL_CONFIG_DICT = { 'temperature': 0 }
MODEL_NAME = MODEL_NAME_LIST[0]


LLM = None


class OpenAILLM: 

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, prompt):
        model_config_dict = MODEL_CONFIG_DICT.copy()
        omni_unsupported_keys = ['temperature']
        global MODEL_NAME
        if 'o1' in MODEL_NAME or 'o3' in MODEL_NAME:
            for key in omni_unsupported_keys:
                del model_config_dict[key]

        max_retry = 2
        exception = None
        completion = None
        for i in range(max_retry):
            try:
                client = OpenAI()
                completion = client.chat.completions.create(
                    seed=42, 
                    model=MODEL_NAME, 
                    messages=[ { "role": "user", "content": prompt } ], 
                    **model_config_dict, 
                ).model_dump()
            except Exception as e:
                exception = e
                time.sleep(5)
                continue
            break
        if completion is None:
            raise exception
        return completion["choices"][0]['message']['content']


def get_missing_and_invalid(tree, all_labels):
    existing = set()
    invalid = set()
    def traverse(tree):
        if isinstance(tree, list):
            for node in tree:
                if node in all_labels:
                    existing.add(node)
                else:
                    invalid.add(node)
        else:
            assert isinstance(tree, dict), f'{type(tree) = }'
            for value in tree.values():
                traverse(value)
    traverse(tree)
    missing = all_labels - existing
    return missing, invalid


def get_pos_path_num(tree):
    cnt = 0
    if isinstance(tree, list):
        cnt += len(set(tree))
    else:
        assert isinstance(tree, dict), f'{type(tree) = }'
        for value in tree.values():
            cnt += get_pos_path_num(value)
    return cnt


def format_tree(tree):
    if isinstance(tree, dict):
        ans = []
        for k, v in tree.items():
            ans.append({
                'name': k.lower().replace('relationships', 'relation')\
                                 .replace('relationship', 'relation')\
                                 .replace('relations', 'relation')\
                                 .replace('relation and attributes', 'relation'), 
                'children': format_tree(v), 
            })
        return ans
    elif isinstance(tree, list):
        return list(map(lambda x: {'name': x}, set(tree)))
    else:
        raise Exception("How's the weather today?")


def dedup(tree): 
    if isinstance(tree, dict): 
        res = {}
        for k, v in tree.items():
            dedupped_v = dedup(v)
            if dedupped_v:
                res[k] = dedupped_v
        return res
    elif isinstance(tree, list):
        return list(set(tree))
    else:
        raise TypeError(f'{type(tree) = }')


def build_cluster_tree(criteria, criterion_examples_list, criterion_explanation_list, level, rel2desc, parent_node):
    global LLM, MODEL_NAME

    if level + 1 >= MAX_TREE_DEPTH:
        return list(rel2desc.keys())
    
    # 1. summarization
    cur_criterion = criteria[level]
    criterion_examples = criterion_examples_list[level]
    criterion_explanation = criterion_explanation_list[level]
    p = f'{P_CRITERION_SUM_PREFIXS[level]}\n\n{P_CRITERION_SUMMARIZATION}'.replace('[CRITERION_NAME]', cur_criterion)\
                                                                          .replace('[CRITERION_EXPLANATION]', criterion_explanation)\
                                                                          .replace('[CRITERION_EXAMPLES]', ', '.join(criterion_examples))\
                                                                          .replace('[PREV_CRITERION_NAME]', criteria[level-1])\
                                                                          .replace('[PREV_CRITERION_INSTANCE]', parent_node)\
                                                                          .replace('[RELATION_WITH_DESC]', json.dumps(rel2desc, indent=2, ensure_ascii=False))
    with open(f'{PROCESS13}/P2-{level}-{parent_node}.txt', 'w', encoding='utf-8') as f:
        f.write(p)
    r_cache = f'{PROCESS13}/R2-{level}-{parent_node}.txt'
    if os.path.exists(r_cache):
        r = open(r_cache, encoding='utf-8').read()
    else:
        r = LLM(p)
        with open(r_cache, 'w', encoding='utf-8') as f:
            f.write(r)
    criterion2desc = json.loads(extract_code(r))
    assert criterion2desc, f'{r_cache = }'
    
    # 2. classification
    clusters = { k: [] for k in criterion2desc.keys() }
    for rel, desc in tqdm(rel2desc.items(), desc=f'{cur_criterion} Classification... (Parent Node={parent_node})'):
        p = P_CRITERION_CLASSIFY.replace('[REL_NAME]', rel)\
                                .replace('[REL_DESC]', desc)\
                                .replace('[CRITERION_NAME]', cur_criterion)\
                                .replace('[CRITERION_INSTANCES]', json.dumps(criterion2desc, indent=2, ensure_ascii=False))
        with open(f'{PROCESS14}/P3-{level}-{parent_node}-{rel.replace("/", "_")}.txt', 'w', encoding='utf-8') as f:
            f.write(p)
        r_cache = f'{PROCESS14}/R3-{level}-{parent_node}-{rel.replace("/", "_")}.txt'
        if os.path.exists(r_cache):
            r = open(r_cache, encoding='utf-8').read()
        else:
            # print(r_cache)
            # import pdb; pdb.set_trace()
            retry_cnt = 0
            while True and retry_cnt < 20: 
                r = LLM(p)
                tmp_related_cri_ins = json.loads(extract_code(r))
                ood_pred_cri_ins = set(tmp_related_cri_ins) - set(criterion2desc.keys())
                for cri_ins in ood_pred_cri_ins:
                    tmp_related_cri_ins.remove(cri_ins)
                # print(tmp_related_cri_ins)
                is_ok = tmp_related_cri_ins and all(criterion2desc.get(cri_ins, False) for cri_ins in tmp_related_cri_ins)
                if is_ok:
                    MODEL_NAME = MODEL_NAME_LIST[0]
                    r = f"```json\n{json.dumps(tmp_related_cri_ins)}\n```"
                    break
                elif retry_cnt >= 10: 
                    MODEL_NAME = MODEL_NAME_LIST[1]
                retry_cnt += 1
            with open(r_cache, 'w', encoding='utf-8') as f:
                f.write(r)
        related_cri_ins = json.loads(extract_code(r))
        assert related_cri_ins and all(criterion2desc.get(cri_ins, False) for cri_ins in related_cri_ins), f'{criterion2desc.keys() = }\n{related_cri_ins = }\n{rel = }'

        for cri_ins in related_cri_ins:
            clusters[cri_ins].append(rel)
    
    # report stats of sum
    missing, invalid = get_missing_and_invalid(clusters, rel2desc.keys())
    print(f'{level}-{parent_node}-sum: {missing = }, {invalid = }')

    # 3. move to the next layer
    tree = {}
    for cri_ins, rels in clusters.items():
        next_level_rel2desc = { rel: rel2desc[rel] for rel in rels }
        tree[cri_ins] = build_cluster_tree(criteria, criterion_examples_list, criterion_explanation_list, level+1, next_level_rel2desc, cri_ins)

    # report stats of cls
    missing, invalid = get_missing_and_invalid(tree, rel2desc.keys())
    print(f'{level}-{parent_node}-cls: {missing = }, {invalid = }')

    return tree


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    
    rel2desc = json.load(open('rel2desc-codred.json', encoding='utf-8'))
    
    global PROCESS12, PROCESS13, PROCESS14
    # PROCESS12 = os.path.join(f"results-{args.exp_name}", PROCESS12)
    PROCESS13 = os.path.join(f"results-{args.exp_name}", PROCESS13)
    PROCESS14 = os.path.join(f"results-{args.exp_name}", PROCESS14)
    os.makedirs(PROCESS12, exist_ok=True)
    os.makedirs(PROCESS13, exist_ok=True)
    os.makedirs(PROCESS14, exist_ok=True)

    global LLM
    LLM = OpenAILLM(args.seed)
    
    # 1. criteria generation
    P1 = P_CRITERION_GENERATION
    with open(f'{PROCESS12}/P1.txt', 'w', encoding='utf-8') as f:
        f.write(P1)
    if os.path.exists(f'{PROCESS12}/R1.txt'):
        R1 = open(f'{PROCESS12}/R1.txt').read()
    else:
        R1 = LLM(P1)
        with open(f'{PROCESS12}/R1.txt', 'w', encoding='utf-8') as f:
            f.write(R1)
    R1 = extract_code(R1)
    criterion_res = json.loads(R1)
    criteria = criterion_res[f'top{MAX_TREE_DEPTH-1} criteria']
    criterion_examples_list = [ criterion_res['clustering criteria'][cri]['possible cluster names'] for cri in criteria ]
    criterion_explanation_list = [ criterion_res['clustering criteria'][cri]['explanation'] for cri in criteria ]
    print(f'{criteria = }')
    # import pdb; pdb.set_trace()

    # 2. divisive clustering
    tree = build_cluster_tree(criteria, criterion_examples_list, criterion_explanation_list, 0, rel2desc, 'POS')
    
    # 3. post-process
    tree = dedup(tree)
    ppn = get_pos_path_num(tree)

    final_tree = [ { 'name': 'no valid relation' }, { 'name': 'valid relations', 'children': format_tree(tree) } ]
    json.dump(tree, open(f'results-{args.exp_name}/tree.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(final_tree, open(f'results-{args.exp_name}/relation_tree_name.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'done. tpn = {ppn + 1}')


if __name__ == "__main__":
    main()
