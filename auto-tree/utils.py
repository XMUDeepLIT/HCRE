
import re
import os
import uuid
import time
import json
import json5
import traceback
# from simhash import Simhash
import matplotlib.pyplot as plt
# from licloud.openai import LiOpenAI


PRINT_LEN = 60


def read_from_jsonl(filepath):
    objs = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            objs.append(json.loads(line.strip()))
    return objs


def hamming_similarity(word1: str, word2: str) -> float:
    hash1 = Simhash(word1)
    hash2 = Simhash(word2)
    hamming_distance = hash1.distance(hash2)
    return 1 - hamming_distance / 64  # 64位哈希


def print_title(s):
    s_len = len(s)
    total_len = PRINT_LEN
    if (total_len - s_len) % 2 == 1:
        total_len += 1
    l = r = total_len // 2 - 1 - s_len // 2
    print(f"{'='*total_len}\n{'='*l} {s} {'='*r}\n{'='*total_len}")


def print_res(s):
    s_len = len(s)
    total_len = PRINT_LEN + 20
    if (total_len - s_len) % 2 == 1:
        total_len += 1
    l = r = total_len // 2 - 1 - s_len // 2
    print(f"{'*'*l} {s} {'*'*r}")


def extract_code(resp):
    patterns = [
        r'```json([\s\S]*?)```',
        r'```plaintext([\s\S]*?)```',
        r'```text([\s\S]*?)```',
        r'```markdown([\s\S]*?)```',
        r'```([\s\S]*?)```',
        r'([\s\S]*?)'
    ]
    code = None

    for pattern in patterns:
        # match = re.search(pattern, resp, re.DOTALL)
        match = re.search(pattern, resp)
        # match = re.findall(pattern, resp)
        if match:
            # code = match[0].strip()
            code = match.group(1).strip()
            break
    if not code:
        return resp
        # raise ValueError(f'answer not found in reponse: \n{resp}')
    # if '\\n' in code:
    #     code = code.replace('\\n', '\n').replace('\\"', '"').strip()
    return code


def unify_cluster_results(cluster_results):
    for k, v in cluster_results.items():
        if 'count' in v:
            cluster_results[k]['count'] = len(v['api_id_occurrence'])
        else:
            cluster_results[k] = { 'count': len(v['api_id_occurrence']), **v }
    cluster_results = dict(sorted(cluster_results.items(), key=lambda p: p[1]['count'], reverse=True))
    for idx, (k, v) in enumerate(cluster_results.items()):
        if 'id' in v:
            cluster_results[k]['id'] = idx
        else:
            cluster_results[k] = { 'id': idx, **v }
    return cluster_results


def dump_cluster_results(fname, cluster_results):
    with open(fname, 'w', encoding='utf-8') as f:
        result_str = json.dumps(cluster_results, default=list, indent=2, ensure_ascii=False)
        result_str = result_str.replace('\n      ', ' ').replace('\n    ]', ' ]')
        f.write(result_str)


def plot_bar(figsize, x, y, xlabel, ylabel, filename, ytop):
    plt.figure(figsize=figsize)
    plt.bar(x, y, color='#1BA1E2')  
    # plt.title("Cumulative Frequncies of Entities", fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ytop:
        plt.ylim(top=ytop)
    # plt.xscale('log', base=4)
    # plt.xticks(data["#TPN"], labels=data['#TPN'], fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.4)
    # plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"stats/{filename}.stats.png", format='png', bbox_inches='tight', transparent=False, dpi=300)


def unify_model_name(client, model_name):
    model_name = model_name.lower()
    if (not model_name.startswith('azure')) and ('gpt' in model_name or 'o1' in model_name or 'o3' in model_name or 'text-embedding-ada' in model_name):
        return f'azure-{model_name}'.replace('.', '')
    elif (not model_name.startswith('bailian')) and 'qwen' in model_name:
        return f'bailian-{model_name}'.replace('.', '_')
    elif (not model_name.startswith('aws')) and 'claude' in model_name:
        return f'aws-{model_name}'.replace('.', '_')
    
    available_model_names = list(map(lambda x: x['id'], client.models.list().model_dump()['data']))
    if model_name in available_model_names:
        return model_name
    
    # unknown model names, guess the user's intent using edit distances
    eds = [ (m_name, hamming_similarity(model_name, m_name)) for m_name in available_model_names ]
    top3_similar_model_name = sorted(eds, key=lambda x: x[1], reverse=True)[:3]
    raise ValueError(f'unknown model name ({model_name}), did you mean {top3_similar_model_name}?')


# class RemoteModel:

#     def __init__(self, client, model_name, generation_archive_path):
#         self.client = client
#         self.model_name = model_name
#         self.generation_archive_path = generation_archive_path
#         self.generation_archive = (
#             [ json.loads(line.strip()) for line in open(generation_archive_path, encoding='utf-8').readlines() ]
#             if os.path.exists(generation_archive_path) else []
#         )
#         self.generation_recover_idx = 0
#         self.latest_stats = self.generation_archive[-1] if self.generation_archive else None

#     def _do_predict(self, prompt):
#         max_retry = 5
#         exception = None
#         completion = None
#         model_name = unify_model_name(self.client, self.model_name)
#         for i in range(max_retry):
#             try:
#                 completion = LiOpenAI().chat.completions.create(
#                     model=model_name,
#                     messages=[ { "role": "user", "content": prompt } ], 
#                     max_tokens=10000, 
#                     temperature=1.0, 
#                     # max_completion_tokens=10000, 
#                 )
#                 completion = completion.model_dump()
#                 # import pdb
#                 # pdb.set_trace()
#             except Exception as e:
#                 exception = e
#                 traceback.print_exc()
#                 print(f'prompt: \n{prompt}')
#                 print(f'重试第{i+1}次...(最大重试次数{max_retry}次)', flush=True)
#                 time.sleep(5)
#                 continue
#             break
#         if completion is None:
#             raise exception
#         return completion

#     def recover_or_generate_response(self, prompt):
#         self.latest_stats = None
#         if self.generation_recover_idx < len(self.generation_archive):  # recover
#             self.latest_stats = self.generation_archive[self.generation_recover_idx]
#             resp = self.latest_stats['response']
#             self.generation_recover_idx += 1
#             return resp
        
#         # generate
#         t1 = time.time()
#         completion = self._do_predict(prompt)
#         t2 = time.time()

#         resp = completion["choices"][0]['message']['content']
#         stats = {
#             'prompt': prompt, 
#             'response': resp, 
#             'latency': f'{t2 - t1:.6f}s', 
#             'model': completion['model'], 
#             'finish_reason': completion['choices'][0]['finish_reason'], 
#             'usage': completion['usage'], 
#         }
#         with open(self.generation_archive_path, 'a+', encoding='utf-8') as generation_archive_file:
#             generation_archive_file.write(json.dumps(stats, ensure_ascii=False) + '\n')
#         self.latest_stats = stats
#         return resp


def robust_json_loads(s: str):
    err1 =  None
    try:
        res = json.loads(s)
    except Exception as e1:
        err1 = traceback.format_exc()
    if err1 is None:
        return res
    
    err2 = None
    try:
        res = json5.loads(s)
    except Exception as e2:
        err2 = traceback.format_exc()
    if err2 is None:
        return res
    
    raise ValueError(
        f"Loading JSON string failed. \n"
        f"Error from built-in `json` lib: \n{err1}\n"
        f"Error from pypi `json5` lib: \n{err2}\n"
        # f"Error from reload trick: \n{err3}\n"
    )


