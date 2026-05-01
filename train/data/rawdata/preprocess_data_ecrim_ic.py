import os
import sys
import json
import redis
from tqdm import tqdm
from transformers import AutoTokenizer
from cdre_utils import process_example_ReoS
from pycdre.sbert_wk import SentenceBert


PRETRAINED_MODELS = os.environ['PRETRAINED_MODELS']
NUM_MAX_TOKENS = 512  # total tokens in a filtered text path

st, ed = int(sys.argv[1]), int(sys.argv[2])
gpu_id = sys.argv[3]
split = sys.argv[4]

os.makedirs('CodRED/key2textpath', exist_ok=True)

redisd = redis.Redis(host='localhost', port=6379, decode_responses=True, db=0)
bert_tokenizer = AutoTokenizer.from_pretrained(f'{PRETRAINED_MODELS}/bert-base-cased')
sbert_wk = SentenceBert(f'{PRETRAINED_MODELS}/bert-base-cased', device=f'cuda:{gpu_id}')

src_dataset = f'CodRED/{split}_dataset.json'
trg_dataset = f'CodRED/key2textpath/{split}_dataset_{st}_{ed}.json'
src_dataset = json.load(open(src_dataset))[st: ed + 1]
# os.makedirs(f'logs/{split}')
# log_file = open(f'logs/{split}/{st}_{ed}.log', 'a+')

cached_text_path = {}
for instance in tqdm(src_dataset):
    hid, tid, (doc1, doc2) = instance['h_id'], instance['t_id'], instance['doc']
    text_path, _, _ = process_example_ReoS(
        hid, tid, doc1, doc2, 
        tokenizer=bert_tokenizer, 
        redisd=redisd, 
        sbert_wk=sbert_wk, 
        num_max_tokens=NUM_MAX_TOKENS
    )
    key = '#'.join([hid, tid, doc1, doc2])
    cached_text_path[key] = text_path
json.dump(cached_text_path, open(trg_dataset, 'w'))
# print(f'done: {st} - {ed}')
