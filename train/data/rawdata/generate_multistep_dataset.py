import os
import json
import redis
import string
import random
import logging
import argparse
from dataclasses import field
from tqdm import tqdm
from transformers.data.data_collator import *
from transformers import BertTokenizer, BertModel, AutoTokenizer, set_seed
from cdre_utils import process_example_ReoS
from pycdre.sbert_wk import SentenceBert
from label_tree import LabelTree
from prompts import MS_PROMPT, MS_PROMPT_ORIGIN, MS_PROMPT_NO_PREV, MS_PROMPT_NO_PREV_ORIGIN

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

ALL_NA_TYPE = [
    ('no valid relation', 'valid relations'), 
    ('not available', 'available'), 
    ('invalid relation', 'valid relation'), 
    ('no', 'yes'), 
]


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

parser = argparse.ArgumentParser()
parser.add_argument('--tree_name', type=str, required=True)
parser.add_argument(
    '--substitute', type=str, 
    choices=['best', 'suboptimal', 'double', 'best,suboptimal', 'best,suboptimal,double'], 
    default=None, 
)
parser.add_argument('--subsets', type=str, default="train,dev")
parser.add_argument('--original_prompt', action="store_true", default=False)
parser.add_argument('--no_prev', action="store_true", default=False)
parser.add_argument('--na_type', type=int, required=True)
parser.add_argument('--serial_num_type', type=str, choices=['letter', 'number', 'roman', 'special_token', 'uni_special_token'], default=None)
args = parser.parse_args()

# skipping = ['CodRED_na_changed+dsre', 'CodRED_small', 'CodRED', 'CodRED_only_pos', 'CodRED_small_na_changed', 'CodRED_small_only_pos']
skipping = []
TRAIN_SRC = '../../../train'  # XXX: Please modify the variable to the

IC_STRATEGY = 'ecrim'
tree_name = f'NA{args.na_type}/{args.tree_name}'
internal_node_strategy = 'echo'
SUBSTITUTE = args.substitute.split(',') if args.substitute else []
subsets = args.subsets.split(',')
NA, POS = ALL_NA_TYPE[args.na_type]

ORIGIN = args.original_prompt
NOPREV = args.no_prev
if args.original_prompt:
    PROMPT_TEMPLATE = MS_PROMPT_NO_PREV_ORIGIN if args.no_prev else MS_PROMPT_ORIGIN
else:
    PROMPT_TEMPLATE = MS_PROMPT_NO_PREV if args.no_prev else MS_PROMPT


def kmp(src: list, pattern: list) -> int:
    n, m = len(src), len(pattern)

    # initilize `nxt` 
    nxt = [0 for _ in range(m)]
    left = 0
    for right in range(1, m):
        while left > 0 and pattern[left] != pattern[right]:
            left = nxt[left - 1]
        if pattern[left] == pattern[right]:
            left += 1
        nxt[right] = left
    
    # do matching!
    j = 0
    for i in range(n):
        while j > 0 and src[i] != pattern[j]:
            j = nxt[j - 1]
        if src[i] == pattern[j]:
            j += 1
        if j == m:
            return i - j + 1
    return -1


def expand(start, end, total_len, max_size):
    e_size = max_size - (end - start)
    _1 = start - (e_size // 2)
    _2 = end + (e_size - e_size // 2)
    if _2 - _1 <= total_len:
        if _1 < 0:
            _2 -= _1
            _1 = 0
        elif _2 > total_len:
            _1 -= (_2 - total_len)
            _2 = total_len
    else:
        _1 = 0
        _2 = total_len
    return _1, _2


def get_entity_by_id(entity_id, doc):
    for entity in doc['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == entity_id:
            return entity
    raise ValueError(f'entity (id={entity_id}) not found in {doc["title"]}')


@dataclass
class DataCollatorForCDRE:
    redisd: redis.Redis
    sbert_wk: SentenceBert
    bert_tokenizer: BertTokenizer
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    ic_strategy: str = "ecrim"
    key2textpath: dict = field(default_factory=dict)
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    num_examples: int = 0
    input_record_file: str = None
    cur_save_time: int = 0

    def _ecrim_ic(self, hid, tid, doc1, doc2, num_max_tokens):
        # print(f'{num_max_tokens = }')
        # import pdb
        # pdb.set_trace()
        return process_example_ReoS(
            hid, tid, doc1, doc2, 
            tokenizer=self.bert_tokenizer, 
            redisd=self.redisd, 
            sbert_wk=self.sbert_wk, 
            num_max_tokens=num_max_tokens
        )

    def _snippet_doc(self, doc_string, entity, num_max_tokens):
        def process_text(text): 
            all_chars = set(text)
            for ch in all_chars:
                if ch.isalnum():
                    continue
                text = text.replace(ch, ' ' + ch + ' ')
            text = ' '.join(text.split())
            text = text.replace(' .', '.')   \
                       .replace(' ,', ',')   \
                       .replace(' !', '!')   \
                       .replace(' ?', '?')   \
                       .replace('..', '. .') \
                       .replace('.,', '. ,') \
                       .replace('.!', '. !') \
                       .replace('.?', '. ?')
            return text

        # preprocess and tokeization
        processed_doc = process_text(doc_string)
        processed_entity = process_text(entity)
        tokenized_doc = self.tokenizer.tokenize(processed_doc)
        tokenized_entity = self.tokenizer.tokenize(' ' + processed_entity)  # for first find, assuming entity is in middle of doc.

        # find entity in doc
        ent_start = kmp(tokenized_doc, tokenized_entity)
        if ent_start == -1:
            tokenized_entity = self.tokenizer.tokenize(processed_entity)    # without a space before entity, when entity is at the beggining of doc.
            ent_start = kmp(tokenized_doc, tokenized_entity)
            if ent_start == -1: 
                logging.warning(f'entity {entity} not found in corresponding doc!')
                with open('tokenized_doc.txt', 'a+') as f:
                    print('='*32 + f'\n\n{doc_string}\n\n{tokenized_doc}\n\n{entity}\t{tokenized_entity}', file=f)
                ent_start = 0
        ent_end = ent_start + len(tokenized_entity)

        # get doc snippet around entity
        snippet_start, snippet_end = expand(ent_start, ent_end, len(tokenized_doc), num_max_tokens)
        snippet_doc = tokenized_doc[snippet_start: snippet_end]
        # print(f'{snippet_end - snippet_start = }')

        return self.tokenizer.convert_tokens_to_string(snippet_doc)

    def _snippet_ic(self, hid, tid, doc1, doc2, num_max_tokens):
        # doc objects
        doc1 = json.loads(self.redisd.get('codred-doc-' + doc1))
        doc2 = json.loads(self.redisd.get('codred-doc-' + doc2))

        entity_h = get_entity_by_id(hid, doc1)['name']
        entity_t = get_entity_by_id(tid, doc2)['name']

        # doc string
        doc1 = self.bert_tokenizer.convert_tokens_to_string(doc1['tokens'])
        doc2 = self.bert_tokenizer.convert_tokens_to_string(doc2['tokens'])

        # filter doc string
        num_tokens_doc1 = num_tokens_doc2 = num_max_tokens // 2
        doc1 = self._snippet_doc(doc1, entity_h, num_tokens_doc1)
        doc2 = self._snippet_doc(doc2, entity_t, num_tokens_doc2)

        return doc1, doc2, entity_h, entity_t

    def _get_doc_and_entity(self, instance):
        subset = instance['subset']
        
        instance = instance['Instance']
        hid, tid = instance['entity_pair']
        doc1, doc2 = instance['doc_pair']  # doc titles

        if doc1 is None:  # dsre
            doc_id = doc2
            h_start, h_end = map(int, hid.split('#'))
            t_start, t_end = map(int, tid.split('#'))
            doc = json.loads(self.redisd.get(f'dsre-doc-{doc_id}'))
            
            entity_h = self.bert_tokenizer.convert_tokens_to_string(doc[h_start: h_end])
            entity_t = self.bert_tokenizer.convert_tokens_to_string(doc[t_start: t_end])

            h_len = t_len = self.max_source_length // 2
            h_1, h_2 = expand(h_start, h_end, len(doc), h_len)
            t_1, t_2 = expand(t_start, t_end, len(doc), t_len)

            doc1 = self.bert_tokenizer.convert_tokens_to_string(doc[h_1: h_2])
            doc2 = self.bert_tokenizer.convert_tokens_to_string(doc[t_1: t_2])
            text_path = f'{doc1}\n{doc2}\n'
        else:  # CodRED
            ## do filtering!
            assert self.key2textpath
            if self.key2textpath:
                key = f'{hid}#{tid}#{doc1}#{doc2}'
                # print(f'{key = }')
                # import pdb
                # pdb.set_trace()
                text_path = self.key2textpath[subset][key] + '\n'
                doc1 = json.loads(self.redisd.get('codred-doc-' + doc1))
                doc2 = json.loads(self.redisd.get('codred-doc-' + doc2))
                entity_h = get_entity_by_id(hid, doc1)['name']
                entity_t = get_entity_by_id(tid, doc2)['name']
            else:
                if self.ic_strategy == 'snippet':
                    doc1, doc2, entity_h, entity_t = self._snippet_ic(
                        hid, tid, doc1, doc2, 
                        num_max_tokens=self.max_source_length
                    )
                    # print(f'{len(self.tokenizer.tokenize(doc1)) = }')
                    # print(f'{len(self.tokenizer.tokenize(doc2)) = }')
                    text_path = f'{doc1}\n{doc2}\n'
                elif self.ic_strategy == 'ecrim':
                    filtered_text, entity_h, entity_t = self._ecrim_ic(
                        hid, tid, doc1, doc2, 
                        num_max_tokens=self.max_source_length
                    )
                    text_path = filtered_text + '\n'
                    # print(f'{text_path = }')
                    # import pdb
                    # pdb.set_trace()
                else:
                    raise NotImplementedError(f'Input contruction method `{self.ic_strategy}` not implemented yet, choose from `snippet` / `ecrim`')

        return text_path, entity_h, entity_t

    def _fill_slots_in_instruction(self, instance):
        """ Replace placeholder with corresponding entities and document content
        """
        instruction = instance['Instance']["instruction"]
        content, entity_h, entity_t = self._get_doc_and_entity(instance)
        instruction = instruction.replace('[HEAD_ENTITY]', entity_h) \
                                 .replace('[TAIL_ENTITY]', entity_t) \
                                 .replace('[CONTEXT]', content)
        return instruction


set_seed(42)
tree_instance_dir = f'{TRAIN_SRC}/data/tree_instances/{tree_name}/'
labeltree = LabelTree(tree_instance_dir, internal_node_strategy)
assert labeltree.na_node == NA and labeltree.pos_start_node == POS, f'NA: <{labeltree.na_node}, {NA}>\nPOS: <{labeltree.pos_start_node}, {POS}>'
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=f'{TRAIN_SRC}/pretrained_models/Meta-Llama-3-8B-Instruct',
    cache_dir=None,
    use_fast=True,
    revision="main",
    use_auth_token=None,
)
redisd = redis.Redis(host='localhost', port=6379, decode_responses=True, db=0)
bert_tokenizer = AutoTokenizer.from_pretrained(f'{TRAIN_SRC}/pretrained_models/bert-base-cased')
sbert_wk = SentenceBert(pretrained_bert_path=f'{TRAIN_SRC}/pretrained_models/bert-base-cased')#, device=f'cuda:{data_args.aux_model_gpu}')
label_pad_token_id = -100
key2textpath = {
    split: json.load(
        open(os.path.join(f'{TRAIN_SRC}/data/rawdata/CodRED/key2textpath', f'{split}_key2textpath.json'))
    ) for split in subsets
}
data_collator = DataCollatorForCDRE(
    redisd=redisd, 
    sbert_wk=sbert_wk,
    bert_tokenizer=bert_tokenizer, 
    tokenizer=tokenizer, 
    padding="longest", 
    ic_strategy=IC_STRATEGY,
    key2textpath=key2textpath,
    max_source_length=2040, 
    max_target_length=64, 
    label_pad_token_id=label_pad_token_id, 
    pad_to_multiple_of=None, 
)
marker = {}

def generate_instances(data, dataset_name, subset):
    
    def serialize_opt(index, option):
        if args.serial_num_type is None:
            return option
        
        if args.serial_num_type == 'uni_special_token':
            node_key = option
            if node_key not in marker:
                marker[node_key] = len(marker)
            return f'<|{marker[node_key]}|>{option}'
        
        serial_num = SERIAL_NUMS[args.serial_num_type][index]
        if args.serial_num_type in ['number', 'letter', 'roman']:
            return f'{serial_num}. {option}'
        elif args.serial_num_type in ['special_token']:
            return f'{serial_num}{option}'
        else:
            raise NotImplementedError(f'Invalid serial_num_type = {args.serial_num_type}')

    def _instance(options, label, prev_nodes):
        assert label in options, f'{label = } not in {options = }'
        l_idx = options.index(label)
        serial_options = [ serialize_opt(idx, opt) for idx, opt in enumerate(options)]
        serial_label = serial_options[l_idx]
        return {
            "Task": "CDRE",
            "Dataset": dataset_name,
            "subset": subset,
            "Instance": {
                "doc_pair": data['doc'],
                "entity_pair": [data['h_id'], data['t_id']],
                "label": serial_label,
                "instruction": PROMPT_TEMPLATE.replace('[OPTIONS]', ', '.join(serial_options))\
                                              .replace('[PREV_NODES]', ' > '.join(prev_nodes)), 
            }
        }
    
    def _substitution_instances(labeltree, cur_node, options, reason_chain, prev_nodes):
        substitution_instances = []

        best_node_idx = options.index(cur_node)
        children = labeltree.children_of(prev_nodes + [cur_node])
        if children and 'best' in SUBSTITUTE:
            options_sub = options[:best_node_idx] + children + options[best_node_idx+1:]
            substitution_instances.append(
                _instance(options=options_sub, label=reason_chain[step+1], prev_nodes=prev_nodes)
            )
            
        if 'suboptimal' in SUBSTITUTE:
            siblings = options[:best_node_idx] + options[best_node_idx+1:]
            if step == 0:
                assert len(siblings) == 1, f'{siblings = }\t{node_idx = }'
                do_substitute = random.choices([0, 1], weights=[0.9, 0.1], k=1)[0]
                if siblings[0] == labeltree.pos_start_node and do_substitute:
                    nodes_to_be_sub = siblings  # [pos_start_node]
                else:
                    nodes_to_be_sub = []
            else:
                # nodes_to_be_sub = random.sample(siblings, k=int(len(siblings) * 0.5))  # 374k
                nodes_to_be_sub = (
                    random.sample(siblings, k=1)
                    if siblings else []
                )  # 273k

            for node in nodes_to_be_sub:
                children = labeltree.children_of(prev_nodes + [node])
                assert children != [], f'{node = }, {prev_nodes = }'
                node_idx = options.index(node)
                options_sub = options[:node_idx] + children + options[node_idx+1:]
                substitution_instances.append(
                    _instance(options=options_sub, label=cur_node, prev_nodes=prev_nodes)
                )
        
        if 'double' in SUBSTITUTE:
            if step == 0:
                if 'best' not in SUBSTITUTE and cur_node == labeltree.pos_start_node: 
                    children = labeltree.children_of(prev_nodes + [cur_node])
                    assert children
                    options_sub = options[:best_node_idx] + children + options[best_node_idx+1:]
                    substitution_instances.append(
                        _instance(options=options_sub, label=reason_chain[step+1], prev_nodes=prev_nodes)
                    )
                if 'suboptimal' not in SUBSTITUTE and cur_node == labeltree.na_node:
                    do_substitute = random.choices([0, 1], weights=[0.9, 0.1], k=1)[0]
                    if do_substitute:
                        suboptimal_nodes = [ labeltree.pos_start_node ]
                        for node in suboptimal_nodes:
                            node_idx = options.index(node)
                            children = labeltree.children_of(prev_nodes + [node])
                            assert children
                            options_sub = options[:node_idx] + children + options[node_idx+1:]
                            substitution_instances.append(
                                _instance(options=options_sub, label=cur_node, prev_nodes=prev_nodes)
                            )
            else:
                best_nodes = [ cur_node ]
                siblings = options[:best_node_idx] + options[best_node_idx+1:]
                suboptimal_nodes = (
                    random.sample(siblings, k=1)
                    if siblings else []
                )
                for bnode, snode in zip(best_nodes, suboptimal_nodes):
                    options_sub = options.copy()
                    for node in [bnode, snode]:
                        children = labeltree.children_of(prev_nodes + [node])
                        assert children != [], f'{node = }, {prev_nodes = }, {[bnode, snode] = }'
                        node_idx = options_sub.index(node)
                        options_sub = options_sub[:node_idx] + children + options_sub[node_idx+1:]
                    substitution_instances.append(
                        _instance(options=options_sub, label=reason_chain[step+1], prev_nodes=prev_nodes)
                    )

        return substitution_instances
    
    if subset == 'train':
        instances = []
        reasoning_chains = labeltree.get_path_by_label(
            NA if data['relation']['id'] == 'n/a' else data['relation']['name'], 
            return_all=True, 
        )
        for reason_chain in reasoning_chains:
            prev_nodes = []
            for step, node_name in enumerate(reason_chain):
                options = (
                    labeltree.root_nodes if step == 0
                    else labeltree.children_of(prev_nodes)
                )

                ## add normal multistep instance
                instances.append(
                    _instance(options=options, label=node_name, prev_nodes=prev_nodes)
                )

                ## add substitution training instances
                if SUBSTITUTE and step != labeltree.max_depth - 1:
                    instances += _substitution_instances(labeltree, node_name, options, reason_chain, prev_nodes)

                prev_nodes.append(node_name)
    
    else:
        instances = [{
            "Task": "CDRE",
            "Dataset": dataset_name,
            "subset": subset,
            "Instance": {
                "doc_pair": data['doc'],
                "entity_pair": [data['h_id'], data['t_id']],
                "label": data['relation']['id'] if 'relation' in data else None,  # dev or test
                # "instruction": "[head entity]$#$[tail entity]$#$[CONTEXT]"
            }
        }]

    return instances


def do_convert(src, trg, dataset_name, subset):
    src_data = json.load(open(src))
    trg_data = []
    idx = 0
    for data in tqdm(src_data, desc=f'{dataset_name}-{subset}'):  # one text pair
        # n reasoning paths for a text pair, k steps for an average reasoning path
        instances = generate_instances(data, dataset_name, subset)  # len = n*k
        for instance in instances:
            if subset == 'train':
                trg_data.append({
                    "id": idx, 
                    "instruction": data_collator._fill_slots_in_instruction(instance),
                    "input": "",
                    "output": instance["Instance"]["label"]
                })
            else:
                content, entity_h, entity_t = data_collator._get_doc_and_entity(instance)
                trg_data.append({
                    "id": idx, 
                    "eh": entity_h, 
                    "et": entity_t, 
                    "context": content, 
                    "label": instance["Instance"]["label"]
                })
            idx += 1
    json.dump(trg_data, open(trg, 'w'), indent=4)
    print(f'\nDataset length of {dataset_name}-{subset}: {len(trg_data)}\n')


CURDIR = f'{TRAIN_SRC}/data/rawdata'
fnames = os.listdir(CURDIR)
fnames = list(filter(lambda x: x.startswith('CodRED') and x not in skipping, fnames))
print(f'{skipping = }')
print(f'{fnames = }')

successes = []
failures  = []
for fname in tqdm(fnames):
    prefix = tree_name
    if NOPREV:
        prefix += '-NoPrev'
    if ORIGIN:
        prefix += '-original_prompt'
    if SUBSTITUTE:
        prefix += f"-Sub{'.'.join(SUBSTITUTE)}"
    if args.serial_num_type:
        prefix += f'-SType{args.serial_num_type}'

    os.makedirs(os.path.join(CURDIR, 'multistep', prefix), exist_ok=True)
    try:
        for subset in subsets:
            do_convert(
                os.path.join(CURDIR, fname, f'{subset}_dataset.json'), 
                os.path.join(CURDIR, 'multistep', prefix, f'{fname}-{subset}.json'), 
                dataset_name=fname, subset=subset  #, labels_str=labels_str
            )
        if marker: 
            json.dump({ str(k): v for k, v in marker.items() }, open(os.path.join(CURDIR, 'multistep', prefix, f'{fname}-marker.json'), mode='w'), indent=2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'error occured in processing dataset ({fname})')
        failures.append(fname)
        continue
    successes.append(fname)

print(f'all done. \n{successes = }\n{failures = }')

