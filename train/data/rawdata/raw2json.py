import json
from tqdm import tqdm

rel_desc = json.load(open('relations_with_desc.json'))
id2rel = { rel['rel_id']: rel for rel in rel_desc }

split2path = {
    'train': 'original_rawdata/train_dataset.json', 
    'dev': 'original_rawdata/dev_dataset.json', 
    'open_dev': 'open_setting_data/dev_data_shared_entities_ranked.json', 
}


for split in ['train', 'dev', 'open_dev']:
    raw = json.load(open(split2path[split]))
    json_lst = []
    for ht, doc1, doc2, label_id in tqdm(raw, desc=f'Formatting {split}...'):
        rel_obj = id2rel[label_id]
        json_lst.append({
            'h_id': ht.split('#')[0], 
            't_id': ht.split('#')[1], 
            'doc': [ doc1, doc2 ], 
            'relation': {
                'id': rel_obj['rel_id'], 
                'name': rel_obj['rel_name'], 
                'description': rel_obj['rel_desc'], 
            }, 
        })
    json.dump(json_lst, open(f'CodRED/{split}_dataset.json', 'w', encoding='utf-8'), indent=4)

