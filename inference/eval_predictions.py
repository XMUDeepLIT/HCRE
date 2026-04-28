import os
import json
import random
import argparse
from sklearn.metrics import (
    f1_score, 
    classification_report, 
)

PRINT_ALL_OOD_PREDS = False
SEED = 42

ALL_NA_TYPE = [
    ('no valid relation', 'valid relations'), 
    ('not available', 'available'), 
    ('invalid relation', 'valid relation'), 
    ('no', 'yes'), 
]

parser = argparse.ArgumentParser()
parser.add_argument('--eval_dir', type=str, required=True)
parser.add_argument('--eval_subset', type=str, choices=['dev', 'open_dev', 'test', 'open_test'], default='dev')
parser.add_argument('--path_label', action="store_true")
parser.add_argument('--tree_name', type=str)
parser.add_argument('--na_type', type=int, required=True)
args = parser.parse_args()

# XXX
NA_TYPE_IDX = args.na_type
PATH_LABEL = args.path_label
TREE_TYPE = args.tree_name
INTERNAL_NODE_STRATEGY = 'echo'

# assign n/a and positive start
NA, POSITIVE_START = ALL_NA_TYPE[NA_TYPE_IDX]


def get_all_labels():
    all_labels = json.load(open('data/rawdata/relations_with_desc.json'))
    assert all_labels[0]['rel_id'] == 'n/a', f'{all_labels[0] = }'
    all_labels[0]['rel_name'] = NA
    return all_labels


def compute_metrics(predictions, references, rname2rid):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    assert isinstance(predictions[0], str), f'{type(predictions[0]) = }'

    relation_num = len(rname2rid)

    wrong_pairs = []
    preds, golds = [], []
    bin_preds, bin_golds = [], []
    for idx, (pred, gold) in enumerate(zip(predictions, references)):
        if pred != gold:
            wrong_pairs.append({'idx': idx, 'pred': pred, 'gold': gold})
        preds.append(rname2rid[pred] if pred in rname2rid else -1)
        golds.append(rname2rid[gold])
        bin_preds.append(int(rname2rid[pred] > 0) if pred in rname2rid else 0)
        bin_golds.append(int(rname2rid[gold] > 0))

    with open(os.path.join(eval_folder, 'cls_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(golds, preds, labels=range(1, relation_num), digits=4, zero_division=0.))

    binary_f1 = f1_score(bin_golds, bin_preds, average='binary')
    micro_f1 = f1_score(golds, preds, labels=range(1, relation_num), average='micro', zero_division=0.)

    metrics = {
        "# of wrong pairs" : len(wrong_pairs), 
        "path_binary_f1": binary_f1, 
        "path_micro_f1" : micro_f1, 
    }

    return metrics


def get_final_pred(maybe_path_pred):
    return (
        maybe_path_pred.split('->')[-1].strip()
        if PATH_LABEL else maybe_path_pred.strip()
    )


if __name__ == '__main__':
    random.seed(SEED)

    eval_folder = args.eval_dir
    print(f'eval folder: {eval_folder}')

    all_labels = get_all_labels()
    rname2rid = { rel['rel_name']: i for i, rel in enumerate(all_labels) }
    pid2rname = { rel['rel_id']: rel['rel_name'] for rel in all_labels }
    latencies, samples = [], []
    predictions, references = [], []
    with open(os.path.join(eval_folder, 'generated_predictions.jsonl')) as f:
        for idx, line in enumerate(f):
            try:
                sample = json.loads(line)
            except: 
                print(f"Load JSON line failed: {line}")
                exit()
            if 'latency' in sample:
                latencies.append(sample['latency'])
            maybe_path_pred = sample['predict'][0] if isinstance(sample['predict'], list) else sample['predict']
            pred = get_final_pred(maybe_path_pred)
            ref = sample['label'].split('->')[-1].strip() if sample['label'] else None
            predictions.append(pred)
            references .append(pid2rname[ref])

    # do metrics!
    if args.eval_subset in ['dev', 'open_dev']:
        results = {
            "seed": SEED, 
            "avg. latency": f'{sum(latencies) / len(latencies)}s' if latencies else None, 
            **compute_metrics(predictions, references, rname2rid), 
        }
        # output results
        results_str = json.dumps(results, indent=4)
        print(f'eval results: \n{results_str}')

        result_file = os.path.join(eval_folder, 'eval_results.json')
    else:
        raise ValueError(f'Invalid eval subset {args.eval_subset}')

    json.dump(results, open(result_file, 'w'), indent=4)
    print(f'results have been dumpped to {result_file}')

