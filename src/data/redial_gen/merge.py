import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--gen_file_prefix", type=str, required=True)
args = parser.parse_args()
gen_file_prefix = args.gen_file_prefix
dataset = 'redial'

## currently at src/
for split in ['train', 'valid', 'test', 'sample_input']:
    raw_file_path = f"/kaggle/working/InferConverRec/src/data/{dataset}/{split}_data_processed.jsonl"
    raw_file = open(raw_file_path, encoding='utf-8')
    raw_data = raw_file.readlines()
    print(f"Number of entries in raw_{split} data: {len(raw_data)}")

    gen_file_path = f"/kaggle/working/InferConverRec/src/save/{dataset}/{gen_file_prefix}_{split}.jsonl"
    gen_file = open(gen_file_path, encoding='utf-8')
    gen_data = gen_file.readlines()
    print(f"Number of entries in gen_{split} data: {len(gen_data)}")

    new_file_path = f'{split}_data_processed.jsonl'
    new_file = open(new_file_path, 'w', encoding='utf-8')
    cnt = 0
    skipped = 0

    for i, raw in enumerate(raw_data):
        raw = json.loads(raw)
        if len(raw['context']) == 1 and raw['context'][0] == '':
            raw['resp'] = ''
        else:
            if cnt < len(gen_data):
                gen = json.loads(gen_data[cnt])
                pred = gen['pred']
                if '<movie>' in pred:
                    raw['resp'] = pred.split('System: ')[-1]
                else:
                    raw['resp'] = ''
                cnt += 1
            else:
                print(f"Warning: Reached end of gen_data at index {i} in raw_data")
                raw['resp'] = ''
                skipped += 1

        new_file.write(json.dumps(raw, ensure_ascii=False) + '\n')

    print(f"Processed entries: {cnt}")
    print(f"Skipped entries due to gen_data shortage: {skipped}")
    print(f"Total entries written: {len(raw_data)}")
    print(f"Entries in gen_data: {len(gen_data)}")
    print("---")

print("Processing complete.")
