"""Pack converted speech tokens + transcripts into a MosaicML streaming dataset.

This is the script form of `packing-stage1.ipynb`. The input dataset (rows
with `filename_audio` / `text` / `speaker`) is paired with the per-utterance
NeuCodec token files written by `convert_neucodec.py`, then chunked into
fixed-length training blocks and written to `--output_dir` as MDS shards.

Progress markers (`[AUTOTRAIN_PROGRESS] step=pack_stage1 percent=N`) are
emitted every ~100 rows so the AutoTrain UI can show a progress bar.
"""
import os
import sys
import copy
import json
import argparse

# Vendored ChiniDataset lives next to this script (shipped with the tts dir) —
# put it on sys.path so `import chinidataset` works on the GPU box without a pip
# install (the repo is private + the box has no GitHub creds).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from chinidataset import ParquetWriter, StreamingDataset
from datasets import load_dataset

# ChiniDataset is Parquet-native — array columns are typed `<dtype>[]` (no custom
# MDS encoding needed; uint32 arrays are stored natively).
COLUMNS = {
    'input_ids': 'uint32[]',
    'position_ids': 'uint32[]',
    'attention_mask': 'uint32[]',
    'audio': 'str',
    'text': 'str',
}


def new_path(f: str) -> str:
    splitted = f.split('/')
    folder = f.split('/')[0]
    folder = folder + '_neucodec'
    new_f = os.path.join(folder, '/'.join(splitted[1:]))
    return new_f.replace('.mp3', '.json').replace('.wav', '.json')


def filter_rows(rows):
    # Pack every row that has a NeuCodec token file — no content filtering.
    # (Previously dropped any transcription containing a digit or 'http'; the
    # user wants all utterances kept, digits included.)
    out, missing = [], []
    for c in rows:
        c = copy.copy(c)
        token_filename = new_path(c['filename_audio'])
        if not os.path.exists(token_filename):
            missing.append(c['filename_audio'])
            continue
        c['token_filename'] = token_filename
        out.append(c)
    return out, missing


def collator(batch, batch_position_ids):
    input_ids, position_ids, masks = [], [], []
    for i in range(len(batch)):
        l = len(batch[i])
        input_ids.extend(batch[i])
        position_ids.extend(batch_position_ids[i])
        masks.append(l)
    return {
        'input_ids': np.array(input_ids).astype(np.uint32),
        'position_ids': np.array(position_ids).astype(np.uint32),
        'attention_mask': np.array(masks).astype(np.uint32),
        'audio': '',
        'text': '',
    }


def loop(rows, folder, tokenizer, sequence_length=4096, progress_every=100):
    block_size = sequence_length
    os.system(f'rm -rf {folder}')
    count = 0
    temp = []
    position_ids = []
    total = len(rows)
    print(
        f'[AUTOTRAIN_PROGRESS] step=pack_stage1 processed=0 total={total} percent=0.0',
        flush=True,
    )
    with ParquetWriter(out=folder, columns=COLUMNS) as out:
        for i, row in enumerate(tqdm(rows)):
            try:
                with open(row['token_filename']) as fopen:
                    token = json.load(fopen)
            except Exception:
                continue

            left = row['speaker'] + ': ' + row['text']
            token = ''.join([f'<|s_{t}|>' for t in token])
            prompt = f'<|im_start|>{left}<|speech_start|>{token}<|im_end|>'
            outputs = tokenizer(prompt, add_special_tokens=False)
            length = len(outputs['input_ids'])

            if length > sequence_length:
                continue

            if count + length > block_size:
                o = collator(temp, position_ids)
                if o['input_ids'].shape[0] > 0:
                    out.write(o)
                temp = [outputs['input_ids']]
                position_ids = [range(length)]
                count = length
            else:
                temp.append(outputs['input_ids'])
                position_ids.append(range(length))
                count += length

            if total > 0 and (i + 1) % progress_every == 0:
                pct = ((i + 1) / total) * 100.0
                print(
                    f'[AUTOTRAIN_PROGRESS] step=pack_stage1 '
                    f'processed={i + 1} total={total} percent={pct:.1f}',
                    flush=True,
                )

        if temp:
            o = collator(temp, position_ids)
            if o['input_ids'].shape[0] > 0:
                out.write(o)

    print(
        f'[AUTOTRAIN_PROGRESS] step=pack_stage1 '
        f'processed={total} total={total} percent=100.0',
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Scicom-intl/process-stage-1',
                        help='HF dataset id or local arrow / json path.')
    parser.add_argument('--split', default='train')
    parser.add_argument('--tokenizer', default='Scicom-intl/Multilingual-Expressive-TTS-1.7B')
    parser.add_argument('--output_dir', required=True,
                        help='MDS output directory (becomes train_file for the trainer).')
    parser.add_argument('--sequence_length', type=int, default=4096)
    args = parser.parse_args()

    print(f'[pack_stage1] loading tokenizer {args.tokenizer}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f'[pack_stage1] loading dataset {args.dataset!r} (split={args.split!r})', flush=True)
    # AutoTrain orchestrator passes a local json/jsonl metadata file; everything
    # else is treated as a HF dataset id / loadable path.
    if args.dataset.endswith(('.json', '.jsonl')):
        ds = load_dataset('json', data_files=args.dataset)
    else:
        ds = load_dataset(args.dataset)
    split = args.split if args.split in ds else list(ds.keys())[0]
    rows = ds[split].to_list()

    print(f'[pack_stage1] filtering / pairing tokens for {len(rows)} rows', flush=True)
    processed, missing = filter_rows(rows)
    print(
        f'[pack_stage1] {len(processed)}/{len(rows)} rows packed; '
        f'{len(missing)} missing token files',
        flush=True,
    )

    loop(processed, args.output_dir, tokenizer, sequence_length=args.sequence_length)

    dataset = StreamingDataset(local=args.output_dir)
    print(f'[pack_stage1] wrote {len(dataset)} packed records to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
