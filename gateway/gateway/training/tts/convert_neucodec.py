"""Convert raw audio files to NeuCodec speech tokens.

Each audio file at `*.mp3` / `*.wav` is encoded with `neuphonic/neucodec` and
written as a JSON list of token IDs to a sibling `<folder>_neucodec/...json`.

Progress reporting:
  Workers share a `multiprocessing.Value` counter that's incremented per file
  processed. A poller in the main process prints
  `[AUTOTRAIN_PROGRESS] step=convert_neucodec percent=NN.N`
  every ~2 s so the AutoTrain worker can update the job step's progress bar.
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import time
import threading
from glob import glob
from functools import partial
import itertools

import click
import librosa
import numpy as np
import soundfile as sf
from multiprocess import Pool, Value
from tqdm import tqdm


def old_chunks(l, n):
    for i in range(0, len(l), n):
        yield (l[i: i + n], i // n)


def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end


def new_path(f):
    splitted = f.split('/')
    folder = f.split('/')[0]
    folder = folder + '_neucodec'
    new_f = os.path.join(folder, '/'.join(splitted[1:]))
    new_f = new_f.replace('.mp3', '.json').replace('.wav', '.json')
    return new_f


def multiprocessing(strings, function, cores=6, returned=True):
    df_split = old_chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()

    if returned:
        return list(itertools.chain(*pooled))


def check(files):
    files, _ = files
    filtered = []
    for file in tqdm(files):
        filename_done = new_path(file)

        if os.path.exists(filename_done):
            try:
                with open(filename_done) as fopen:
                    json.load(fopen)
                    continue
            except Exception:
                pass

        filtered.append(file)
    return filtered


# ---- worker (per device) ---------------------------------------------------

# `_counter` and `_total` are set by `_init_worker` so each forked process
# inherits a reference to the shared counter without us having to thread it
# through the Pool.map argument tuple.
_counter = None
_total = 0


def _init_worker(counter, total):
    global _counter, _total
    _counter = counter
    _total = total


def loop(indices_device_pair):
    files, device = indices_device_pair
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    from neucodec import NeuCodec
    import torch

    torch.autograd.set_grad_enabled(False)

    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model.eval().cuda()

    for f in files:
        # Exactly one progress bump per file, in `finally` (the old explicit
        # bumps in the skip branches double-counted → 122/61 = 200%).
        try:
            filename = new_path(f)
            if os.path.exists(filename):
                try:
                    with open(filename) as fopen:
                        json.load(fopen)
                    continue
                except Exception:
                    pass

            try:
                y, sr = librosa.load(f, sr=16000)
                # No duration cap — encode the whole clip. (The old 20s cap
                # silently skipped every long-form clip, writing zero tokens, so
                # pack_stage1 had nothing to pack.) Over-long utterances are still
                # bounded later by pack_stage1's --sequence_length filter.
                wav_tensor = torch.from_numpy(y).float().unsqueeze(0)
                fsq_codes = model.encode_code(wav_tensor.unsqueeze(1))
                tokens = fsq_codes[0, 0].tolist()

                os.makedirs(os.path.split(filename)[0], exist_ok=True)
                with open(filename, 'w') as fopen:
                    json.dump(tokens, fopen)
            except Exception as e:
                print(f'[convert_neucodec] error processing {f!r}: {e}', flush=True)
        finally:
            _bump()


def _bump():
    """Bump the shared counter by one. Best-effort — never crash the worker."""
    if _counter is None:
        return
    try:
        with _counter.get_lock():
            _counter.value += 1
    except Exception:
        pass


# ---- progress poller -------------------------------------------------------

def _start_progress_poller(counter, total, interval=2.0):
    stop = threading.Event()

    def poll():
        last = -1
        while not stop.is_set():
            v = counter.value
            if total > 0 and v != last:
                pct = (v / total) * 100.0
                print(
                    f'[AUTOTRAIN_PROGRESS] step=convert_neucodec '
                    f'processed={v} total={total} percent={pct:.1f}',
                    flush=True,
                )
                last = v
            stop.wait(interval)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    return stop, t


@click.command()
@click.option('--file', required=True, help='JSON file containing a list of audio paths.')
@click.option('--replication', default=1)
def main(file, replication):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        import torch
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices
    print(f'[convert_neucodec] devices: {devices}', flush=True)

    with open(file) as fopen:
        files = json.load(fopen)
    print(
        f'[AUTOTRAIN_PROGRESS] step=convert_neucodec processed=0 total={len(files)} percent=0.0',
        flush=True,
    )

    filtered = multiprocessing(files, check, 30)
    already_done = len(files) - len(filtered)
    total = len(files)
    print(
        f'[convert_neucodec] {already_done}/{total} already converted, {len(filtered)} to do',
        flush=True,
    )

    counter = Value('i', already_done)
    stop, poller = _start_progress_poller(counter, total)

    df_split = list(chunks(filtered, devices))

    try:
        with Pool(
            len(devices),
            initializer=_init_worker,
            initargs=(counter, total),
        ) as pool:
            pool.map(loop, df_split)
    finally:
        stop.set()
        poller.join(timeout=3.0)
        v = counter.value
        pct = (v / total) * 100.0 if total > 0 else 100.0
        print(
            f'[AUTOTRAIN_PROGRESS] step=convert_neucodec '
            f'processed={v} total={total} percent={pct:.1f}',
            flush=True,
        )


if __name__ == '__main__':
    main()
