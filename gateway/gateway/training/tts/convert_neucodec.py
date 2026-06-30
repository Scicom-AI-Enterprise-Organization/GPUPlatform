"""Convert audio to NeuCodec speech tokens — streaming + prefetching from S3.

Instead of bulk-downloading the whole dataset to disk first, each GPU worker runs
a producer/consumer pipeline: a pool of downloader threads (comm) fetch + decode
audio (from an S3/HTTP URL, in memory) into a bounded queue while the GPU drains
it (comp). Download and encode therefore overlap, and disk use stays ~constant.

Input (`--file`): a JSON list of items, each either
  - a string  → a local audio path (key == path), or
  - {"key": <token-key>, "src": <local-path-or-URL>}
`key` is the token-file key (matches pack_stage1's `filename_audio`); `src` is
where the audio comes from. Tokens are written to `<key-folder>_neucodec/...json`.

Progress / logs:
  - `[AUTOTRAIN_PROGRESS] step=convert_neucodec processed=N total=M percent=P`
    (overall bar, emitted from the main process every ~2 s)
  - per-GPU granular lines every ~5 s: encoded/queued counts, queue depth, and
    encode vs download throughput (files/s, MB/s).
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import io
import json
import queue as _queue
import threading
import time
import urllib.request

import click
import librosa
from multiprocess import Pool, Value

# Per-GPU downloader threads + how far to prefetch ahead (bounded → ~constant mem).
_DL_THREADS = int(os.environ.get('NEUCODEC_DL_THREADS', '8') or '8')
_PREFETCH = int(os.environ.get('NEUCODEC_PREFETCH', '48') or '48')
_SR = 16000


def new_path(key):
    """Token-file path for an audio key — must match pack_stage1.new_path:
    `<first-segment>_neucodec/<rest>.json`."""
    splitted = key.split('/')
    folder = splitted[0] + '_neucodec'
    new_f = os.path.join(folder, '/'.join(splitted[1:]))
    return new_f.replace('.mp3', '.json').replace('.wav', '.json')


def _norm(item):
    """(key, src) for an input item — a bare string is a local path (key==src)."""
    if isinstance(item, str):
        return item, item
    return item['key'], (item.get('src') or item['key'])


def _token_done(key):
    """True if this clip's token file already exists + is valid JSON (resume)."""
    p = new_path(key)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            json.load(f)
        return True
    except Exception:
        return False


def _fetch_decode(src):
    """Return (waveform float32 mono @ _SR, bytes_moved). Streams from S3/HTTP into
    memory when `src` is a URL (no file written), else decodes the local file."""
    if src.startswith(('http://', 'https://')):
        with urllib.request.urlopen(urllib.request.Request(src), timeout=120) as r:
            raw = r.read()
        try:
            y, _ = librosa.load(io.BytesIO(raw), sr=_SR)
        except Exception:
            # Some audio backends can't decode (e.g.) mp3 from memory — fall back
            # to a single transient temp file (still streaming: one small file at
            # a time, deleted immediately; never the whole dataset on disk).
            import tempfile
            suffix = os.path.splitext(src.split('?')[0])[1] or '.mp3'
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                    tf.write(raw)
                    tmp = tf.name
                y, _ = librosa.load(tmp, sr=_SR)
            finally:
                if tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        return y, len(raw)
    y, _ = librosa.load(src, sr=_SR)
    try:
        return y, os.path.getsize(src)
    except OSError:
        return y, 0


# ---- shared progress counter (set per worker via the Pool initializer) -----
_counter = None
_total = 0


def _init_worker(counter, total):
    global _counter, _total
    _counter = counter
    _total = total


def _bump(n=1):
    if _counter is None:
        return
    try:
        with _counter.get_lock():
            _counter.value += n
    except Exception:
        pass


def loop(items_device):
    """One GPU worker: prefetch+decode audio on threads, encode on the GPU."""
    items, device = items_device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    from neucodec import NeuCodec
    import torch

    torch.autograd.set_grad_enabled(False)
    # Upstream neucodec (neuphonic/neucodec, 24 kHz). The frozen encoder resamples to
    # 16 kHz internally, so the FSQ speech codes are the standard NeuCodec tokens.
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model.eval().cuda()

    # Resume: skip clips whose tokens already exist (count them as done up front).
    todo, already = [], 0
    for it in items:
        key, src = _norm(it)
        if _token_done(key):
            already += 1
        else:
            todo.append((key, src))
    if already:
        _bump(already)
    if not todo:
        print(f'[convert_neucodec] dev{device}: all {already} clips already encoded', flush=True)
        return

    q: _queue.Queue = _queue.Queue(maxsize=_PREFETCH)
    stats = {'dl': 0, 'bytes': 0, 'fail': 0}
    slock = threading.Lock()

    def producer(sub):
        for key, src in sub:
            try:
                y, nbytes = _fetch_decode(src)
                q.put((key, y))
                with slock:
                    stats['dl'] += 1
                    stats['bytes'] += nbytes
            except Exception as e:  # noqa: BLE001 — skip the clip, keep streaming
                q.put((key, None))
                with slock:
                    stats['fail'] += 1
                print(f'[convert_neucodec] dev{device} fetch failed {src!r}: {e}', flush=True)

    n_threads = max(1, min(_DL_THREADS, len(todo)))
    # Round-robin split so every downloader thread stays busy regardless of order.
    subs = [todo[i::n_threads] for i in range(n_threads)]
    threads = [threading.Thread(target=producer, args=(s,), daemon=True) for s in subs]
    for t in threads:
        t.start()
    print(f'[convert_neucodec] dev{device}: streaming {len(todo)} clips '
          f'({already} cached) · {n_threads} dl-threads · prefetch {_PREFETCH}', flush=True)

    enc = 0
    t0 = time.time()
    last_log = t0
    while enc < len(todo):
        try:
            key, y = q.get(timeout=300)
        except _queue.Empty:
            if not any(t.is_alive() for t in threads):
                print(f'[convert_neucodec] dev{device}: producers ended early at '
                      f'{enc}/{len(todo)}', flush=True)
                break
            continue
        enc += 1
        if y is not None and len(y) > 0:
            try:
                wav = torch.from_numpy(y).float().unsqueeze(0)
                codes = model.encode_code(wav.unsqueeze(1))
                tokens = codes[0, 0].tolist()
                out = new_path(key)
                os.makedirs(os.path.split(out)[0], exist_ok=True)
                with open(out, 'w') as f:
                    json.dump(tokens, f)
            except Exception as e:  # noqa: BLE001
                print(f'[convert_neucodec] dev{device} encode failed {key!r}: {e}', flush=True)
        _bump()
        now = time.time()
        if now - last_log >= 5.0:
            with slock:
                dl, mb, fail = stats['dl'], stats['bytes'] / 1e6, stats['fail']
            dt = max(1e-6, now - t0)
            print(f'[convert_neucodec] dev{device} enc={enc}/{len(todo)} dl={dl} '
                  f'q={q.qsize()} fail={fail} · enc {enc / dt:.1f}/s · '
                  f'dl {mb:.0f}MB ({mb / dt:.1f}MB/s)', flush=True)
            last_log = now
    for t in threads:
        t.join(timeout=2.0)
    dt = max(1e-6, time.time() - t0)
    print(f'[convert_neucodec] dev{device} done: {enc} encoded in {dt:.0f}s '
          f'({enc / dt:.1f}/s), {stats["fail"]} failed', flush=True)


def chunks(items, devices):
    """Split items across devices (one GPU worker per device), as evenly as possible."""
    chunk_size = len(items) // len(devices)
    remainder = len(items) % len(devices)
    start = 0
    for i in range(len(devices)):
        end = start + chunk_size + (1 if i < remainder else 0)
        yield (items[start:end], devices[i])
        start = end


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
@click.option('--file', required=True,
              help='JSON list of items: local paths, or {"key","src"} (src may be an S3 URL).')
@click.option('--replication', default=1)
def main(file, replication):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        import torch
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',') if d.strip()]
    devices = replication * devices
    print(f'[convert_neucodec] devices: {devices}', flush=True)

    with open(file) as fopen:
        items = json.load(fopen)
    total = len(items)
    print(
        f'[AUTOTRAIN_PROGRESS] step=convert_neucodec processed=0 total={total} percent=0.0',
        flush=True,
    )
    print(f'[convert_neucodec] {total} clips · streaming + prefetch '
          f'({_DL_THREADS} dl-threads/GPU, prefetch {_PREFETCH}) over {len(devices)} GPU(s)',
          flush=True)

    counter = Value('i', 0)
    stop, poller = _start_progress_poller(counter, total)
    df_split = list(chunks(items, devices))
    try:
        with Pool(len(devices), initializer=_init_worker, initargs=(counter, total)) as pool:
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
        print(f'[convert_neucodec] complete: {v}/{total} clips encoded', flush=True)


if __name__ == '__main__':
    main()
