uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.10.0 torchvision==0.25.0 
uv pip install huggingface_hub transformers ipykernel liger-kernel wandb mlflow
uv pip install "kernels<=0.14.0"
# uv pip install flash-linear-attention[cuda]
uv pip install git+https://github.com/QwenLM/FlashQLA.git
uv pip install causal_conv1d
uv pip install git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git

cat << EOF > pack_dataset.py
import os
import sys
import json
import argparse

import numpy as np

# --- Read HF_TOKEN from the gateway .env (the model repo may be GATED) --------
GATEWAY_ENV = "/home/husein/ssd3/GPUPlatform/gateway/.env"


def _load_hf_token_from_env_file(path: str) -> None:
    """Set HF_TOKEN / HUGGING_FACE_HUB_TOKEN from `grep 'HF_TOKEN=' <gateway/.env>`.

    Does not clobber an HF_TOKEN already present in the environment.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        os.environ["HF_TOKEN"] = val
                        os.environ["HUGGING_FACE_HUB_TOKEN"] = val
                    return
    except FileNotFoundError:
        print(f"[pack_dataset] WARN: {path} not found; relying on ambient HF_TOKEN",
              file=sys.stderr)


_load_hf_token_from_env_file(GATEWAY_ENV)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# COLUMN ENCODINGS -- see module docstring.
COLUMNS = {
    "input_ids": "int64[]",
    "labels": "int64[]",
    "position_ids": "uint32[]",
    "attention_mask": "uint32[]",
}

IGNORE_INDEX = -100

DEFAULT_REPO = "Scicom-intl/Function-Call-TaaS"
DEFAULT_FILE = "glm5.1-fp8-test/test-00000-of-00001.parquet"
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-27B"

# --- Reasoning rendering ------------------------------------------------------
# Qwen3.5's stock template only emits <think>...</think> on assistant turns AFTER
# `ns.last_query_index` (the last real user query, not a tool-response turn).
# In a multi-step agentic trajectory every intermediate assistant turn is before
# that index, so their reasoning is silently dropped.
#
# We relax this by replacing the guard:
#   STOCK : loop.index0 > ns.last_query_index
#   ALL   : loop.index0 >= 0   (always true)
#
# This is a surgical substring swap on the stock template string. If the template
# changes upstream the swap fails to find the string, warns, and returns the stock
# template rather than crashing.
_REASONING_GUARD_STOCK = (
    "{%- if loop.index0 > ns.last_query_index %}"
)
_REASONING_GUARD_ALL = (
    "{%- if loop.index0 >= 0 %}"
)


def build_chat_template(tokenizer, all_reasoning: bool):
    """Return the chat-template string to use (stock, or reasoning-relaxed).

    ``all_reasoning=False`` -> use the tokenizer's stock template unchanged.
    ``all_reasoning=True``  -> render every model turn's thinking block.

    Unlike the original version this function does NOT raise when the guard string
    is absent -- it warns and returns the stock template so the script stays runnable
    even if the upstream template changes or the guard constant above needs updating.
    """
    stock = tokenizer.chat_template
    if not all_reasoning:
        return stock
    if _REASONING_GUARD_STOCK not in stock:
        print(
            "[pack_dataset] WARN: could not find the Qwen3.5 reasoning guard in "
            "the chat template. Either the template changed upstream or "
            "_REASONING_GUARD_STOCK needs updating. Inspect "
            "tokenizer.chat_template and update the constant. Falling back to "
            "the stock template (native-reasoning behaviour).",
            file=sys.stderr,
        )
        return stock
    return stock.replace(_REASONING_GUARD_STOCK, _REASONING_GUARD_ALL)


def load_source_dataframe(repo: str, filename: str):
    """Load the source parquet as a pandas DataFrame.

    Tries the ``hf://`` pandas path first (honors HF_TOKEN), falling back to an
    explicit ``hf_hub_download(repo_type="dataset")`` + ``read_parquet``.
    """
    import pandas as pd

    hf_uri = f"hf://datasets/{repo}/{filename}"
    try:
        print(f"[pack_dataset] reading {hf_uri}", flush=True)
        return pd.read_parquet(hf_uri)
    except Exception as e:  # noqa: BLE001
        print(f"[pack_dataset] hf:// read failed ({type(e).__name__}: {e}); "
              f"falling back to hf_hub_download", flush=True)
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        local = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
        return pd.read_parquet(local)


def _normalize_tool_calls(tool_calls):
    """Convert GLM-style tool_calls to the format Qwen3.5's template expects.

    The Qwen3.5 chat template iterates tool_call.function.arguments with:
        for args_name, args_value in tool_call.function.arguments | items
    so `arguments` MUST be a plain Python dict, never a JSON string.

    GLM stores calls as ``[{"name": "fn", "arguments": {...}}]`` -- bare format.
    Qwen3.5 needs ``[{"type": "function", "function": {"name": "fn", "arguments": {...}}}]``.
    Already-wrapped entries are passed through; JSON-string arguments are parsed back to dict.
    """
    normalized = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        # Already in OpenAI format.
        if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
            fn = dict(tc["function"])
            # arguments must be a dict, not a JSON string.
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn["arguments"] = {}
            elif fn.get("arguments") is None:
                fn["arguments"] = {}
            entry = {"type": "function", "function": fn}
            entry["id"] = tc.get("id", f"call_{i}")
            normalized.append(entry)
        elif "name" in tc:
            # Bare GLM format: wrap it.
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            elif not isinstance(args, dict):
                args = {}
            normalized.append({
                "type": "function",
                "function": {"name": tc["name"], "arguments": args},
                "id": tc.get("id", f"call_{i}"),
            })
    return normalized


def extract_messages(value):
    """Normalize a ``messages`` cell into a list[dict] compatible with Qwen3.5.

    The source column stores ``messages`` as a JSON *string* (verified on the real
    parquet), but also tolerate a row that is already a list / numpy array of dicts.

    Normalizations applied for Qwen3.5 compatibility:
      - ``content: null``       -> ``content: ""``  (null crashes the template)
      - ``role: "observation"`` -> ``role: "tool"``  (GLM role name)
      - ``tool_calls``          -> OpenAI format with JSON-string arguments
      - tool messages           -> ensure ``tool_call_id`` is present
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None

    # First pass: collect tool-call ids in order so we can assign matching
    # tool_call_ids to subsequent tool-response turns.
    call_ids = []
    for turn in value:
        if isinstance(turn, dict) and turn.get("role") in ("assistant", "model"):
            for i, tc in enumerate(turn.get("tool_calls") or []):
                if isinstance(tc, dict):
                    cid = tc.get("id") or (
                        tc.get("function", {}).get("id") if isinstance(tc.get("function"), dict) else None
                    ) or f"call_{i}"
                    call_ids.append(cid)

    tool_response_idx = 0
    out = []
    for turn in value:
        if not (isinstance(turn, dict) and "role" in turn):
            continue
        t = dict(turn)

        # Remap GLM observation role.
        if t["role"] == "observation":
            t["role"] = "tool"

        # Null content -> empty string.
        if t.get("content") is None:
            t["content"] = ""

        # Normalize tool_calls in assistant turns.
        if t.get("tool_calls"):
            t["tool_calls"] = _normalize_tool_calls(t["tool_calls"])

        # Ensure tool-response turns have a tool_call_id.
        if t["role"] == "tool" and not t.get("tool_call_id"):
            if tool_response_idx < len(call_ids):
                t["tool_call_id"] = call_ids[tool_response_idx]
            else:
                t["tool_call_id"] = f"call_{tool_response_idx}"
            tool_response_idx += 1

        out.append(t)
    return out or None


def extract_tools(value):
    """Normalize a ``functions`` cell into the OpenAI tool list the template wants.

    The source ``functions`` column is a JSON *string* of BARE function objects
    (``{"name", "description", "parameters", ...}``). The Qwen3.5 template's
    ``format_function_declaration`` accesses ``tool['function']['name']``, so each
    bare function must be wrapped as ``{"type": "function", "function": <fn>}`` --
    passing the bare list is exactly what raises
    ``'dict object' has no attribute 'function'``.
    Returns a list (possibly empty) suitable for ``apply_chat_template(tools=...)``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    tools = []
    for fn in value:
        if not isinstance(fn, dict):
            continue
        if fn.get("type") == "function" and isinstance(fn.get("function"), dict):
            tools.append(dict(fn))
        else:
            tools.append({"type": "function", "function": dict(fn)})
    return tools


def tokenize_row(tokenizer, messages, tools=None, chat_template=None):
    """Render+tokenize one conversation; return (input_ids, labels, used_mask).

    ``tools``         : OpenAI-wrapped tool list (from extract_tools) -- rendered
                        into the system block so the model conditions on the
                        available functions. An empty list means no tool defs.
    ``chat_template`` : the template string to render with (stock or
                        reasoning-relaxed); None uses the tokenizer's default.

    PREFERS assistant-only labels via ``return_assistant_tokens_mask=True``; falls
    back to ``labels = input_ids`` when the template lacks ``{% generation %}``
    (mask all-zero) or the kwarg is unsupported.
    """
    kw = {}
    if tools:
        kw["tools"] = tools
    if chat_template is not None:
        kw["chat_template"] = chat_template

    # First attempt: assistant-token mask.
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kw,
        )
        ids = list(out["input_ids"])
        mask = out.get("assistant_masks")
        if mask is not None and any(mask):
            labels = [tid if m else IGNORE_INDEX for tid, m in zip(ids, mask)]
            return ids, labels, True
        return ids, list(ids), False
    except Exception:
        # return_assistant_tokens_mask not accepted — try without it.
        pass

    # Fallback: plain tokenization (no assistant mask).
    import traceback as _tb
    try:
        out = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **kw)
        ids = list(out["input_ids"])
        return ids, list(ids), False
    except Exception as e:
        # Re-raise with the full traceback attached so pack() can log it.
        raise RuntimeError(
            f"{type(e).__name__}: {e}\n" + _tb.format_exc()
        ) from e


def collate_bin(docs_ids, docs_labels):
    """Build one packed-bin sample dict from a list of per-doc token/label arrays.

    Mirrors pack_stage1.py's ``collator``: concatenate ids, build per-doc reset
    position_ids, and store per-doc lengths in attention_mask.
    """
    input_ids, labels, position_ids, lengths = [], [], [], []
    for ids, labs in zip(docs_ids, docs_labels):
        L = len(ids)
        input_ids.extend(ids)
        labels.extend(labs)
        position_ids.extend(range(L))
        lengths.append(L)
    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "labels": np.array(labels, dtype=np.int64),
        "position_ids": np.array(position_ids, dtype=np.uint32),
        "attention_mask": np.array(lengths, dtype=np.uint32),
    }


def assert_invariants(sample):
    """Assert sum(attention_mask) == len(input_ids) == len(position_ids) == len(labels)."""
    n = len(sample["input_ids"])
    assert len(sample["labels"]) == n, "labels length mismatch"
    assert len(sample["position_ids"]) == n, "position_ids length mismatch"
    assert int(np.sum(sample["attention_mask"])) == n, (
        f"sum(attention_mask)={int(np.sum(sample['attention_mask']))} != len(input_ids)={n}"
    )


def pack(df, tokenizer, out_dir, max_seq_len, max_rows=None,
         chat_template=None, enable_tools=True):
    """Tokenize + multipack ``df`` into a ChiniDataset at ``out_dir``.

    Each row is rendered with its ``messages`` AND (when ``enable_tools``) the
    wrapped ``functions`` column as ``tools=``, using ``chat_template`` (stock or
    reasoning-relaxed).

    Long-document handling: a single conversation whose tokenized length exceeds
    ``max_seq_len`` is DROPPED (same policy as pack_stage1.py). We never split a
    conversation across bins, so position_ids/labels stay coherent per document.

    Returns a stats dict for the run summary.
    """
    from chinidataset import ParquetWriter

    if max_rows is not None:
        df = df.head(max_rows)

    has_functions = enable_tools and ("functions" in df.columns)

    total_rows = len(df)
    n_bins = 0
    n_docs_packed = 0
    n_dropped_long = 0
    n_dropped_empty = 0
    n_assistant_masked = 0
    n_rows_with_tools = 0
    total_tokens = 0

    cur_ids, cur_labels, cur_count = [], [], 0
    first_sample_for_check = None

    writer = ParquetWriter(out=out_dir, columns=COLUMNS, exist_ok=True)
    with writer as out:
        for i in range(total_rows):
            row = df.iloc[i]
            messages = extract_messages(
                row.get("messages") if hasattr(row, "get") else row["messages"]
            )
            if not messages:
                n_dropped_empty += 1
                continue

            tools = extract_tools(row["functions"]) if has_functions else []
            if tools:
                n_rows_with_tools += 1

            try:
                ids, labels, used_mask = tokenize_row(
                    tokenizer, messages, tools=tools, chat_template=chat_template
                )
            except Exception as e:  # noqa: BLE001
                print(f"[pack_dataset] row {i}: tokenize failed "
                      f"({type(e).__name__}: {e}); skipping", flush=True)
                n_dropped_empty += 1
                continue

            length = len(ids)
            if length == 0:
                n_dropped_empty += 1
                continue
            if length > max_seq_len:
                n_dropped_long += 1
                continue

            if used_mask:
                n_assistant_masked += 1

            if cur_count + length > max_seq_len:
                if cur_ids:
                    sample = collate_bin(cur_ids, cur_labels)
                    assert_invariants(sample)
                    if first_sample_for_check is None:
                        first_sample_for_check = sample
                    out.write(sample)
                    n_bins += 1
                    total_tokens += len(sample["input_ids"])
                cur_ids, cur_labels, cur_count = [ids], [labels], length
            else:
                cur_ids.append(ids)
                cur_labels.append(labels)
                cur_count += length
            n_docs_packed += 1

            if (i + 1) % 100 == 0:
                print(f"[pack_dataset] processed {i + 1}/{total_rows} rows, "
                      f"{n_bins} bins so far", flush=True)

        # Flush the final partial bin.
        if cur_ids:
            sample = collate_bin(cur_ids, cur_labels)
            assert_invariants(sample)
            if first_sample_for_check is None:
                first_sample_for_check = sample
            out.write(sample)
            n_bins += 1
            total_tokens += len(sample["input_ids"])

    efficiency = (total_tokens / n_bins / max_seq_len) if n_bins else 0.0
    return {
        "total_rows": total_rows,
        "docs_packed": n_docs_packed,
        "dropped_long": n_dropped_long,
        "dropped_empty": n_dropped_empty,
        "assistant_masked_rows": n_assistant_masked,
        "rows_with_tools": n_rows_with_tools,
        "n_bins": n_bins,
        "total_tokens": total_tokens,
        "max_seq_len": max_seq_len,
        "efficiency": efficiency,
        "first_sample": first_sample_for_check,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pack a HF chat-parquet dataset into ChiniDataset multipacking format "
                    "for Qwen3.5 training."
    )
    parser.add_argument("--out", default="./packed_data",
                        help="Output ChiniDataset directory (default: ./packed_data).")
    parser.add_argument("--max-seq-len", type=int, default=131072,
                        help="Max tokens per packed bin (default: 131072 = 128k).")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help=f"Tokenizer to load (tokenizer only, no weights). "
                             f"Default: {DEFAULT_TOKENIZER}.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap source rows for a quick test (default: all).")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"HF dataset repo (default: {DEFAULT_REPO}).")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help=f"Parquet file within the repo (default: {DEFAULT_FILE}).")
    parser.add_argument("--native-reasoning", action="store_true",
                        help="Use the stock Qwen3.5 template unchanged (may suppress "
                             "reasoning on most turns). Default relaxes the guard to train "
                             "ALL model turn reasoning.")
    parser.add_argument("--no-tools", action="store_true",
                        help="Do NOT pass the `functions` column as tools to the chat template.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"[pack_dataset] loading tokenizer {args.tokenizer} (tokenizer only)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    all_reasoning = not args.native_reasoning
    chat_template = build_chat_template(tokenizer, all_reasoning=all_reasoning)
    print(
        f"[pack_dataset] reasoning: "
        f"{'ALL model turns' if all_reasoning else 'stock (native template)'}"
        f" | tools: {'OFF' if args.no_tools else 'from functions column'}",
        flush=True,
    )

    df = load_source_dataframe(args.repo, args.file)
    print(f"[pack_dataset] source rows: {len(df)}; columns: {list(df.columns)}", flush=True)
    if "messages" not in df.columns:
        raise SystemExit("[pack_dataset] ERROR: source parquet has no 'messages' column")
    if not args.no_tools and "functions" not in df.columns:
        print("[pack_dataset] WARN: no 'functions' column -- packing without tool definitions",
              flush=True)

    stats = pack(df, tokenizer, args.out, args.max_seq_len, max_rows=args.max_rows,
                 chat_template=chat_template, enable_tools=not args.no_tools)

    # ----- Summary -----------------------------------------------------------
    print("\n========== PACK SUMMARY ==========", flush=True)
    print(f"source rows seen        : {stats['total_rows']}")
    print(f"docs packed             : {stats['docs_packed']}")
    print(f"dropped (too long)      : {stats['dropped_long']} "
          f"(len > max_seq_len={stats['max_seq_len']}; dropped, never split)")
    print(f"dropped (empty/bad)     : {stats['dropped_empty']}")
    print(f"rows with tool defs     : {stats['rows_with_tools']}")
    print(f"packed bins written     : {stats['n_bins']}")
    print(f"total packed tokens     : {stats['total_tokens']}")
    mean_per_bin = (stats['total_tokens'] / stats['n_bins']) if stats['n_bins'] else 0
    print(f"mean tokens / bin       : {mean_per_bin:.1f}")
    print(f"packing efficiency      : {stats['efficiency'] * 100:.1f}% "
          f"(mean tokens-per-bin / max_seq_len)")
    if stats["assistant_masked_rows"] > 0:
        print(f"label masking           : assistant-only on "
              f"{stats['assistant_masked_rows']} rows (template has {{% generation %}})")
    else:
        print("label masking           : FALLBACK labels = input_ids "
              "(template lacks {% generation %}; assistant mask all-zero) "
              "-> trains on the full packed sequence")
    print("==================================\n", flush=True)

    # ----- Read back + verify invariants on a few samples --------------------
    from chinidataset import StreamingDataset

    ds = StreamingDataset(local=args.out)
    print(f"[pack_dataset] read back {len(ds)} packed records from {args.out}", flush=True)
    n_check = min(3, len(ds))
    for idx in range(n_check):
        s = ds[idx]
        ii = np.asarray(s["input_ids"])
        lab = np.asarray(s["labels"])
        pos = np.asarray(s["position_ids"])
        am = np.asarray(s["attention_mask"])
        n = len(ii)
        assert len(lab) == n, f"sample {idx}: labels length mismatch"
        assert len(pos) == n, f"sample {idx}: position_ids length mismatch"
        assert int(am.sum()) == n, (
            f"sample {idx}: sum(attention_mask)={int(am.sum())} != len(input_ids)={n}"
        )
        off = 0
        for L in am.tolist():
            seg = pos[off:off + int(L)]
            assert list(seg) == list(range(int(L))), (
                f"sample {idx}: position_ids do not reset per doc at offset {off}"
            )
            off += int(L)
        print(f"[pack_dataset] sample {idx}: len(input_ids)={n} "
              f"docs={len(am)} sum(am)={int(am.sum())} OK", flush=True)

    print("[pack_dataset] invariants verified. Done.", flush=True)


if __name__ == "__main__":
    main()

EOF

cat << EOF > train.py
from transformers import (
    AutoConfig,
    Qwen3_5ForConditionalGeneration,
)
from transformers.cache_utils import Cache
from transformers.models.qwen3_5 import modeling_qwen3_5
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import logging 
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, fsdp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
    offload_wrapper
)
import os
import json
import time
from functools import partial
from tqdm import tqdm
import argparse
from chinidataset import StreamingDataset
from flash_qla import chunk_gated_delta_rule


logging.basicConfig(
    level=logging.INFO, 
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
apply_liger_kernel_to_qwen3_5(
    rms_norm=True,
    swiglu=True,          # fuses gate+act+up, cuts MLP peak memory ~50%
    cross_entropy=False,  # you're already using LigerFusedLinearCrossEntropyLoss manually
)


def ddp_setup():
    # cpu:gloo + cuda:nccl. With CPUOffloadPolicy the LoRA params are DTensors whose shards live
    # on CPU; the checkpoint's full_tensor() all-gather is then a CPU collective that NCCL can't
    # service ("No backend type associated with device type cpu"). gloo handles CPU collectives;
    # nccl stays the fast path for the GPU all-gather/reduce-scatter during forward/backward.
    init_process_group(backend="cpu:gloo,cuda:nccl")


class Dataset(Dataset):
    def __init__(self, limit: int = 0):
        self.dataset = StreamingDataset(local="./packed_data")
        # limit > 0 caps the dataset to the first N packed bins.
        self._len = min(limit, len(self.dataset)) if limit and limit > 0 else len(self.dataset)
        # NOTE: training consumes packed bins AS-IS (no truncation here). The packed-bin length is
        # controlled at dataset-prep time (pack_dataset.py --max-seq-len) so bins are both trainable
        # AND contain the actual conversation turns. (Truncating here to a small window only fed the
        # model the system/tool-schema preamble — the conversation starts ~25k tokens in — which is
        # why the earlier finetune degenerated.)

    def __getitem__(self, idx):
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "attention_mask": self.dataset[idx]["attention_mask"],  # np.array of per-doc lengths
            "labels": self.dataset[idx]["labels"],
            "position_ids": self.dataset[idx]["position_ids"],
        }

    def __len__(self):
        return self._len

def collator(batch):
    batch = [b for b in batch if b is not None]
    input_ids = [b['input_ids'] for b in batch]
    position_ids = [b['position_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]

    # query_lens = the length of every packed document across the whole batch.
    query_lens = np.concatenate(attention_mask)

    input_ids = np.concatenate(input_ids)
    position_ids = np.concatenate(position_ids)
    labels = np.concatenate(labels)

    cumsum = [0] + np.cumsum(query_lens).tolist()
    cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
    max_seqlen_q = int(np.max(query_lens))

    # No dense (1, S, S) attention_mask: the packing is fully described by cu_seq_lens_* +
    # position_ids. dynamic_attention rebuilds the causal block-diagonal mask itself in the
    # SDPA branch and consumes cu_seqlens directly in the FA3 branch. A dense (1, S, S) mask
    # both wastes O(S^2) memory and confuses transformers' mask preparation (it expects a
    # 2D padding mask or a 4D causal mask, not 3D).
    input_ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    return {
        'input_ids': input_ids_t,
        'position_ids': torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
        'attention_mask': None,
        # Gemma-4 is multimodal; create_causal_mask_mapping requires mm_token_type_ids during
        # training. Text-only packing => all-zero (every token is a text token).
        'mm_token_type_ids': torch.zeros_like(input_ids_t),
        'labels': torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        'cu_seq_lens_q': cu_seq_lens_q,
        'cu_seq_lens_k': cu_seq_lens_q,
        'max_length_q': max_seqlen_q,
        'max_length_k': max_seqlen_q
    }

def apply_linear_lora(base_model: nn.Module, r: int = 8, alpha:int = 16): 
    # no nn.Linear instance for up_proj, down_proj, gate_proj after attention block
    linear_layers = [
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj"
    ]
    for name, module in list(base_model.named_modules()):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and child_name in linear_layers:
                if 'vision' in name:
                    continue 

                lora = LinearLoRA(child_module, r, alpha)
                setattr(module, child_name, lora)

class CustomQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            mm_token_type_ids: torch.IntTensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs
        ):
        
        # Text-only packed training: pass just the essentials + **kwargs (carries the packing
        # metadata cu_seq_lens_q/k + max_length_q/k through to dynamic_attention). The multimodal
        # inputs are all None here, and per_layer_inputs must NOT be passed — Gemma4Model computes
        # it internally and re-passes it to the language model ("got multiple values for keyword
        # argument 'per_layer_inputs'" otherwise).
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        # overwrite to disable the logits materializatioon

        loss = None
        if labels is not None:
            # B, S, D -> (B*S, D)
            shifted_hidden_states = hidden_states[:, :-1, :].contiguous().reshape(-1, hidden_states.shape[-1])
            shifted_labels = labels[:, 1:].contiguous().reshape(-1)
            loss = self.loss_fn(
                self.lm_head.weight,
                shifted_hidden_states, 
                shifted_labels,
            )
        else:
            raise NotImplementedError("Loss calculation is not implemented yet.")

        return {
            "loss": loss
        }

class LinearLoRA(nn.Module): 
    def __init__(self, linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_a = nn.Linear(in_features, r, bias=False, dtype=torch.bfloat16)
        self.lora_b = nn.Linear(r, out_features, bias=False, dtype=torch.bfloat16)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a.weight)
            nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        out_non_lora = self.linear(x)
        lora_out = self.lora_b(self.lora_a(x))
        return out_non_lora + self.scaling * lora_out

def main(
        r:int=256,
        alpha: int=512,
        batch_size:int = 2,
        max_steps:int = 0,
        checkpointing_step:int = 100,
        limit_samples:int = 0,
        max_epochs:int = 1,
        lr:float = 1e-4,
        use_wandb:bool = False,
        wandb_project:str = "qwen3.5-autotrain",
        use_mlflow:bool = False,
        mlflow_experiment:str = "qwen3.5-autotrain",
    ):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Pin this process to its GPU BEFORE init'ing NCCL / the device mesh, otherwise every
    # rank lands on cuda:0 (NCCL hang / OOM).
    torch.cuda.set_device(rank)
    ddp_setup()  # init_process_group(nccl) — required before init_device_mesh / fully_shard
    mesh_device = init_device_mesh("cuda", (world_size, ), mesh_dim_names=("shard", ))
    torch.cuda.memory._record_memory_history()

    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-27B")
    # modify the confg
    model = CustomQwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-27B", 
        config=config, 
        dtype=torch.bfloat16, # native bf16 training
        attn_implementation = "kernels-community/flash-attn3"
    )
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {total_params/(1024*1024):.0f}M")

    for param in model.parameters():
        param.requires_grad = False
    apply_linear_lora(model, r=r, alpha=alpha)

    total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable_params/(1024*1024):.2f}M")

    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    fsdp_kwargs["offload_policy"] = fsdp.CPUOffloadPolicy()
    shard_modules = (
        modeling_qwen3_5.Qwen3_5DecoderLayer, 
        # modeling_qwen3_5.Qwen3_5MLP,
        # modeling_qwen3_5.Qwen3_5GatedDeltaNet, 
        # modeling_qwen3_5.Qwen3_5Attention, 
        modeling_qwen3_5.Qwen3_5VisionBlock
    )

    for module in model.modules():
        if isinstance(module, shard_modules):
            fsdp.fully_shard(module, **fsdp_kwargs, mesh=mesh_device, reshard_after_forward=True) # shard on submodules
    fsdp.fully_shard(model, **fsdp_kwargs, mesh=mesh_device) # full shard on root module

    # Get the local shard
    model_sd = model.state_dict()
    local_shard = sum(v.to_local().numel() if hasattr(v, "to_local") else v.numel() for _ , v in model_sd.items())
    logger.info(f"[Rank{rank}]: Total local/shard param: {local_shard/(1024*1024):.2f}M")

    # checkpointing module 
    checkpointing_modules = [
        # modeling_qwen3_5.Qwen3_5TextModel
        modeling_qwen3_5.Qwen3_5DecoderLayer,
        # modeling_qwen3_5.Qwen3_5VisionBlock,
        # modeling_qwen3_5.Qwen3_5VisionMLP,
        # modeling_qwen3_5.Qwen3_5VisionAttention,
        # modeling_qwen3_5.Qwen3_5MLP,
        # modeling_qwen3_5.Qwen3_5GatedDeltaNet, 
        # modeling_qwen3_5.Qwen3_5Attention
    ]
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model, 
        checkpoint_wrapper_fn = non_reentrant_wrapper,
        check_fn=lambda x: isinstance(x, tuple(checkpointing_modules))
    )

    def _patched_chunk_gated_delta_rule(q, k, v, *args, **kwargs):
        # TileLang kernel requires v to be contiguous (stride[-1] == 1)
        v = v.contiguous()
        return chunk_gated_delta_rule(q, k, v, *args, **kwargs)

    # use qwen flash qla
    for module in model.modules():
        if isinstance(module, modeling_qwen3_5.Qwen3_5GatedDeltaNet):
            module.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule
    
    # max_steps > 0 caps the run regardless of epochs (0 = run all max_epochs). For overfitting a
    # small dataset, set max_epochs high and max_steps 0.
    dataset = Dataset(limit=limit_samples)
    sampler = DistributedSampler(
        dataset,
        num_replicas=mesh_device.size(),
        rank=mesh_device.get_rank(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
        prefetch_factor=4,
        num_workers=4,
    )

    # Optimize ONLY the trainable LoRA params. fused=False: with CPUOffloadPolicy the
    # optimizer step runs on CPU-resident params, and fused AdamW is CUDA-only.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, fused=False)
    model.train()

    # ---- wandb (rank 0 only) ----
    wandb_run = None
    if use_wandb and rank == 0:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "model": "Qwen/Qwen3.5-27B", "r": r, "alpha": alpha,
                "batch_size": batch_size, "lr": lr, "max_epochs": max_epochs,
                "max_steps": max_steps,
                "limit_samples": limit_samples, "world_size": world_size,
                "num_bins": len(dataset), "trainable_params_M": round(total_trainable_params/1e6, 2),
            },
        )
        logger.info(f"wandb run: {wandb_run.url}")
    # ---- mlflow (rank 0 only) ----
    if use_mlflow and rank == 0:
        import mlflow
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run(run_name=f"Qwen3.5-27B_r{r}_alpha{alpha}_bs{batch_size}_lr{lr}")
        mlflow.log_params({
            "model": "Qwen/Qwen3.5-27B", "r": r, "alpha": alpha,
            "batch_size": batch_size, "lr": lr, "max_epochs": max_epochs,
            "max_steps": max_steps,
            "limit_samples": limit_samples, "world_size": world_size,
            "num_bins": len(dataset), "trainable_params_M": round(total_trainable_params/1e6, 2),
        })

    # seen_tokens = torch.tensor(0, dtype=torch.long, device=f'cuda:{rank}')
    for i in range(max_epochs):
        sampler.set_epoch(i)  # reshuffle each epoch
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            global_step = i * len(dataloader) + idx
            if rank == 0:
                start_time = time.time()

            # Move the packed batch onto this rank's GPU (collator builds CPU tensors).
            # Python ints (max_length_q/k) are left as-is.
            batch = {
                k: (v.to(f'cuda:{rank}', non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            output = model(**batch, use_cache=False) # forward pass and calculate losses
            output["loss"].backward() # calculate gradient 
            
            optimizer.step()
            optimizer.zero_grad()

            # synchronize
            # last_seen_tokens = seen_tokens.item()
            token_count = batch["input_ids"].numel()
            token_count = torch.tensor(token_count, device=f'cuda:{rank}')
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
            # seen_tokens += token_count

            if rank == 0:
                loss = output['loss'].item()
                delta_time = time.time() - start_time
                tps = token_count.item() / delta_time
                logger.info(f"Epoch: {i}, mb: {idx}, step: {global_step}, loss: {loss}, tokens/s: {tps:.2f}")
                metrics = {"loss": loss, "lr": optimizer.param_groups[0]['lr'], "tps": tps, "epoch": i}
                if wandb_run is not None:
                    try:
                        wandb_run.log(metrics, step=global_step)
                    except Exception as e:
                        logger.warning(f"wandb.log failed (continuing): {e}")
                if use_mlflow:
                    try:
                        mlflow.log_metrics(metrics, step=global_step)
                    except Exception as e:
                        logger.warning(f"mlflow.log_metrics failed (continuing): {e}")
            
            if global_step == 1:
                torch.cuda.memory._dump_snapshot(f"memory_profile_{rank}.pickle")

            reached_max = max_steps > 0 and (global_step + 1) >= max_steps
            if idx == len(dataloader)-1 or (idx+1) % checkpointing_step == 0 or reached_max:
                logger.info("Checkpointing LoRA adapters..")
                # Save only the trainable LoRA params. full_tensor() is a collective, so every rank
                # must iterate the SAME params; requires_grad is identical across ranks.
                lora_state_dict = {}
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    full_param = param.full_tensor() if hasattr(param, "full_tensor") else param
                    if rank == 0:
                        lora_state_dict[name] = full_param.detach().to('cpu')
                if rank == 0:
                    os.makedirs("checkpointing", exist_ok=True)
                    torch.save(lora_state_dict, "checkpointing/lora.pt")
                    with open("checkpointing/lora_meta.json", "w") as f:
                        json.dump({
                            "model_id": "Qwen/Qwen3.5-27B",
                            "r": r, "alpha": alpha, "scaling": alpha / r,
                            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                            "wrapped_attr": "linear",
                        }, f, indent=2)
                    logger.info(f"Saved LoRA ({len(lora_state_dict)} tensors) to checkpointing/lora.pt")

            if reached_max:
                logger.info(f"Reached max_steps={max_steps}, stopping.")
                break
        if reached_max:
            break

    if wandb_run is not None:
        wandb_run.finish()
    destroy_process_group()
    
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank", 
        type=int, 
        default=256, 
        help="LoRA rank"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=512,
        help="LoRA alpha scaling factor"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps (0 = full epoch). Use a small value to "
             "produce a LoRA checkpoint quickly for merge/inference validation.",
    )
    parser.add_argument(
        "--checkpointing_step",
        type=int,
        default=100,
        help="Save the LoRA adapters every N steps.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="Cap the dataset to the first N packed bins (0 = all). Use a small N to deliberately "
             "overfit a tiny subset as an end-to-end sanity check.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs over the dataset (set high to overfit a small dataset).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow.")
    parser.add_argument("--wandb_project", default="gemma4-autotrain", help="wandb project name.")
    parser.add_argument("--mlflow_experiment", default="gemma4-autotrain", help="mlflow experiment name.")
    args = parser.parse_args()
    main(
        args.rank,
        args.alpha,
        args.batch_size,
        args.max_steps,
        args.checkpointing_step,
        args.limit_samples,
        args.max_epochs,
        args.lr,
        args.wandb,
        args.mlflow,
        args.wandb_project,
        args.mlflow_experiment,
    )



EOF

python pack_dataset.py --out ./packed_data --max-seq-len 50000
torchrun --nproc_per_node=2 train.py \
    --batch_size 1 \
    --max_steps 1000 \
    --checkpointing_step 100 \
    --limit_samples 10 \
    --max_epochs 1 \
    --lr 1e-4 \
    --rank 256 \
    --alpha 512 
