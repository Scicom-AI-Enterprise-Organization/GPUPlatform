#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling
# task. Pointers for this are left as comments.

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import transformers
import random
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AddedToken,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers import Qwen3ForCausalLM
import json
import numpy as np
# Vendored ChiniDataset (shipped next to this script) — no pip/git on the box.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chinidataset import StreamingDataset
from cut_cross_entropy import linear_cross_entropy
from liger_kernel.transformers import apply_liger_kernel_to_qwen3, LigerFusedLinearCrossEntropyLoss

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

apply_liger_kernel_to_qwen3(
    rope=True,
    swiglu=True,
    rms_norm=True,
    cross_entropy=False,
    fused_linear_cross_entropy=False,
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " +
            ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")}, )
    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"}, )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine.")}, )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "auto",
                "bfloat16",
                "float16",
                "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    # ---- LoRA (peft) ----------------------------------------------------
    use_lora: bool = field(
        default=False, metadata={"help": "Wrap the model in LoRA adapters (peft) instead of full finetuning."})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (absolute)."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout on the adapters."})
    lora_target_modules: str = field(
        default="all-linear",
        metadata={"help": "Comma-separated module names, or 'all-linear' for every nn.Linear (excl. the output proj)."})

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    skip_eval: bool = field(
        default=False, metadata={
            "help": "Disable evaluation even if the packed dir has a test/ split (the 'No test set' option)."})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

class Model(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")

    def _lm_head_weight(self):
        """The lm_head weight to project hidden states with in the fused loss.
        Normally a plain nn.Linear (frozen under LoRA — we don't adapt the head).
        Defensive: if the head is ever kept trainable via peft's modules_to_save
        (ModulesToSaveWrapper), return the active copy; otherwise the base weight."""
        head = self.lm_head
        msd = getattr(head, "modules_to_save", None)
        if msd is not None:
            active = getattr(head, "active_adapter", None)
            if isinstance(active, (list, tuple)):
                active = active[0] if active else None
            if active is not None and active in msd:
                return msd[active].weight
        return head.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        # peft's PeftModelForCausalLM.forward injects output_hidden_states /
        # output_attentions / return_dict / inputs_embeds into kwargs; drop them
        # so they don't collide with the explicit output_hidden_states=True (and
        # so a None inputs_embeds doesn't fight the real input_ids).
        for _k in ("output_hidden_states", "output_attentions", "return_dict", "inputs_embeds"):
            kwargs.pop(_k, None)
        super_out = self.model.forward(
            input_ids = input_ids,
            position_ids = position_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            embeddings = embeddings[:,:-1].reshape(-1, embeddings.shape[-1])
            labels = labels[..., 1:].contiguous()
            labels = labels.reshape(-1)
            loss = self.loss(self._lm_head_weight(), embeddings, labels)
            num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
            return {'loss': loss}
        return super_out

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    extra = [AddedToken('<|speech_start|>')]
    for i in range(65536):
        extra.append(AddedToken(f'<|s_{i}|>'))
    tokenizer.add_tokens(extra)
    
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    min_dtype = torch.finfo(torch_dtype).min
    sequence_length = data_args.block_size

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local, split=None):
            self.dataset = StreamingDataset(local=local, split=split)

        def __getitem__(self, idx):
            data = dict(self.dataset[idx])
            data.pop('audio', None)
            data.pop('text', None)
            data.pop('token_type_ids', None)

            # ChiniDataset returns array columns as numpy already, but be robust
            # to list-typed values.
            for k in list(data.keys()):
                data[k] = np.asarray(data[k])

            if data['attention_mask'].max() > sequence_length:
                print(data)
                return

            for k in data.keys():
                data[k] = data[k].astype(np.int64)

            return data

        def __len__(self):
            return len(self.dataset)

    # This model packs multiple utterances per block and relies on flash-attn
    # VARLEN attention (the collator passes cu_seq_lens_q/k) to keep them from
    # attending across pack boundaries. Only flash-attn backends honour that —
    # sdpa/eager would silently mis-attend across the packed sequence — so we use
    # the FA3 Hopper hub kernel (H20 is sm_90), fall back only to FA2, and fail
    # loudly otherwise rather than degrade to a backend that breaks packing.
    model = None
    for _ai in ('kernels-community/vllm-flash-attn3', 'flash_attention_2'):
        try:
            model = Model.from_pretrained(
                model_args.model_name_or_path,
                attn_implementation=_ai,
                torch_dtype=model_args.torch_dtype,
            )
            print(f'[qwen3_tts] attn_implementation={_ai}', flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f'[qwen3_tts] attn {_ai!r} unavailable: {e}', flush=True)
    if model is None:
        raise RuntimeError(
            'no flash-attn backend available (need the `kernels` package for FA3, or '
            '`flash_attn` for FA2). This model packs utterances and requires flash-attn '
            'varlen attention — sdpa/eager would mis-attend across pack boundaries.'
        )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
    print(model)

    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model

        tgt = (model_args.lora_target_modules or "all-linear").strip()
        target_modules = (
            "all-linear" if tgt in ("", "all-linear", "all")
            else [t.strip() for t in tgt.split(",") if t.strip()]
        )
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # Adapt all linear layers only. peft's "all-linear" auto-excludes the
            # output proj (lm_head); embed_tokens is an nn.Embedding so it's never
            # a target — embeddings + lm_head stay frozen at their base weights.
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        # Gradient checkpointing + (mostly) frozen base needs the embedding output
        # to require grad so the checkpointed graph stays connected.
        if getattr(training_args, "gradient_checkpointing", False):
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    # Split-aware packed dataset keeps train/ + test/ subdirs (StreamingDataset
    # reads local/<split>/); a flat packed dir (legacy) has shards at the root.
    # Train on `train`, evaluate on `test` when present.
    _tf = data_args.train_file
    _has_train = os.path.isdir(os.path.join(_tf, 'train'))
    _has_test = os.path.isdir(os.path.join(_tf, 'test'))
    _has_flat = os.path.exists(os.path.join(_tf, 'index.json'))
    # Resolve the training split: an explicit `train/` → a flat pack (index.json at
    # the root) → else a single named split (e.g. `default`, produced when the source
    # dataset had no split column). Without the last case a `default`-only pack hits
    # "index.json not found at <packed>/index.json".
    if _has_train:
        _train_split = 'train'
    elif _has_flat:
        _train_split = None
    else:
        _subs = [d for d in sorted(os.listdir(_tf))
                 if os.path.isdir(os.path.join(_tf, d))
                 and os.path.exists(os.path.join(_tf, d, 'index.json'))]
        _train_split = _subs[0] if _subs else None
    dataset = DatasetFixed(_tf, split=_train_split)
    # "No test set" (--skip_eval) → never evaluate, even when a test/ split exists.
    eval_dataset = DatasetFixed(_tf, split='test') if (_has_test and not data_args.skip_eval) else None
    print('dataset', len(dataset), dataset[0]['attention_mask'].shape,
          '| eval', (len(eval_dataset) if eval_dataset is not None else 0),
          '| skip_eval', data_args.skip_eval)
    def _evs_name(a):
        return str(getattr(a, "eval_strategy", None) or getattr(a, "evaluation_strategy", "") or "").split(".")[-1].lower()
    if eval_dataset is not None:
        # Per-epoch/step eval loss on the held-out test split. Respect the cadence
        # the orchestrator passed (--eval_strategy / --eval_steps); only default to
        # 'epoch' when it wasn't set (HF's default is 'no').
        training_args.do_eval = True
        if _evs_name(training_args) in ("", "no"):
            try:
                training_args.eval_strategy = 'epoch'
            except Exception:
                training_args.evaluation_strategy = 'epoch'
    else:
        # No test split → don't evaluate (a passed --eval_strategy steps would
        # otherwise crash the Trainer with no eval_dataset).
        training_args.do_eval = False
        try:
            training_args.eval_strategy = 'no'
        except Exception:
            training_args.evaluation_strategy = 'no'

    def collator(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [b['input_ids'] for b in batch]
        position_ids = [b['position_ids'] for b in batch]
        labels = [b['input_ids'].copy() for b in batch]
        attention_mask = [b['attention_mask'] for b in batch]
        input_ids = np.concatenate(input_ids)
        position_ids = np.concatenate(position_ids)
        labels = np.concatenate(labels)
        query_lens = np.concatenate(attention_mask)
        cumsum = [0] + np.cumsum(query_lens).tolist()
        max_cumsum = int(np.max(cumsum))
        cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
        cu_seq_lens_k = torch.tensor(cumsum, dtype=torch.int32)
        max_seqlen_q = np.max(query_lens)
        return {
            'input_ids': torch.tensor(input_ids)[None],
            'position_ids': torch.tensor(position_ids)[None],
            'labels': torch.tensor(labels)[None],
            'cu_seq_lens_q': cu_seq_lens_q,
            'cu_seq_lens_k': cu_seq_lens_k,
            'max_length_q': max_seqlen_q,
            'max_length_k': max_seqlen_q
        }

    # Emit @@STEP (per-log train loss) + @@METRIC (per-epoch eval loss on the test
    # split) so the gateway's live loss curve has points — HF's default
    # `{'loss': …}` console log isn't parsed by the gateway.
    class _ProgressEmitter(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or not state.is_world_process_zero:
                return
            if "loss" in logs:
                print("@@STEP " + json.dumps({
                    "step": int(state.global_step),
                    "loss": float(logs["loss"]),
                    "lr": float(logs.get("learning_rate") or 0.0),
                    "epoch": float(logs.get("epoch") or 0.0),
                }), flush=True)
            if "eval_loss" in logs:
                print("@@METRIC " + json.dumps({
                    "epoch": float(logs.get("epoch") or 0.0),
                    "eval_loss": float(logs["eval_loss"]),
                }), flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        callbacks=[_ProgressEmitter()],
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        if model_args.use_lora:
            # Merge LoRA adapters back into the base → a plain Qwen3 checkpoint
            # that downstream eval and serving load with no peft (embed/lm_head
            # are unchanged — they were frozen). merge_and_unload runs per-rank
            # (local op); only the main process writes the artifact.
            to_save = trainer.accelerator.unwrap_model(trainer.model)
            to_save = to_save.merge_and_unload()
            if training_args.should_save:
                to_save.save_pretrained(training_args.output_dir, safe_serialization=True)
                tokenizer.save_pretrained(training_args.output_dir)
        else:
            trainer.save_model()
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()