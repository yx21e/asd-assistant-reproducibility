#!/usr/bin/env python3
"""
Train a lightweight task expert for Autism GPT SFT data.

The script uses plain Hugging Face Trainer so it stays compatible with the
current MVP environment. LoRA is optional through PEFT.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gene_registry import filter_rows_by_gene, resolve_gene_filter


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_limit(rows: List[dict], max_rows: Optional[int], seed: int) -> List[dict]:
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:max_rows]


def split_rows(rows: List[dict], eval_ratio: float, seed: int) -> tuple[List[dict], List[dict]]:
    if len(rows) < 2:
        return rows, rows
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    eval_size = max(1, int(round(len(rows) * eval_ratio)))
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:] or rows[:1]
    return train_rows, eval_rows


def detect_lora_targets(model_name: str) -> List[str]:
    name = model_name.lower()
    if any(token in name for token in ("llama", "mistral", "qwen", "gemma")):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "gpt2" in name:
        return ["c_attn", "c_proj"]
    if "phi" in name:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_tokenize_fn(tokenizer, max_length: int):
    eos = tokenizer.eos_token or ""

    def tokenize_row(row: dict) -> dict:
        prompt_text = row["prompt_text"]
        full_text = prompt_text + row["response"] + eos

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "response_tokens": max(0, len(input_ids) - prompt_len),
        }

    return tokenize_row


def has_response_tokens(row: dict) -> bool:
    return row.get("response_tokens", 0) > 0


def keep_model_columns(ds: Dataset) -> Dataset:
    keep = {"input_ids", "attention_mask", "labels", "response_tokens"}
    drop = [col for col in ds.column_names if col not in keep]
    return ds.remove_columns(drop) if drop else ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a task expert on Autism GPT SFT data.")
    parser.add_argument("--train-jsonl", required=True, help="Path to task-specific SFT JSONL.")
    parser.add_argument("--eval-jsonl", default=None, help="Optional separate eval JSONL.")
    parser.add_argument("--out-dir", required=True, help="Output directory for model artifacts.")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2", help="Base causal LM.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--gene-registry", default=None, help="Optional gene registry file for task1/task2 filtering.")
    parser.add_argument("--gene-list", default=None, help="Optional explicit gene list file for task1/task2 filtering.")
    parser.add_argument("--support-status", default=None, help="Optional comma-separated support status filter.")
    parser.add_argument("--tiers", default=None, help="Optional comma-separated tier filter.")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=None, help="Comma-separated override for LoRA target modules.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-compute-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--save-strategy", default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    allowed_genes = resolve_gene_filter(
        gene_registry=args.gene_registry,
        gene_list=args.gene_list,
        support_statuses=args.support_status,
        tiers=args.tiers,
    )

    train_rows = filter_rows_by_gene(read_jsonl(Path(args.train_jsonl)), allowed_genes)
    if args.eval_jsonl:
        eval_rows = filter_rows_by_gene(read_jsonl(Path(args.eval_jsonl)), allowed_genes)
    else:
        train_rows, eval_rows = split_rows(train_rows, eval_ratio=args.eval_ratio, seed=args.seed)

    train_rows = maybe_limit(train_rows, args.max_train_samples, args.seed)
    eval_rows = maybe_limit(eval_rows, args.max_eval_samples, args.seed + 1)

    if not train_rows:
        raise ValueError("No training rows available after filtering.")
    if not eval_rows:
        eval_rows = train_rows[:1]

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_load_kwargs = {"trust_remote_code": args.trust_remote_code}
    selected_dtype = None
    if args.model_dtype == "auto":
        if torch.cuda.is_available():
            selected_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        selected_dtype = dtype_map[args.model_dtype]
    if selected_dtype is not None:
        model_load_kwargs["torch_dtype"] = selected_dtype

    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "--load-in-4bit requires bitsandbytes and transformers BitsAndBytesConfig support."
            ) from exc
        compute_dtype = dtype_map[args.bnb_4bit_compute_dtype]
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_load_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False

    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError("PEFT is required for --use-lora. Install `peft` first.") from exc

        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        else:
            target_modules = detect_lora_targets(args.model_name)

        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
        if args.gradient_checkpointing:
            model.enable_input_require_grads()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenize_fn = build_tokenize_fn(tokenizer=tokenizer, max_length=args.max_length)
    train_ds = Dataset.from_list(train_rows).map(tokenize_fn)
    eval_ds = Dataset.from_list(eval_rows).map(tokenize_fn)
    train_ds = train_ds.filter(has_response_tokens)
    eval_ds = eval_ds.filter(has_response_tokens)
    train_ds = keep_model_columns(train_ds)
    eval_ds = keep_model_columns(eval_ds)

    if len(train_ds) == 0:
        raise ValueError("All train samples lost their response tokens after truncation. Increase --max-length.")
    if len(eval_ds) == 0:
        eval_ds = train_ds.select(range(min(1, len(train_ds))))

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targs_kwargs = {
        "output_dir": str(out_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
        "load_best_model_at_end": False,
    }
    if args.save_strategy == "steps" and args.save_steps > 0:
        targs_kwargs["save_steps"] = args.save_steps
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_args_params:
        targs_kwargs["evaluation_strategy"] = "epoch"
    else:
        targs_kwargs["eval_strategy"] = "epoch"

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            targs_kwargs["bf16"] = True
        else:
            targs_kwargs["fp16"] = True
    if args.gradient_checkpointing:
        targs_kwargs["gradient_checkpointing"] = True

    targs = TrainingArguments(**targs_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": targs,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    summary = {
        "model_name": args.model_name,
        "gene_filter_count": len(allowed_genes) if allowed_genes is not None else None,
        "gene_registry": args.gene_registry,
        "gene_list": args.gene_list,
        "support_status": args.support_status,
        "tiers": args.tiers,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_tokens_retained": len(train_ds),
        "eval_tokens_retained": len(eval_ds),
        "use_lora": bool(args.use_lora),
        "model_dtype": args.model_dtype,
        "load_in_4bit": bool(args.load_in_4bit),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    write_json(out_dir / "train_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
