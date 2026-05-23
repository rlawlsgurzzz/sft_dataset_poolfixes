# Gemma 4 E4B를 4bit QLoRA로 SFT한다.
# train/validation messages JSONL을 prompt/completion dataset으로 변환한다.
# completion에만 loss를 걸고 epoch 1/2/3 LoRA adapter를 저장한다.
# sft_run_report.json과 sft_loss_history.csv를 생성한다.
# GGUF 변환, merge, llama.cpp export, test generation 평가는 수행하지 않는다.

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


DEFAULT_MODEL_NAME = "google/gemma-4-E4b-it"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
EXCLUDED_LOSS_HISTORY_FIELDS = {"train_samples_seen", "eval_samples_seen"}


@dataclass
class EpochSaveRecord:
    epoch: int
    adapter_dir: str
    saved_at_utc: str
    seconds_since_train_start: float


class EpochAdapterSaverCallback(TrainerCallback):
    def __init__(self, adapter_root: Path, tokenizer: Any, max_epoch: int) -> None:
        self.adapter_root = adapter_root
        self.tokenizer = tokenizer
        self.max_epoch = max_epoch
        self.saved_epochs: dict[int, EpochSaveRecord] = {}
        self.train_started_at = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if not state.is_world_process_zero:
            return control
        if state.epoch is None:
            return control

        rounded_epoch = int(round(float(state.epoch)))
        if rounded_epoch < 1 or rounded_epoch > self.max_epoch:
            return control
        if abs(float(state.epoch) - rounded_epoch) > 0.02:
            return control
        if rounded_epoch in self.saved_epochs:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        adapter_dir = self.adapter_root / f"adapter_epoch_{rounded_epoch}"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))

        record = EpochSaveRecord(
            epoch=rounded_epoch,
            adapter_dir=str(adapter_dir),
            saved_at_utc=datetime.now(timezone.utc).isoformat(),
            seconds_since_train_start=round(time.time() - self.train_started_at, 3),
        )
        self.saved_epochs[rounded_epoch] = record
        print(f"[checkpoint] saved epoch {rounded_epoch} adapter to {adapter_dir}")
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gemma 4 E4B QLoRA with train/validation messages JSONL and save epoch adapters."
    )
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--train-path", required=True, help="Train messages JSONL path.")
    parser.add_argument("--validation-path", required=True, help="Validation messages JSONL path.")
    parser.add_argument("--output-dir", default="/workspace/sft_exp/outputs/gemma4_e4b_qlora_3epoch")
    parser.add_argument("--run-name", default="gemma4_e4b_qlora_3epoch")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--data-seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--make-archive", action="store_true", help="Create a tar.gz archive of output_dir after training.")
    parser.add_argument("--archive-path", default="", help="Optional tar.gz output path.")
    return parser.parse_args()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from error
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            records.append(obj)
    return records


def assert_messages_jsonl(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    count = 0
    for line_no, obj in enumerate(read_jsonl(path), start=1):
        messages = obj.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{path}:{line_no}: missing messages list")
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{path}:{line_no}: expected roles ['system','user','assistant'], got {roles}")
        assistant_content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
        if not isinstance(assistant_content, str) or not assistant_content.strip():
            raise ValueError(f"{path}:{line_no}: empty assistant content")
        try:
            parsed = json.loads(assistant_content)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_no}: assistant content is not JSON") from error
        if set(parsed.keys()) != {"thinking", "dialog", "action"}:
            raise ValueError(f"{path}:{line_no}: assistant JSON keys invalid: {list(parsed.keys())}")
        count += 1

    if count == 0:
        raise ValueError(f"Dataset has 0 usable samples: {path}")
    print(f"[dataset] checked {count} samples from {path}")
    return count


def get_package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def run_text_command(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    text = (result.stdout or result.stderr or "").strip()
    return text or None


def get_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
        "nvidia_smi": run_text_command(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"]),
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return info


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def basic_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"avg": None, "p50": None, "p95": None, "max": None}
    return {
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def convert_to_prompt_completion(dataset, tokenizer):  # type: ignore[no-untyped-def]
    def to_prompt_completion(example):  # type: ignore[no-untyped-def]
        messages = example["messages"]
        prompt_messages = messages[:-1]
        assistant_message = messages[-1]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        completion = assistant_message["content"]
        if tokenizer.eos_token is not None and not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        return {"prompt": prompt, "completion": completion}

    converted = dataset.map(
        to_prompt_completion,
        remove_columns=dataset.column_names,
        desc="Converting messages to prompt/completion",
    )
    print("[dataset] converted:", converted)
    return converted


def compute_token_stats(records: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    over_max_count = 0

    for obj in records:
        messages = obj["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        completion = messages[-1]["content"]
        if tokenizer.eos_token is not None and not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        completion_len = len(tokenizer(completion, add_special_tokens=False).input_ids)
        total_len = prompt_len + completion_len
        prompt_lengths.append(prompt_len)
        completion_lengths.append(completion_len)
        total_lengths.append(total_len)
        if total_len > max_seq_length:
            over_max_count += 1

    return {
        "prompt_tokens": basic_stats(prompt_lengths),
        "completion_tokens": basic_stats(completion_lengths),
        "total_tokens": basic_stats(total_lengths),
        "over_max_seq_length_count": over_max_count,
        "truncated_count": 0,
        "skipped_count": 0,
    }


def make_sft_config(args: argparse.Namespace, output_dir: Path) -> SFTConfig:
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "no",
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "bf16": bf16,
        "fp16": not bf16,
        "packing": False,
        "report_to": "tensorboard",
        "completion_only_loss": True,
        "dataset_num_proc": args.dataset_num_proc,
        "remove_unused_columns": False,
    }

    params = inspect.signature(SFTConfig.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    if "max_length" in params:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = args.max_seq_length

    return SFTConfig(**kwargs)


def make_trainer(model, tokenizer, train_dataset, eval_dataset, sft_args, callback):  # type: ignore[no-untyped-def]
    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": sft_args,
        "callbacks": [callback],
    }
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not directory.exists():
        return hashes
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        hashes[str(path.relative_to(directory))] = sha256_file(path)
    return hashes


def dir_size_bytes(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def find_best_eval_checkpoint(history: list[dict[str, Any]], callback: EpochAdapterSaverCallback) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in history:
        if "eval_loss" not in row:
            continue
        try:
            eval_loss = float(row["eval_loss"])
        except Exception:
            continue
        epoch_value = row.get("epoch")
        epoch = int(round(float(epoch_value))) if epoch_value is not None else None
        candidate = {"epoch": epoch, "eval_loss": eval_loss}
        if epoch in callback.saved_epochs:
            candidate["checkpoint_path"] = callback.saved_epochs[epoch].adapter_dir
        if best is None or eval_loss < best["eval_loss"]:
            best = candidate
    return best


def save_loss_history(history: list[dict[str, Any]], report_dir: Path) -> None:
    jsonl_path = report_dir / "sft_loss_history.jsonl"
    csv_path = report_dir / "sft_loss_history.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in history:
            clean = {k: v for k, v in row.items() if k not in EXCLUDED_LOSS_HISTORY_FIELDS}
            clean.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            if "step" in clean:
                clean.setdefault("global_step", clean["step"])
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    fieldnames = sorted(
        {
            key
            for row in history
            for key in row.keys()
            if key not in EXCLUDED_LOSS_HISTORY_FIELDS
        }
        | {"timestamp", "global_step"}
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            clean = {k: v for k, v in row.items() if k not in EXCLUDED_LOSS_HISTORY_FIELDS}
            clean.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            if "step" in clean:
                clean.setdefault("global_step", clean["step"])
            writer.writerow(clean)

    print(f"[report] saved {csv_path}")
    print(f"[report] saved {jsonl_path}")


def make_archive(output_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
    sha_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sha_path.write_text(f"{sha256_file(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    print(f"[archive] saved {archive_path}")
    print(f"[archive] saved {sha_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    adapter_root = output_dir / "adapters"
    report_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path)
    validation_path = Path(args.validation_path)
    train_count = assert_messages_jsonl(train_path)
    validation_count = assert_messages_jsonl(validation_path)

    started_at = time.time()
    started_at_utc = datetime.now(timezone.utc).isoformat()

    raw_train_records = read_jsonl(train_path)
    raw_validation_records = read_jsonl(validation_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_stats = {
        "train": compute_token_stats(raw_train_records, tokenizer, args.max_seq_length),
        "validation": compute_token_stats(raw_validation_records, tokenizer, args.max_seq_length),
    }

    train_raw_dataset = load_dataset("json", data_files={"train": str(train_path)}, split="train")
    eval_raw_dataset = load_dataset("json", data_files={"validation": str(validation_path)}, split="validation")
    train_dataset = convert_to_prompt_completion(train_raw_dataset, tokenizer)
    eval_dataset = convert_to_prompt_completion(eval_raw_dataset, tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    callback = EpochAdapterSaverCallback(adapter_root=adapter_root, tokenizer=tokenizer, max_epoch=args.num_train_epochs)
    sft_args = make_sft_config(args, output_dir)
    trainer = make_trainer(model, tokenizer, train_dataset, eval_dataset, sft_args, callback)

    train_result = trainer.train()

    final_adapter_dir = adapter_root / "adapter_final"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_adapter_dir))
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    history = trainer.state.log_history
    save_loss_history(history, report_dir)

    finished_at = time.time()
    finished_at_utc = datetime.now(timezone.utc).isoformat()

    checkpoint_adapters: dict[str, Any] = {}
    for epoch in range(1, args.num_train_epochs + 1):
        record = callback.saved_epochs.get(epoch)
        if record is None:
            continue
        path = Path(record.adapter_dir)
        checkpoint_adapters[f"epoch_{epoch}"] = {
            **asdict(record),
            "size_bytes": dir_size_bytes(path),
            "sha256": sha256_tree(path),
        }

    checkpoint_adapters["final"] = {
        "adapter_dir": str(final_adapter_dir),
        "size_bytes": dir_size_bytes(final_adapter_dir),
        "sha256": sha256_tree(final_adapter_dir),
    }

    total_training_time = round(finished_at - started_at, 3)
    total_train_tokens = int(token_stats["train"]["total_tokens"]["avg"] * train_count * args.num_train_epochs) if token_stats["train"]["total_tokens"]["avg"] else None

    report = {
        "created_at_utc": finished_at_utc,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "dataset_count": {
            "train": train_count,
            "validation": validation_count,
        },
        "run_config": {
            "run_name": args.run_name,
            "model_id": args.model_name,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "max_seq_length": args.max_seq_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * max(torch.cuda.device_count(), 1),
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "num_train_epochs": args.num_train_epochs,
            "optimizer": args.optim,
            "lr_scheduler_type": args.lr_scheduler_type,
            "weight_decay": args.weight_decay,
            "bf16": bool(getattr(sft_args, "bf16", False)),
            "fp16": bool(getattr(sft_args, "fp16", False)),
            "load_in_4bit": True,
            "full_finetuning": False,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": DEFAULT_TARGET_MODULES,
            "completion_only_loss": True,
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "output_dir": str(output_dir),
        },
        "checkpoint_adapters": checkpoint_adapters,
        "runtime": {
            "total_training_time_sec": total_training_time,
            "samples_per_second_approx": (train_count * args.num_train_epochs / total_training_time) if total_training_time > 0 else None,
            "tokens_per_second_approx": (total_train_tokens / total_training_time) if total_training_time > 0 and total_train_tokens is not None else None,
            "epoch_save_times": {key: value for key, value in checkpoint_adapters.items() if key.startswith("epoch_")},
        },
        "gpu_info": get_gpu_info(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "unsloth": get_package_version("unsloth"),
            "trl": get_package_version("trl"),
            "transformers": get_package_version("transformers"),
            "peft": get_package_version("peft"),
            "bitsandbytes": get_package_version("bitsandbytes"),
            "datasets": get_package_version("datasets"),
        },
        "token_length_stats": token_stats,
        "skipped_or_truncated_count": {
            "train_over_max_seq_length_count": token_stats["train"]["over_max_seq_length_count"],
            "validation_over_max_seq_length_count": token_stats["validation"]["over_max_seq_length_count"],
            "truncated_count": 0,
            "skipped_count": 0,
        },
        "best_eval_checkpoint": find_best_eval_checkpoint(history, callback),
        "train_result_raw": str(train_result),
    }

    report_path = report_dir / "sft_run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] saved {report_path}")

    if args.make_archive:
        archive_path = Path(args.archive_path) if args.archive_path else output_dir.with_suffix(".tar.gz")
        make_archive(output_dir, archive_path)

    print("[done] SFT completed")
    print("[done] output_dir:", output_dir)
    print("[done] reports:", report_dir)
    print("[done] adapters:", adapter_root)


if __name__ == "__main__":
    main()
