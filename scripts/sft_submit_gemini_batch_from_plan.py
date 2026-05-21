# Builds mixed SFT request payloads from an automation plan and submits them to Gemini Batch API.
# This script does not call the validator and does not write accepted/rejected files.
# It reuses the existing compromised automation plan parsing and payload build logic.
# It submits inline batch requests; use this for small test runs such as 30 samples.

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from google import genai
except ModuleNotFoundError as error:
    raise SystemExit(
        "missing package 'google-genai'. Install it with: py -3.11 -m pip install google-genai"
    ) from error

try:
    from sft_auto_generate_compromised import (
        DEFAULT_AUTOMATION_DIR,
        DEFAULT_DATASET_ROOT,
        DEFAULT_MAX_PER_REQUEST,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TAXONOMY_PATH,
        advance_cycle_cursors_from_payload,
        build_batch_tasks,
        build_task_payload,
        path_text_from_payload,
        read_plan,
        run_plan_preflight_validation,
        task_request_items,
        task_with_cycle_offsets,
    )
    from sft_teacher_client import SYSTEM_PROMPT, get_api_key
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_auto_generate_compromised import (
        DEFAULT_AUTOMATION_DIR,
        DEFAULT_DATASET_ROOT,
        DEFAULT_MAX_PER_REQUEST,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TAXONOMY_PATH,
        advance_cycle_cursors_from_payload,
        build_batch_tasks,
        build_task_payload,
        path_text_from_payload,
        read_plan,
        run_plan_preflight_validation,
        task_request_items,
        task_with_cycle_offsets,
    )
    from sft_teacher_client import SYSTEM_PROMPT, get_api_key


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sdk_object_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "to_json_dict"):
        data = value.to_json_dict()
        return data if isinstance(data, dict) else None
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }
    return None


def select_tasks_by_sample_limit(tasks: list[Any], sample_limit: int | None) -> list[Any]:
    if sample_limit is None:
        return tasks
    if sample_limit <= 0:
        raise ValueError("--sample-limit must be greater than zero")

    selected: list[Any] = []
    total = 0

    for task in tasks:
        next_total = total + int(task.requested_count)
        if next_total > sample_limit:
            break
        selected.append(task)
        total = next_total
        if total == sample_limit:
            break

    if not selected:
        raise ValueError(
            f"No full batch task can fit into --sample-limit={sample_limit}. "
            "Use a larger sample limit or lower --max-per-request."
        )

    return selected


def build_inline_generate_content_request(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    user_text = json.dumps(payload, ensure_ascii=False, indent=2)

    config: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "temperature": args.temperature,
        "top_p": args.top_p,
        "candidate_count": 1,
        "max_output_tokens": args.max_tokens,
        "response_mime_type": "application/json",
    }

    if args.thinking_level:
        config["thinking_config"] = {"thinking_level": args.thinking_level}

    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "config": config,
    }


def build_batch_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = Path(args.dataset_root)
    taxonomy_path = Path(args.taxonomy)
    automation_dir = Path(args.automation_dir)
    plan_path = Path(args.plan)

    if args.preflight:
        result = run_plan_preflight_validation(
            dataset_root=dataset_root,
            taxonomy_path=taxonomy_path,
            automation_dir=automation_dir,
            plan_path=plan_path,
        )
        if result != 0:
            raise RuntimeError(f"plan preflight validation failed: code={result}")

    plan_items = read_plan(plan_path)
    tasks = build_batch_tasks(plan_items, args.max_per_request)
    tasks = tasks[args.start_index - 1 :]
    tasks = select_tasks_by_sample_limit(tasks, args.sample_limit)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "raw_generations" / "gemini_batch" / run_id
    payload_dir = output_dir / "payloads"

    inline_requests: list[dict[str, Any]] = []
    manifest_requests: list[dict[str, Any]] = []
    cycle_cursors: dict[tuple[str, str], int] = {}

    expected_sample_count = 0

    for ordinal, task in enumerate(tasks, start=1):
        task = task_with_cycle_offsets(task, cycle_cursors)
        payload = build_task_payload(
            task=task,
            dataset_root=dataset_root,
            taxonomy_path=taxonomy_path,
        )
        advance_cycle_cursors_from_payload(payload, cycle_cursors)

        request_key = f"request_{ordinal:04d}"
        payload_path = payload_dir / f"{request_key}.json"
        write_json(payload_path, payload)

        inline_requests.append(build_inline_generate_content_request(payload, args))
        expected_sample_count += int(payload.get("count_to_generate", task.requested_count))

        manifest_requests.append(
            {
                "request_key": request_key,
                "response_index_0_based": ordinal - 1,
                "task_index": task.task_index,
                "source_plan_line": task.source_plan_line,
                "split": task.split,
                "path": path_text_from_payload(payload),
                "request_prefix": task.request_prefix,
                "request": task.request,
                "request_items": task_request_items(task),
                "requested_count": task.requested_count,
                "payload_count_to_generate": payload.get("count_to_generate"),
                "payload_file": str(payload_path),
            }
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "inline_batch",
        "model": args.model,
        "display_name": args.display_name,
        "plan": str(plan_path),
        "dataset_root": str(dataset_root),
        "taxonomy": str(taxonomy_path),
        "max_per_request": args.max_per_request,
        "sample_limit": args.sample_limit,
        "start_index": args.start_index,
        "selected_task_count": len(tasks),
        "inline_request_count": len(inline_requests),
        "expected_sample_count": expected_sample_count,
        "output_dir": str(output_dir),
        "payload_dir": str(payload_dir),
        "requests": manifest_requests,
        "status": "built_not_submitted",
    }

    write_json(output_dir / "inline_requests.json", inline_requests)
    write_json(output_dir / "manifest.json", manifest)

    return inline_requests, manifest


def submit_batch(args: argparse.Namespace) -> dict[str, Any]:
    inline_requests, manifest = build_batch_inputs(args)

    if args.dry_run:
        print("dry_run: built inline batch inputs only")
        print(f"manifest: {Path(manifest['output_dir']) / 'manifest.json'}")
        print(f"inline_requests: {Path(manifest['output_dir']) / 'inline_requests.json'}")
        print(f"inline_request_count: {len(inline_requests)}")
        print(f"expected_sample_count: {manifest['expected_sample_count']}")
        return manifest

    client = genai.Client(api_key=get_api_key())

    batch_job = client.batches.create(
        model=args.model,
        src=inline_requests,
        config={"display_name": args.display_name},
    )

    batch_job_dict = sdk_object_to_dict(batch_job)

    manifest["status"] = "submitted"
    manifest["submitted_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["batch_job_name"] = getattr(batch_job, "name", None)
    manifest["batch_job"] = batch_job_dict

    manifest_path = Path(manifest["output_dir"]) / "manifest.json"
    write_json(manifest_path, manifest)

    print(f"created_batch_job: {manifest['batch_job_name']}")
    print(f"manifest: {manifest_path}")
    print(f"inline_request_count: {len(inline_requests)}")
    print(f"expected_sample_count: {manifest['expected_sample_count']}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit mixed SFT generation payloads from an automation plan to Gemini Batch API."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--automation-dir", default=str(DEFAULT_AUTOMATION_DIR))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--display-name", default="sft-batch")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-per-request", type=int, default=DEFAULT_MAX_PER_REQUEST)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-level", default="minimal")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.start_index < 1:
        parser.error("--start-index must be greater than zero")
    if args.max_per_request <= 0:
        parser.error("--max-per-request must be greater than zero")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")

    submit_batch(args)


if __name__ == "__main__":
    main()
