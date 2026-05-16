# 계획 문서의 split별 generation request를 단일 path 요청 단위로 자동 실행한다.
# 각 요청은 최대 N개 sample만 생성하고, 생성 직후 validate를 실행한다.
# Gemini 요청 실패 시 10회까지 같은 요청을 재시도하고, 모두 실패하면 다음 요청으로 넘어간다.
# 자동 생성 전에 automation plan validator를 실행해 invalid path가 있으면 생성하지 않는다.
# Ctrl+C가 들어오면 다음 요청을 시작하지 않고 즉시 종료한다.

from __future__ import annotations

import random
import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from sft_file_sequence import next_numbered_index, numbered_path
    from sft_generation_request import (
        DEFAULT_TARGET_SPLIT,
        VALID_SPLITS,
        build_generation_payload,
        write_json,
    )
    from sft_validate_automation_plans import validate_plans as validate_automation_plans
    from sft_validator import validate_file
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_file_sequence import next_numbered_index, numbered_path
    from sft_generation_request import (
        DEFAULT_TARGET_SPLIT,
        VALID_SPLITS,
        build_generation_payload,
        write_json,
    )
    from sft_validate_automation_plans import validate_plans as validate_automation_plans
    from sft_validator import validate_file


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY_PATH = DEFAULT_DATASET_ROOT / "config" / "taxonomy_sot.json"
DEFAULT_AUTOMATION_DIR = DEFAULT_DATASET_ROOT / "generation_automation"
DEFAULT_PLAN_PATH = DEFAULT_AUTOMATION_DIR / "auto_generation_plan_0001.txt"
DEFAULT_TEACHER_MODEL = "gemma-4-31b-it"
DEFAULT_MAX_PER_REQUEST = 10
DEFAULT_MAX_TOKENS = 60000
DEFAULT_API_REQUEST_ATTEMPTS = 10
DEFAULT_API_RETRY_DELAY_MIN_SEC = 10.0
DEFAULT_API_RETRY_DELAY_MAX_SEC = 12.0


@dataclass(frozen=True)
class PlanItem:
    line_number: int
    split: str
    request_prefix: str
    count: int
    raw_request: str


@dataclass(frozen=True)
class BatchTask:
    task_index: int
    source_plan_line: int
    split: str
    request_prefix: str
    requested_count: int
    request: str


def strip_inline_comment(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    return re.split(r"\s+#", stripped, maxsplit=1)[0].strip()


def split_request_count(raw_request: str) -> tuple[str, int]:
    request = raw_request.strip()
    if not request:
        raise ValueError("empty request")

    if "." not in request:
        raise ValueError(f"request must end with .<count>: {raw_request}")

    prefix, count_text = request.rsplit(".", 1)
    if not prefix or not count_text.isdigit():
        raise ValueError(f"request must end with .<count>: {raw_request}")

    count = int(count_text)
    if count <= 0:
        raise ValueError(f"request count must be greater than zero: {raw_request}")

    return prefix, count


def parse_plan_line(line: str, line_number: int) -> PlanItem | None:
    cleaned = strip_inline_comment(line)
    if not cleaned:
        return None

    parts = cleaned.split()
    if len(parts) != 2:
        raise ValueError(
            f"line {line_number}: expected '<split> <generation_request>', got: {line.rstrip()}"
        )

    split, raw_request = parts
    if split not in VALID_SPLITS:
        raise ValueError(
            f"line {line_number}: split must be one of {VALID_SPLITS}, got: {split}"
        )

    request_prefix, count = split_request_count(raw_request)
    return PlanItem(
        line_number=line_number,
        split=split,
        request_prefix=request_prefix,
        count=count,
        raw_request=raw_request,
    )


def read_plan(path: Path) -> list[PlanItem]:
    if not path.exists():
        raise FileNotFoundError(f"plan file not found: {path}")

    items: list[PlanItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        item = parse_plan_line(line, line_number)
        if item is not None:
            items.append(item)

    if not items:
        raise ValueError(f"plan file has no generation items: {path}")

    return items


def build_batch_tasks(items: list[PlanItem], max_per_request: int) -> list[BatchTask]:
    if max_per_request <= 0:
        raise ValueError("max_per_request must be greater than zero")

    tasks: list[BatchTask] = []
    task_index = 1

    for item in items:
        remaining = item.count
        while remaining > 0:
            requested_count = min(max_per_request, remaining)
            request = f"{item.request_prefix}.{requested_count}"
            tasks.append(
                BatchTask(
                    task_index=task_index,
                    source_plan_line=item.line_number,
                    split=item.split,
                    request_prefix=item.request_prefix,
                    requested_count=requested_count,
                    request=request,
                )
            )
            task_index += 1
            remaining -= requested_count

    return tasks


def infer_plan_label(plan_path: Path) -> str:
    match = re.search(r"auto_generation_plan_(\d{4})", plan_path.stem)
    if match:
        return f"plan_{match.group(1)}"
    return plan_path.stem


def make_run_paths(automation_dir: Path, plan_path: Path) -> tuple[Path, Path]:
    automation_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_label = infer_plan_label(plan_path)
    result_path = automation_dir / f"auto_run_{timestamp}_{plan_label}.md"
    error_path = automation_dir / f"auto_run_{timestamp}_{plan_label}_errors.jsonl"
    return result_path, error_path


def make_batch_paths(raw_dir: Path) -> tuple[Path, Path, Path]:
    request_index = next_numbered_index(raw_dir, "request", ".json")
    batch_index = next_numbered_index(raw_dir, "batch", "_raw.jsonl")

    return (
        numbered_path(raw_dir, "request", request_index, ".json"),
        numbered_path(raw_dir, "batch", batch_index, "_raw.jsonl"),
        numbered_path(raw_dir, "batch", batch_index, "_trace.jsonl"),
    )


def relpath(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def list_files(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {path for path in directory.iterdir() if path.is_file()}


def get_new_files(before: set[Path], after: set[Path], dataset_root: Path) -> list[str]:
    return [relpath(path, dataset_root) for path in sorted(after - before)]


def compact_validation_result(
    validation_result: dict[str, Any],
    accepted_files: list[str],
    rejected_files: list[str],
) -> dict[str, Any]:
    compact: dict[str, Any] = {}

    for key in (
        "input",
        "total",
        "accepted_count",
        "rejected_count",
        "valid_count",
        "invalid_count",
        "skipped",
    ):
        if key in validation_result:
            compact[key] = validation_result[key]

    compact["accepted_files"] = accepted_files
    compact["rejected_files"] = rejected_files

    return compact


def count_from_validation(result: dict[str, Any], count_key: str, list_key: str) -> int:
    value = result.get(count_key)
    if isinstance(value, int):
        return value

    value = result.get(list_key)
    if isinstance(value, list):
        return len(value)

    return 0


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def path_text_from_payload(payload: dict[str, Any]) -> str:
    request_info = payload.get("request")
    if not isinstance(request_info, dict):
        return ""

    stable_path = request_info.get("stable_path")
    skill_stable_path = request_info.get("skill_stable_path")

    if isinstance(stable_path, str) and stable_path:
        if isinstance(skill_stable_path, str) and skill_stable_path:
            return f"{stable_path}/{skill_stable_path}"
        return stable_path

    return ""


def run_plan_preflight_validation(
    *,
    dataset_root: Path,
    taxonomy_path: Path,
    automation_dir: Path,
    plan_path: Path,
) -> int:
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        result_code = validate_automation_plans(
            dataset_root=dataset_root,
            taxonomy_path=taxonomy_path,
            automation_dir=automation_dir,
            plan=str(plan_path),
        )

    output = buffer.getvalue()
    if result_code != 0:
        print(output, end="")
        return result_code

    return 0


def validate_tasks_for_dry_run(
    *,
    tasks: list[BatchTask],
    dataset_root: Path,
    taxonomy_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for task in tasks:
        started = time.monotonic()
        started_at = datetime.now().isoformat(timespec="seconds")
        payload = build_generation_payload(
            raw_request=task.request,
            dataset_root=dataset_root,
            taxonomy_path=taxonomy_path,
            target_split=task.split,
        )

        records.append(
            {
                "task_index": task.task_index,
                "source_plan_line": task.source_plan_line,
                "split": task.split,
                "path": path_text_from_payload(payload),
                "requested_count": task.requested_count,
                "request_payload_file": "",
                "raw_generation_file": "",
                "trace_file": "",
                "validate_result": {
                    "dry_run_request": task.request,
                    "split": task.split,
                },
                "accepted_count": 0,
                "rejected_count": 0,
                "status": "dry_run_valid",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_sec": round(time.monotonic() - started, 2),
            }
        )

    return records


def summarize_path(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    summary: dict[tuple[str, str], dict[str, Any]] = {}

    for record in records:
        split = str(record.get("split", ""))
        path = str(record.get("path", ""))
        if not path:
            continue

        key = (split, path)
        row = summary.setdefault(
            key,
            {
                "requested": 0,
                "accepted": 0,
                "rejected": 0,
                "api_failed": 0,
                "batches": 0,
                "elapsed_sec": 0.0,
            },
        )

        requested = int(record.get("requested_count", 0) or 0)
        accepted = int(record.get("accepted_count", 0) or 0)
        rejected = int(record.get("rejected_count", 0) or 0)
        elapsed = float(record.get("elapsed_sec", 0.0) or 0.0)

        row["requested"] += requested
        row["accepted"] += accepted
        row["rejected"] += rejected
        row["batches"] += 1
        row["elapsed_sec"] += elapsed

        if str(record.get("status", "")) in {"api_error", "request_build_error"}:
            row["api_failed"] += requested

    return summary


def render_run_document(
    *,
    result_path: Path,
    error_path: Path,
    plan_path: Path,
    plan_text: str,
    expanded_tasks: list[BatchTask],
    records: list[dict[str, Any]],
    status: str,
    started_at: str,
    finished_at: str | None,
    interrupted_at: str | None,
) -> str:
    completed_indexes = [
        int(record["task_index"])
        for record in records
        if isinstance(record.get("task_index"), int)
        and record.get("status") not in {"dry_run_valid"}
    ]
    last_completed = max(completed_indexes) if completed_indexes else 0
    next_task_index = last_completed + 1 if last_completed < len(expanded_tasks) else None

    total_elapsed_sec = sum(float(record.get("elapsed_sec", 0.0) or 0.0) for record in records)
    total_requested = sum(
        int(record.get("requested_count", 0) or 0)
        for record in records
        if record.get("status") != "dry_run_valid"
    )
    total_accepted = sum(int(record.get("accepted_count", 0) or 0) for record in records)
    total_rejected = sum(int(record.get("rejected_count", 0) or 0) for record in records)
    accepted_per_min = (total_accepted / total_elapsed_sec * 60.0) if total_elapsed_sec > 0 else 0.0
    samples_per_min = (total_requested / total_elapsed_sec * 60.0) if total_elapsed_sec > 0 else 0.0

    lines: list[str] = []
    lines.append("# 생성 자동화 결과")
    lines.append("")
    lines.append("## Plan snapshot")
    lines.append("")
    lines.append("```text")
    lines.append(plan_text.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Expanded tasks")
    lines.append("")
    lines.append("| task_index | source_plan_line | split | request | requested_count |")
    lines.append("|---:|---:|---|---|---:|")
    for task in expanded_tasks:
        lines.append(
            f"| {task.task_index} | {task.source_plan_line} | `{task.split}` | `{task.request}` | {task.requested_count} |"
        )

    lines.append("")
    lines.append("## Process status")
    lines.append("")
    lines.append(f"- status: `{status}`")
    lines.append(f"- result_document: `{result_path.name}`")
    lines.append(f"- error_document: `{error_path.name}`")
    lines.append(f"- plan_file: `{plan_path.name}`")
    lines.append(f"- started_at: `{started_at}`")
    lines.append(f"- finished_at: `{finished_at or ''}`")
    lines.append(f"- interrupted_at: `{interrupted_at or ''}`")
    lines.append(f"- last_completed_task_index: `{last_completed}`")
    lines.append(f"- next_task_index: `{'' if next_task_index is None else next_task_index}`")
    lines.append("")
    lines.append("## Request results")
    lines.append("")
    lines.append(
        "| task_index | source_plan_line | path | requested_count | request_payload_file | raw_generation_file | trace_file | validate_result | accepted_count | rejected_count | status | time |"
    )
    lines.append(
        "|---:|---:|---|---:|---|---|---|---|---:|---:|---|---|"
    )

    for record in records:
        validate_result = compact_json(record.get("validate_result", {}))
        time_text = (
            f"started_at={record.get('started_at', '')}; "
            f"finished_at={record.get('finished_at', '')}; "
            f"elapsed_sec={record.get('elapsed_sec', '')}"
        )
        lines.append(
            "| "
            f"{record.get('task_index', '')} | "
            f"{record.get('source_plan_line', '')} | "
            f"`{record.get('path', '')}` | "
            f"{record.get('requested_count', '')} | "
            f"`{record.get('request_payload_file', '')}` | "
            f"`{record.get('raw_generation_file', '')}` | "
            f"`{record.get('trace_file', '')}` | "
            f"`{validate_result}` | "
            f"{record.get('accepted_count', 0)} | "
            f"{record.get('rejected_count', 0)} | "
            f"`{record.get('status', '')}` | "
            f"`{time_text}` |"
        )

    lines.append("")
    lines.append("## Path summary")
    lines.append("")
    lines.append("| split | path | requested | accepted | rejected | api_failed | batches | elapsed_sec | samples_per_min | accepted_per_min |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for (split, path), row in sorted(summarize_path(records).items()):
        elapsed = float(row["elapsed_sec"])
        requested = int(row["requested"])
        accepted = int(row["accepted"])
        row_samples_per_min = (requested / elapsed * 60.0) if elapsed > 0 else 0.0
        row_accepted_per_min = (accepted / elapsed * 60.0) if elapsed > 0 else 0.0
        lines.append(
            f"| `{split}` | `{path}` | {requested} | {accepted} | {row['rejected']} | "
            f"{row['api_failed']} | {row['batches']} | {elapsed:.2f} | "
            f"{row_samples_per_min:.2f} | {row_accepted_per_min:.2f} |"
        )

    lines.append("")
    lines.append("## Total summary")
    lines.append("")
    lines.append(f"- total_requested: `{total_requested}`")
    lines.append(f"- total_accepted: `{total_accepted}`")
    lines.append(f"- total_rejected: `{total_rejected}`")
    lines.append(f"- total_elapsed_sec: `{total_elapsed_sec:.2f}`")
    lines.append(f"- samples_per_min: `{samples_per_min:.2f}`")
    lines.append(f"- accepted_per_min: `{accepted_per_min:.2f}`")
    lines.append("")

    return "\n".join(lines)


def write_run_document(
    *,
    result_path: Path,
    error_path: Path,
    plan_path: Path,
    plan_text: str,
    expanded_tasks: list[BatchTask],
    records: list[dict[str, Any]],
    status: str,
    started_at: str,
    finished_at: str | None,
    interrupted_at: str | None,
) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        render_run_document(
            result_path=result_path,
            error_path=error_path,
            plan_path=plan_path,
            plan_text=plan_text,
            expanded_tasks=expanded_tasks,
            records=records,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            interrupted_at=interrupted_at,
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_report(dataset_root: Path, taxonomy_path: Path) -> None:
    script_path = Path(__file__).resolve().parent / "sft_coverage_report.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dataset-root",
            str(dataset_root),
            "--taxonomy",
            str(taxonomy_path),
        ],
        check=True,
    )


def print_plan_summary(tasks: list[BatchTask], plan_path: Path, max_per_request: int) -> None:
    total_requested = sum(task.requested_count for task in tasks)
    unique_requests = len({(task.split, task.request_prefix) for task in tasks})

    print("auto generation plan loaded")
    print(f"plan_file: {plan_path}")
    print(f"unique_request_prefixes: {unique_requests}")
    print(f"batch_tasks: {len(tasks)}")
    print(f"total_requested_samples: {total_requested}")
    print(f"max_per_request: {max_per_request}")
    print("batches:")
    for task in tasks:
        print(
            f"  {task.task_index:04d}. line={task.source_plan_line} "
            f"split={task.split} request={task.request}"
        )


def run_teacher(
    *,
    input_path: Path,
    output_path: Path,
    trace_path: Path,
    model_name: str,
    max_tokens: int,
    stream_output: bool,
) -> dict[str, Any]:
    try:
        from sft_teacher_client import run_teacher_generation
    except ModuleNotFoundError as error:
        if error.name in {"google", "google.genai"}:
            raise RuntimeError(
                "missing package 'google-genai'. Install it with: python -m pip install google-genai"
            ) from error
        raise

    return run_teacher_generation(
        input_path=input_path,
        output_path=output_path,
        trace_path=trace_path,
        model_name=model_name,
        max_tokens=max_tokens,
        print_json=False,
        stream_output=stream_output,
    )



def run_teacher_with_retry(
    *,
    input_path: Path,
    output_path: Path,
    trace_path: Path,
    model_name: str,
    max_tokens: int,
    stream_output: bool,
    attempts: int,
    delay_min_sec: float,
    delay_max_sec: float,
) -> dict[str, Any]:
    if attempts <= 0:
        raise ValueError("attempts must be greater than zero")
    if delay_min_sec < 0 or delay_max_sec < delay_min_sec:
        raise ValueError("retry delay range is invalid")

    last_error: Exception | None = None

    for attempt_index in range(1, attempts + 1):
        try:
            return run_teacher(
                input_path=input_path,
                output_path=output_path,
                trace_path=trace_path,
                model_name=model_name,
                max_tokens=max_tokens,
                stream_output=stream_output,
            )
        except Exception as error:
            last_error = error
            if attempt_index >= attempts:
                break

            delay_sec = random.uniform(delay_min_sec, delay_max_sec)
            print(
                "api_request_failed: "
                f"attempt={attempt_index}/{attempts}, "
                f"retry_after_sec={delay_sec:.2f}, "
                f"error={error}"
            )
            time.sleep(delay_sec)

    raise RuntimeError(
        f"api request failed after {attempts} attempts: {last_error}"
    )


def run_auto_generate(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    taxonomy_path = Path(args.taxonomy)
    automation_dir = Path(args.automation_dir)
    plan_path = Path(args.plan)
    raw_dir = dataset_root / "raw_generations"
    accepted_dir = dataset_root / "accepted"
    rejected_dir = dataset_root / "rejected"

    preflight_result = run_plan_preflight_validation(
        dataset_root=dataset_root,
        taxonomy_path=taxonomy_path,
        automation_dir=automation_dir,
        plan_path=plan_path,
    )
    if preflight_result != 0:
        return preflight_result

    items = read_plan(plan_path)
    tasks = build_batch_tasks(items, args.max_per_request)
    selected_tasks = tasks[args.start_index - 1 :]
    plan_text = plan_path.read_text(encoding="utf-8")

    if not selected_tasks:
        print("no tasks selected")
        return 0

    result_path, error_path = make_run_paths(automation_dir, plan_path)
    started_at = datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []

    print_plan_summary(
        tasks=selected_tasks,
        plan_path=plan_path,
        max_per_request=args.max_per_request,
    )
    print(f"result_document: {result_path}")
    print(f"error_document: {error_path}")

    if args.dry_run:
        try:
            records = validate_tasks_for_dry_run(
                tasks=selected_tasks,
                dataset_root=dataset_root,
                taxonomy_path=taxonomy_path,
            )
            status = "dry_run_valid"
            print("dry_run validation ok")
        except Exception as error:
            status = "dry_run_failed"
            append_jsonl(
                error_path,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            print(f"dry_run failed: {error}")

        write_run_document(
            result_path=result_path,
            error_path=error_path,
            plan_path=plan_path,
            plan_text=plan_text,
            expanded_tasks=selected_tasks,
            records=records,
            status=status,
            started_at=started_at,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            interrupted_at=None,
        )
        return 0 if status == "dry_run_valid" else 1

    write_run_document(
        result_path=result_path,
        error_path=error_path,
        plan_path=plan_path,
        plan_text=plan_text,
        expanded_tasks=selected_tasks,
        records=records,
        status="running",
        started_at=started_at,
        finished_at=None,
        interrupted_at=None,
    )

    if not args.yes:
        answer = input("type y to start auto generation, n to cancel: ").strip().lower()
        if answer != "y":
            write_run_document(
                result_path=result_path,
                error_path=error_path,
                plan_path=plan_path,
                plan_text=plan_text,
                expanded_tasks=selected_tasks,
                records=records,
                status="cancelled",
                started_at=started_at,
                finished_at=datetime.now().isoformat(timespec="seconds"),
                interrupted_at=None,
            )
            print("cancelled")
            return 0

    raw_dir.mkdir(parents=True, exist_ok=True)

    status = "completed"
    interrupted_at: str | None = None

    try:
        for task in selected_tasks:
            task_start_monotonic = time.monotonic()
            task_started_at = datetime.now().isoformat(timespec="seconds")
            print("")
            print(
                f"[{task.task_index}/{selected_tasks[-1].task_index}] "
                f"start split={task.split} request={task.request} line={task.source_plan_line}"
            )

            request_path, raw_output_path, trace_output_path = make_batch_paths(raw_dir)

            try:
                payload = build_generation_payload(
                    raw_request=task.request,
                    dataset_root=dataset_root,
                    taxonomy_path=taxonomy_path,
                    target_split=task.split,
                )
                stable_path = path_text_from_payload(payload)
                write_json(request_path, payload)
            except Exception as error:
                task_finished_at = datetime.now().isoformat(timespec="seconds")
                elapsed_sec = round(time.monotonic() - task_start_monotonic, 2)
                error_record = {
                    "time": task_finished_at,
                    "task_index": task.task_index,
                    "source_plan_line": task.source_plan_line,
                    "split": task.split,
                    "request": task.request,
                    "requested_count": task.requested_count,
                    "request_payload_file": relpath(request_path, dataset_root),
                    "raw_generation_file": relpath(raw_output_path, dataset_root),
                    "trace_file": relpath(trace_output_path, dataset_root),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                append_jsonl(error_path, error_record)

                records.append(
                    {
                        "task_index": task.task_index,
                        "source_plan_line": task.source_plan_line,
                        "split": task.split,
                        "path": task.request_prefix,
                        "requested_count": task.requested_count,
                        "request_payload_file": relpath(request_path, dataset_root),
                        "raw_generation_file": relpath(raw_output_path, dataset_root),
                        "trace_file": relpath(trace_output_path, dataset_root),
                        "validate_result": {},
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "status": "request_build_error",
                        "started_at": task_started_at,
                        "finished_at": task_finished_at,
                        "elapsed_sec": elapsed_sec,
                    }
                )

                print(f"request_build_error: {error}")

            teacher_result: dict[str, Any] | None = None

            try:
                teacher_result = run_teacher_with_retry(
                    input_path=request_path,
                    output_path=raw_output_path,
                    trace_path=trace_output_path,
                    model_name=args.model,
                    max_tokens=args.max_tokens,
                    stream_output=args.stream_output,
                    attempts=args.api_request_attempts,
                    delay_min_sec=args.api_retry_delay_min_sec,
                    delay_max_sec=args.api_retry_delay_max_sec,
                )
            except Exception as error:
                task_finished_at = datetime.now().isoformat(timespec="seconds")
                elapsed_sec = round(time.monotonic() - task_start_monotonic, 2)
                error_record = {
                    "time": task_finished_at,
                    "task_index": task.task_index,
                    "source_plan_line": task.source_plan_line,
                    "split": task.split,
                    "path": stable_path,
                    "requested_count": task.requested_count,
                    "request": task.request,
                    "request_payload_file": relpath(request_path, dataset_root),
                    "raw_generation_file": relpath(raw_output_path, dataset_root),
                    "trace_file": relpath(trace_output_path, dataset_root),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "attempts": args.api_request_attempts,
                }
                append_jsonl(error_path, error_record)

                records.append(
                    {
                        "task_index": task.task_index,
                        "source_plan_line": task.source_plan_line,
                        "split": task.split,
                        "path": stable_path,
                        "requested_count": task.requested_count,
                        "request_payload_file": relpath(request_path, dataset_root),
                        "raw_generation_file": relpath(raw_output_path, dataset_root),
                        "trace_file": relpath(trace_output_path, dataset_root),
                        "validate_result": {
                            "api_error": str(error),
                            "attempts": args.api_request_attempts,
                        },
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "status": "api_error",
                        "started_at": task_started_at,
                        "finished_at": task_finished_at,
                        "elapsed_sec": elapsed_sec,
                    }
                )

                print(f"api_error_after_retries: {error}")

            if teacher_result is not None:
                try:
                    accepted_before = list_files(accepted_dir)
                    rejected_before = list_files(rejected_dir)

                    validation_result = validate_file(
                        input_path=raw_output_path,
                        dataset_root=dataset_root,
                        taxonomy_path=taxonomy_path,
                        write_outputs=not args.validate_dry_run,
                    )

                    accepted_after = list_files(accepted_dir)
                    rejected_after = list_files(rejected_dir)
                    accepted_files = get_new_files(accepted_before, accepted_after, dataset_root)
                    rejected_files = get_new_files(rejected_before, rejected_after, dataset_root)

                    accepted_count = count_from_validation(
                        validation_result,
                        "accepted_count",
                        "accepted",
                    )
                    rejected_count = count_from_validation(
                        validation_result,
                        "rejected_count",
                        "rejected",
                    )

                    if rejected_count > 0:
                        task_status = "rejected"
                    elif accepted_count > 0:
                        task_status = "accepted"
                    else:
                        task_status = "validated"

                    record = {
                        "task_index": task.task_index,
                        "source_plan_line": task.source_plan_line,
                        "split": task.split,
                        "path": stable_path,
                        "requested_count": task.requested_count,
                        "request_payload_file": relpath(request_path, dataset_root),
                        "raw_generation_file": relpath(raw_output_path, dataset_root),
                        "trace_file": relpath(trace_output_path, dataset_root),
                        "validate_result": compact_validation_result(
                            validation_result,
                            accepted_files,
                            rejected_files,
                        ),
                        "accepted_count": accepted_count,
                        "rejected_count": rejected_count,
                        "status": task_status,
                        "started_at": task_started_at,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_sec": round(time.monotonic() - task_start_monotonic, 2),
                    }
                    records.append(record)

                    print("validation:")
                    print(json.dumps(record["validate_result"], ensure_ascii=False, indent=2))

                except Exception as error:
                    task_finished_at = datetime.now().isoformat(timespec="seconds")
                    elapsed_sec = round(time.monotonic() - task_start_monotonic, 2)
                    error_record = {
                        "time": task_finished_at,
                        "task_index": task.task_index,
                        "source_plan_line": task.source_plan_line,
                        "split": task.split,
                        "path": stable_path,
                        "requested_count": task.requested_count,
                        "request": task.request,
                        "request_payload_file": relpath(request_path, dataset_root),
                        "raw_generation_file": relpath(raw_output_path, dataset_root),
                        "trace_file": relpath(trace_output_path, dataset_root),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                    append_jsonl(error_path, error_record)

                    records.append(
                        {
                            "task_index": task.task_index,
                            "source_plan_line": task.source_plan_line,
                            "split": task.split,
                            "path": stable_path,
                            "requested_count": task.requested_count,
                            "request_payload_file": relpath(request_path, dataset_root),
                            "raw_generation_file": relpath(raw_output_path, dataset_root),
                            "trace_file": relpath(trace_output_path, dataset_root),
                            "validate_result": {"validation_error": str(error)},
                            "accepted_count": 0,
                            "rejected_count": 0,
                            "status": "validation_error",
                            "started_at": task_started_at,
                            "finished_at": task_finished_at,
                            "elapsed_sec": elapsed_sec,
                        }
                    )

                    print(f"validation_error: {error}")

            write_run_document(
                result_path=result_path,
                error_path=error_path,
                plan_path=plan_path,
                plan_text=plan_text,
                expanded_tasks=selected_tasks,
                records=records,
                status=status if status != "completed" else "running",
                started_at=started_at,
                finished_at=None,
                interrupted_at=None,
            )
            
            if task.task_index != selected_tasks[-1].task_index:
                time.sleep(random.uniform(1.0, 2.0))

    except KeyboardInterrupt:
        status = "interrupted_by_keyboard"
        interrupted_at = datetime.now().isoformat(timespec="seconds")
        write_run_document(
            result_path=result_path,
            error_path=error_path,
            plan_path=plan_path,
            plan_text=plan_text,
            expanded_tasks=selected_tasks,
            records=records,
            status=status,
            started_at=started_at,
            finished_at=interrupted_at,
            interrupted_at=interrupted_at,
        )
        print("")
        print("interrupted_by_keyboard")
        print(f"result_document: {result_path}")
        return 130

    finished_at = datetime.now().isoformat(timespec="seconds")
    error_statuses = {"api_error", "request_build_error", "validation_error"}
    final_status = (
        "completed_with_errors"
        if any(record.get("status") in error_statuses for record in records)
        else status
    )

    write_run_document(
        result_path=result_path,
        error_path=error_path,
        plan_path=plan_path,
        plan_text=plan_text,
        expanded_tasks=selected_tasks,
        records=records,
        status=final_status,
        started_at=started_at,
        finished_at=finished_at,
        interrupted_at=interrupted_at,
    )

    if args.refresh_report and not args.validate_dry_run:
        run_report(dataset_root, taxonomy_path)

    print("")
    print("auto generation finished")
    print(f"status: {final_status}")
    print(f"result_document: {result_path}")
    print(f"error_document: {error_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sequential SFT teacher generation batches from a split/request plan."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--automation-dir", default=str(DEFAULT_AUTOMATION_DIR))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-per-request", type=int, default=DEFAULT_MAX_PER_REQUEST)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-dry-run", action="store_true")
    parser.add_argument("--refresh-report", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--stop-on-rejected", action="store_true")
    parser.add_argument("--api-request-attempts", type=int, default=DEFAULT_API_REQUEST_ATTEMPTS)
    parser.add_argument("--api-retry-delay-min-sec", type=float, default=DEFAULT_API_RETRY_DELAY_MIN_SEC)
    parser.add_argument("--api-retry-delay-max-sec", type=float, default=DEFAULT_API_RETRY_DELAY_MAX_SEC)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.start_index < 1:
        parser.error("--start-index must be greater than zero")
    if args.api_request_attempts < 1:
        parser.error("--api-request-attempts must be greater than zero")
    if args.api_retry_delay_min_sec < 0:
        parser.error("--api-retry-delay-min-sec must be at least zero")
    if args.api_retry_delay_max_sec < args.api_retry_delay_min_sec:
        parser.error("--api-retry-delay-max-sec must be greater than or equal to --api-retry-delay-min-sec")

    raise SystemExit(run_auto_generate(args))


if __name__ == "__main__":
    main()
