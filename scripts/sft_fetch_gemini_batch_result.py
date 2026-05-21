# Gemini Batch inline 결과를 회수하고 복구 가능한 원본을 모두 저장한다.
# count mismatch, base_command_text 보정 실패, manifest/response 개수 불일치로 샘플을 버리지 않는다.
# parse 가능한 sample은 batch_raw.jsonl에 저장하고, 모든 inline response와 raw text는 별도 파일로 보존한다.
# validation, accepted/rejected 저장은 수행하지 않는다.

from __future__ import annotations

import argparse
import json
import sys
import time
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
    from sft_teacher_client import (
        get_api_key,
        normalize_samples,
        parse_teacher_json,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_teacher_client import (
        get_api_key,
        normalize_samples,
        parse_teacher_json,
    )


COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def sdk_object_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "to_json_dict"):
        return value.to_json_dict()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [sdk_object_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: sdk_object_to_dict(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            key: sdk_object_to_dict(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }
    return str(value)


def get_value(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def get_state_name(batch_job: Any) -> str:
    state = get_value(batch_job, "state")
    if state is None:
        return ""
    name = get_value(state, "name")
    if isinstance(name, str):
        return name
    if isinstance(state, str):
        return state
    return str(state)


def wait_for_job(client: Any, job_name: str, poll_sec: float) -> Any:
    batch_job = client.batches.get(name=job_name)
    while get_state_name(batch_job) not in COMPLETED_STATES:
        print(f"Current state: {get_state_name(batch_job)}", flush=True)
        time.sleep(poll_sec)
        batch_job = client.batches.get(name=job_name)
    return batch_job


def extract_inline_responses_best_effort(batch_job: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    dest = get_value(batch_job, "dest")
    if dest is None:
        errors.append(
            {
                "scope": "batch_job",
                "status": "batch_job_dest_missing",
                "error_message": "batch_job.dest is missing",
            }
        )
        return [], errors

    responses = get_value(dest, "inlined_responses", "inlinedResponses")
    if responses is not None:
        return list(responses), errors

    file_name = get_value(dest, "file_name", "fileName")
    if file_name:
        errors.append(
            {
                "scope": "batch_job",
                "status": "file_result_not_inline",
                "error_message": f"batch result is in file: {file_name}",
                "file_name": file_name,
            }
        )
        return [], errors

    errors.append(
        {
            "scope": "batch_job",
            "status": "inline_responses_missing",
            "error_message": "batch result has no inline responses",
            "dest": sdk_object_to_dict(dest),
        }
    )
    return [], errors


def extract_response_text_best_effort(response: Any) -> tuple[str, list[str]]:
    warnings: list[str] = []

    text = get_value(response, "text")
    if isinstance(text, str) and text.strip():
        return text.strip(), warnings

    data = sdk_object_to_dict(response)

    if isinstance(data, dict):
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip(), warnings

        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list):
                chunks = [
                    part.get("text")
                    for part in parts
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                ]
                joined = "".join(chunks).strip()
                if joined:
                    return joined, warnings

    warnings.append("could_not_extract_text_from_response")
    return "", warnings


def assign_global_sample_ids(samples: list[dict[str, Any]], output_path: Path, start_index: int) -> None:
    batch_id = output_path.stem
    if batch_id.endswith("_raw"):
        batch_id = batch_id.removesuffix("_raw")

    for offset, sample in enumerate(samples):
        sample["id"] = f"3{batch_id}_{start_index + offset:03d}"


def load_payload_best_effort(request_record: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    payload_file = request_record.get("payload_file")
    if not isinstance(payload_file, str) or not payload_file:
        warnings.append("payload_file_missing")
        return None, warnings

    path = Path(payload_file)
    try:
        data = load_json(path)
    except Exception as error:
        warnings.append(f"payload_load_failed:{type(error).__name__}:{error}")
        return None, warnings

    if not isinstance(data, dict):
        warnings.append("payload_root_not_object")
        return None, warnings

    return data, warnings


def best_effort_force_base_command_text(
    samples: list[dict[str, Any]],
    payload: dict[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []

    if payload is None:
        warnings.append("base_command_text_not_forced_payload_missing")
        return warnings

    items = payload.get("mixed_generation_requests")
    if not isinstance(items, list) or not items:
        request = payload.get("request")
        if isinstance(request, dict):
            base_command_text = request.get("base_command_text")
            if isinstance(base_command_text, str) and base_command_text:
                for sample in samples:
                    command_spec = sample.setdefault("command_spec", {})
                    if isinstance(command_spec, dict):
                        command_spec["base_command_text"] = base_command_text
                    else:
                        warnings.append("sample_command_spec_not_object_single_payload")
                return warnings
        warnings.append("base_command_text_not_forced_no_mixed_items")
        return warnings

    sample_index = 0

    for item_index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            warnings.append(f"mixed_item_{item_index}_not_object")
            continue

        request = item.get("request")
        if not isinstance(request, dict):
            warnings.append(f"mixed_item_{item_index}_request_missing")
            continue

        base_command_text = request.get("base_command_text")
        if not isinstance(base_command_text, str) or not base_command_text:
            warnings.append(f"mixed_item_{item_index}_base_command_text_missing")
            continue

        count_to_generate = item.get("count_to_generate")
        if isinstance(count_to_generate, bool) or not isinstance(count_to_generate, int) or count_to_generate < 0:
            warnings.append(f"mixed_item_{item_index}_count_to_generate_invalid")
            continue

        remaining = len(samples) - sample_index
        if remaining <= 0:
            warnings.append(f"mixed_item_{item_index}_no_remaining_samples")
            break

        take_count = min(count_to_generate, remaining)
        item_samples = samples[sample_index : sample_index + take_count]
        sample_index += take_count

        if take_count != count_to_generate:
            warnings.append(
                f"mixed_item_{item_index}_partial_base_command_text_force:"
                f"expected={count_to_generate},available={take_count}"
            )

        for sample in item_samples:
            command_spec = sample.setdefault("command_spec", {})
            if isinstance(command_spec, dict):
                command_spec["base_command_text"] = base_command_text
            else:
                warnings.append("sample_command_spec_not_object")

    if sample_index < len(samples):
        warnings.append(f"trailing_samples_without_mixed_item_mapping:{len(samples) - sample_index}")

    return warnings


def make_placeholder_request_record(index: int, reason: str) -> dict[str, Any]:
    request_key = f"unmatched_response_{index + 1:04d}" if reason == "extra_response" else f"missing_response_{index + 1:04d}"
    return {
        "request_key": request_key,
        "response_index_0_based": index,
        "task_index": None,
        "source_plan_line": None,
        "split": "",
        "path": "",
        "request": "",
        "requested_count": None,
        "payload_count_to_generate": None,
        "payload_file": "",
        "placeholder_reason": reason,
    }


def parse_samples_best_effort(raw_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not raw_text.strip():
        warnings.append("raw_text_empty")
        return [], warnings

    try:
        parsed = parse_teacher_json(raw_text)
    except Exception as error:
        warnings.append(f"parse_teacher_json_failed:{type(error).__name__}:{error}")
        return [], warnings

    try:
        samples = normalize_samples(parsed)
    except Exception as error:
        warnings.append(f"normalize_samples_failed:{type(error).__name__}:{error}")
        return [], warnings

    return samples, warnings


def process_one_response(
    *,
    inline_response: Any | None,
    request_record: dict[str, Any],
    output_path: Path,
    next_sample_index: int,
    raw_response_dir: Path,
    inline_response_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], int]:
    request_key = str(request_record.get("request_key") or f"response_{request_record.get('response_index_0_based', 0):04d}")
    safe_request_key = request_key.replace("/", "_").replace("\\", "_").replace(":", "_")

    errors: list[dict[str, Any]] = []
    warnings: list[str] = []

    inline_response_file = inline_response_dir / f"{safe_request_key}_inline_response.json"
    raw_response_file = raw_response_dir / f"{safe_request_key}_raw_response.txt"

    if inline_response is None:
        errors.append(
            {
                "request_key": request_key,
                "status": "missing_inline_response",
                "error_message": "manifest request has no corresponding inline response",
            }
        )
        write_json(inline_response_file, None)
        write_text(raw_response_file, "")
        return [], {}, errors, next_sample_index

    inline_response_data = sdk_object_to_dict(inline_response)
    write_json(inline_response_file, inline_response_data)

    error = get_value(inline_response, "error")
    if error:
        errors.append(
            {
                "request_key": request_key,
                "status": "inline_response_error_preserved",
                "error": sdk_object_to_dict(error),
                "inline_response_file": str(inline_response_file),
            }
        )
        write_text(raw_response_file, "")
        return [], {}, errors, next_sample_index

    response = get_value(inline_response, "response")
    if response is None:
        errors.append(
            {
                "request_key": request_key,
                "status": "missing_response_preserved",
                "error_message": "inline response has no response object",
                "inline_response_file": str(inline_response_file),
            }
        )
        write_text(raw_response_file, "")
        return [], {}, errors, next_sample_index

    raw_text, text_warnings = extract_response_text_best_effort(response)
    warnings.extend(text_warnings)
    write_text(raw_response_file, raw_text)

    samples, parse_warnings = parse_samples_best_effort(raw_text)
    warnings.extend(parse_warnings)

    payload, payload_warnings = load_payload_best_effort(request_record)
    warnings.extend(payload_warnings)

    expected_count = None
    if payload is not None:
        expected_count = payload.get("count_to_generate")
    if expected_count is None:
        expected_count = request_record.get("payload_count_to_generate")

    actual_count = len(samples)
    count_mismatch = isinstance(expected_count, int) and actual_count != expected_count

    if count_mismatch:
        errors.append(
            {
                "request_key": request_key,
                "status": "teacher_sample_count_mismatch_preserved",
                "error_type": "ValueError",
                "error_message": f"teacher sample count mismatch: expected={expected_count}, actual={actual_count}",
                "expected_count": expected_count,
                "actual_count": actual_count,
                "inline_response_file": str(inline_response_file),
                "raw_response_file": str(raw_response_file),
            }
        )

    base_warnings = best_effort_force_base_command_text(samples, payload)
    warnings.extend(base_warnings)

    assign_global_sample_ids(samples, output_path, next_sample_index)

    if not samples:
        errors.append(
            {
                "request_key": request_key,
                "status": "no_parseable_samples_preserved",
                "error_message": "No parseable samples were produced from this response.",
                "inline_response_file": str(inline_response_file),
                "raw_response_file": str(raw_response_file),
                "warnings": warnings,
            }
        )

    trace_record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "request_key": request_key,
        "response_index_0_based": request_record.get("response_index_0_based"),
        "task_index": request_record.get("task_index"),
        "source_plan_line": request_record.get("source_plan_line"),
        "split": request_record.get("split"),
        "path": request_record.get("path"),
        "request": request_record.get("request"),
        "requested_count": request_record.get("requested_count"),
        "payload_count_to_generate": expected_count,
        "sample_count": actual_count,
        "count_mismatch": count_mismatch,
        "warnings": warnings,
        "inline_response_file": str(inline_response_file),
        "raw_response_file": str(raw_response_file),
    }

    return samples, trace_record, errors, next_sample_index + len(samples)


def fetch_batch(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)

    job_name = args.job or manifest.get("batch_job_name")
    if not isinstance(job_name, str) or not job_name:
        raise ValueError("Batch job name is missing. Pass --job or use a manifest with batch_job_name.")

    output_dir = Path(manifest.get("output_dir") or manifest_path.parent)
    output_path = Path(args.output) if args.output else output_dir / "batch_raw.jsonl"
    trace_path = Path(args.trace) if args.trace else output_dir / "batch_trace.jsonl"
    error_path = Path(args.errors) if args.errors else output_dir / "batch_errors.jsonl"
    status_path = output_dir / "batch_status.json"
    preserved_dir = output_dir / "preserved_batch_responses"
    raw_response_dir = preserved_dir / "raw_text"
    inline_response_dir = preserved_dir / "inline_response_json"

    client = genai.Client(api_key=get_api_key())

    if args.poll:
        batch_job = wait_for_job(client, job_name, args.poll_sec)
    else:
        batch_job = client.batches.get(name=job_name)

    state_name = get_state_name(batch_job)
    write_json(status_path, sdk_object_to_dict(batch_job))

    print(f"job: {job_name}")
    print(f"state: {state_name}")
    print(f"status_file: {status_path}")

    if state_name != "JOB_STATE_SUCCEEDED":
        error = get_value(batch_job, "error")
        if error:
            print(f"error: {sdk_object_to_dict(error)}")
        if not args.allow_not_succeeded:
            return 1

    inline_responses, extraction_errors = extract_inline_responses_best_effort(batch_job)
    requests = manifest.get("requests")
    if not isinstance(requests, list):
        requests = []
        extraction_errors.append(
            {
                "scope": "manifest",
                "status": "manifest_requests_not_list",
                "error_message": "manifest.requests is missing or not a list",
            }
        )

    response_count = len(inline_responses)
    request_count = len(requests)

    for path in (output_path, trace_path, error_path):
        if path.exists():
            if args.overwrite:
                path.unlink()
            else:
                raise FileExistsError(f"output file already exists: {path}. Use --overwrite.")

    if args.overwrite and preserved_dir.exists():
        for child in sorted(preserved_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass

    all_samples: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []

    for error_record in extraction_errors:
        error_record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        error_records.append(error_record)

    if response_count != request_count:
        error_records.append(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "scope": "batch",
                "status": "inline_response_manifest_count_mismatch_preserved",
                "error_message": (
                    f"inline response count mismatch: responses={response_count}, "
                    f"manifest_requests={request_count}"
                ),
                "response_count": response_count,
                "manifest_request_count": request_count,
            }
        )

    next_sample_index = 1
    max_count = max(response_count, request_count)

    for index in range(max_count):
        inline_response = inline_responses[index] if index < response_count else None

        if index < request_count and isinstance(requests[index], dict):
            request_record = dict(requests[index])
        elif index < request_count:
            request_record = make_placeholder_request_record(index, "invalid_manifest_request")
            error_records.append(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "scope": "manifest",
                    "status": "manifest_request_not_object",
                    "response_index_0_based": index,
                    "manifest_request": sdk_object_to_dict(requests[index]),
                }
            )
        else:
            request_record = make_placeholder_request_record(index, "extra_response")

        try:
            samples, trace_record, per_response_errors, next_sample_index = process_one_response(
                inline_response=inline_response,
                request_record=request_record,
                output_path=output_path,
                next_sample_index=next_sample_index,
                raw_response_dir=raw_response_dir,
                inline_response_dir=inline_response_dir,
            )
            if samples:
                all_samples.extend(samples)
            if trace_record:
                trace_records.append(trace_record)
            for error_record in per_response_errors:
                error_record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
                error_record.setdefault("response_index_0_based", index)
                error_record.setdefault("task_index", request_record.get("task_index"))
                error_record.setdefault("request", request_record.get("request"))
                error_record.setdefault("path", request_record.get("path"))
                error_records.append(error_record)
        except Exception as error:
            # 최후의 안전망. 여기까지 와도 이미 inline_response JSON은 저장되도록 시도한다.
            request_key = str(request_record.get("request_key") or f"response_{index:04d}")
            safe_request_key = request_key.replace("/", "_").replace("\\", "_").replace(":", "_")
            try:
                write_json(
                    inline_response_dir / f"{safe_request_key}_inline_response_unhandled.json",
                    sdk_object_to_dict(inline_response),
                )
            except Exception:
                pass

            error_records.append(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "request_key": request_key,
                    "response_index_0_based": index,
                    "task_index": request_record.get("task_index"),
                    "request": request_record.get("request"),
                    "path": request_record.get("path"),
                    "status": "unhandled_processing_error_preserved_best_effort",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    append_jsonl(output_path, all_samples)
    append_jsonl(trace_path, trace_records)
    append_jsonl(error_path, error_records)

    manifest["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["fetch_state"] = state_name
    manifest["raw_output_file"] = str(output_path)
    manifest["trace_file"] = str(trace_path)
    manifest["error_file"] = str(error_path)
    manifest["preserved_response_dir"] = str(preserved_dir)
    manifest["fetched_sample_count"] = len(all_samples)
    manifest["fetch_error_count"] = len(error_records)
    manifest["inline_response_count"] = response_count
    manifest["manifest_request_count"] = request_count
    manifest["preserve_all_mode"] = True
    write_json(manifest_path, manifest)

    print(f"raw_output: {output_path}")
    print(f"trace: {trace_path}")
    print(f"errors: {error_path}")
    print(f"preserved_response_dir: {preserved_dir}")
    print(f"fetched_sample_count: {len(all_samples)}")
    print(f"fetch_error_count: {len(error_records)}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Gemini Batch inline responses while preserving all raw responses."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--trace", default="")
    parser.add_argument("--errors", default="")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-not-succeeded", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.poll_sec <= 0:
        parser.error("--poll-sec must be greater than zero")

    raise SystemExit(fetch_batch(args))


if __name__ == "__main__":
    main()
