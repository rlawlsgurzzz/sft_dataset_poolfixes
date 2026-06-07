# Gemini student batch 결과를 회수하고 actual responses JSONL을 만든다.
# file-batch 결과의 key를 manifest row_index와 매칭해 입력 JSONL 순서로 재정렬한다.
# evaluator가 읽는 raw_content/request_success/error 형태를 기존 responses JSONL과 맞춘다.
# raw 결과와 trace/error 파일은 manifest 폴더 아래에 보존한다.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from google import genai
except ModuleNotFoundError as error:
    raise SystemExit(
        "missing package 'google-genai'.\n"
        "Install it with: py -3.11 -m pip install google-genai"
    ) from error

try:
    from sft_teacher_client import get_api_key
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        from sft_teacher_client import get_api_key
    except ImportError:
        def get_api_key() -> str:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수가 없습니다.\n"
                    "PowerShell 예시:\n"
                    '$env:GEMINI_API_KEY="네_API_키"'
                )
            return api_key


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


def write_jsonl(path: Path, records: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_for_job(client: Any, job_name: str, poll_sec: float) -> Any:
    batch_job = client.batches.get(name=job_name)
    while get_state_name(batch_job) not in COMPLETED_STATES:
        print(f"Current state: {get_state_name(batch_job)}", flush=True)
        time.sleep(poll_sec)
        batch_job = client.batches.get(name=job_name)
    return batch_job


def output_file_name_from_job(batch_job: Any) -> str | None:
    dest = get_value(batch_job, "dest")
    if dest is None:
        return None
    file_name = get_value(dest, "file_name", "fileName")
    return file_name if isinstance(file_name, str) and file_name else None


def extract_text_from_response(response: Any) -> tuple[str | None, list[str]]:
    warnings: list[str] = []

    if response is None:
        return None, ["response_missing"]

    text = get_value(response, "text")
    if isinstance(text, str) and text.strip():
        return text.strip(), warnings

    data = sdk_object_to_dict(response)
    if not isinstance(data, dict):
        return None, ["response_not_object"]

    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip(), warnings

    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        first_candidate = candidates[0]
        if isinstance(first_candidate, dict):
            content = first_candidate.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
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
    return None, warnings


def usage_output_tokens(response: Any) -> int | None:
    data = sdk_object_to_dict(response)
    if not isinstance(data, dict):
        return None

    usage = (
        data.get("usage_metadata")
        or data.get("usageMetadata")
        or data.get("usage")
    )
    if not isinstance(usage, dict):
        return None

    for key in ("candidates_token_count", "candidatesTokenCount", "output_tokens", "outputTokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return None


def response_id(response: Any) -> str | None:
    data = sdk_object_to_dict(response)
    if not isinstance(data, dict):
        return None
    value = data.get("response_id") or data.get("responseId") or data.get("id")
    return value if isinstance(value, str) else None


def load_result_records(raw_results_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with raw_results_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                records.append(
                    {
                        "key": f"__invalid_json_line_{line_number}",
                        "error": {
                            "status": "invalid_result_json",
                            "message": str(error),
                            "line_number": line_number,
                        },
                    }
                )
                continue
            if not isinstance(data, dict):
                records.append(
                    {
                        "key": f"__invalid_object_line_{line_number}",
                        "error": {
                            "status": "result_line_not_object",
                            "line_number": line_number,
                            "value": data,
                        },
                    }
                )
                continue
            records.append(data)
    return records


def result_key(record: dict[str, Any]) -> str | None:
    key = record.get("key")
    if isinstance(key, str) and key:
        return key

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        key = metadata.get("key")
        if isinstance(key, str) and key:
            return key

    return None


def result_error(record: dict[str, Any]) -> Any:
    for key in ("error", "status"):
        value = record.get(key)
        if value:
            return value
    return None


def result_response(record: dict[str, Any]) -> Any:
    for key in ("response", "generateContentResponse", "generate_content_response"):
        value = record.get(key)
        if value is not None:
            return value
    return None


def make_actual_record(
    *,
    request_record: dict[str, Any],
    result_record: dict[str, Any] | None,
    model_label: str,
    model: str,
    max_tokens: int | None,
    temperature: float | None,
    batch_job_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_index = request_record.get("row_index")
    request_key = request_record.get("request_key") or request_record.get("response_key")

    trace: dict[str, Any] = {
        "row_index": row_index,
        "request_key": request_key,
        "command": request_record.get("command"),
        "result_present": result_record is not None,
        "warnings": [],
    }

    if result_record is None:
        error_text = "batch result is missing for this request key"
        actual = {
            "model_label": model_label,
            "row_index": row_index,
            "model": model,
            "base_url": None,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "request_success": False,
            "timeout": False,
            "latency_ms": None,
            "http_status": None,
            "output_tokens": None,
            "raw_content": None,
            "api_response_id": None,
            "error": error_text,
            "request_started_at_utc": None,
            "request_finished_at_utc": None,
            "batch_job_name": batch_job_name,
            "request_key": request_key,
        }
        trace["error"] = error_text
        return actual, trace

    error_value = result_error(result_record)
    response = result_response(result_record)
    raw_content, warnings = extract_text_from_response(response)
    trace["warnings"].extend(warnings)

    if error_value:
        error_text = json.dumps(error_value, ensure_ascii=False, separators=(",", ":"))
    elif raw_content is None:
        error_text = "could not extract raw_content from batch result response"
    else:
        error_text = None

    actual = {
        "model_label": model_label,
        "row_index": row_index,
        "model": model,
        "base_url": None,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "request_success": error_text is None,
        "timeout": False,
        "latency_ms": None,
        "http_status": None,
        "output_tokens": usage_output_tokens(response),
        "raw_content": raw_content,
        "api_response_id": response_id(response),
        "error": error_text,
        "request_started_at_utc": None,
        "request_finished_at_utc": None,
        "batch_job_name": batch_job_name,
        "request_key": request_key,
    }
    if error_text:
        trace["error"] = error_text
    return actual, trace


def build_keyed_result_map(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    keyed: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    for ordinal, record in enumerate(records, start=1):
        key = result_key(record)
        if not key:
            errors.append(
                {
                    "scope": "result",
                    "status": "missing_result_key",
                    "result_ordinal_1_based": ordinal,
                    "record": record,
                }
            )
            continue

        if key in keyed:
            errors.append(
                {
                    "scope": "result",
                    "status": "duplicate_result_key",
                    "key": key,
                    "result_ordinal_1_based": ordinal,
                }
            )
            continue

        keyed[key] = record

    return keyed, errors


def fetch_batch(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a JSON object")

    job_name = args.job or manifest.get("batch_job_name")
    if not isinstance(job_name, str) or not job_name:
        raise ValueError("Batch job name is missing. Pass --job or use a manifest with batch_job_name.")

    output_dir = Path(manifest.get("output_dir") or manifest_path.parent)
    raw_results_path = Path(args.raw_results) if args.raw_results else output_dir / "batch_results_raw.jsonl"
    trace_path = Path(args.trace) if args.trace else output_dir / "student_batch_trace.jsonl"
    error_path = Path(args.errors) if args.errors else output_dir / "student_batch_errors.jsonl"
    status_path = output_dir / "batch_status.json"
    actual_output_path = Path(args.actual_output)

    for path in (raw_results_path, trace_path, error_path, actual_output_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output file already exists: {path}. Use --overwrite.")

    client = genai.Client(api_key=get_api_key())

    if args.poll:
        batch_job = wait_for_job(client, job_name, args.poll_sec)
    else:
        batch_job = client.batches.get(name=job_name)

    state_name = get_state_name(batch_job)
    batch_job_dict = sdk_object_to_dict(batch_job)
    write_json(status_path, batch_job_dict)

    print(f"job: {job_name}")
    print(f"state: {state_name}")
    print(f"status_file: {status_path}")

    if state_name != "JOB_STATE_SUCCEEDED":
        error = get_value(batch_job, "error")
        if error:
            print(f"error: {sdk_object_to_dict(error)}")
        if not args.allow_not_succeeded:
            return 1

    result_file_name = output_file_name_from_job(batch_job)
    if not result_file_name:
        raise RuntimeError(
            "batch result file is missing. "
            "This fetch script expects a file-batch job created by "
            "sft_submit_gemini_student_batch_from_messages.py."
        )

    file_content = client.files.download(file=result_file_name)
    if isinstance(file_content, bytes):
        raw_results_text = file_content.decode("utf-8")
    else:
        raw_results_text = str(file_content)

    write_text(raw_results_path, raw_results_text)

    result_records = load_result_records(raw_results_path)
    keyed_results, result_errors = build_keyed_result_map(result_records)

    requests = manifest.get("requests")
    if not isinstance(requests, list):
        raise ValueError("manifest.requests must be a list")

    max_tokens = None
    temperature = None
    generation_config = manifest.get("generation_config")
    if isinstance(generation_config, dict):
        max_tokens_value = generation_config.get("max_output_tokens")
        temperature_value = generation_config.get("temperature")
        if isinstance(max_tokens_value, int) and not isinstance(max_tokens_value, bool):
            max_tokens = max_tokens_value
        if isinstance(temperature_value, (int, float)) and not isinstance(temperature_value, bool):
            temperature = float(temperature_value)

    actual_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = list(result_errors)

    seen_request_keys: set[str] = set()

    sorted_requests = sorted(
        [request for request in requests if isinstance(request, dict)],
        key=lambda item: int(item.get("row_index", 0)),
    )

    for request_record in sorted_requests:
        request_key = request_record.get("request_key") or request_record.get("response_key")
        if not isinstance(request_key, str) or not request_key:
            error_records.append(
                {
                    "scope": "manifest",
                    "status": "request_key_missing",
                    "request_record": request_record,
                }
            )
            continue

        seen_request_keys.add(request_key)
        result_record = keyed_results.get(request_key)
        actual, trace = make_actual_record(
            request_record=request_record,
            result_record=result_record,
            model_label=args.model_label,
            model=str(manifest.get("model") or args.model),
            max_tokens=max_tokens,
            temperature=temperature,
            batch_job_name=job_name,
        )
        actual_records.append(actual)
        trace_records.append(trace)

        if not actual["request_success"]:
            error_records.append(
                {
                    "scope": "actual",
                    "status": "request_failed_or_unreadable",
                    "row_index": actual.get("row_index"),
                    "request_key": request_key,
                    "error": actual.get("error"),
                }
            )

    extra_keys = sorted(set(keyed_results.keys()) - seen_request_keys)
    for extra_key in extra_keys:
        error_records.append(
            {
                "scope": "result",
                "status": "result_key_not_in_manifest",
                "key": extra_key,
            }
        )

    expected_count = int(manifest.get("expected_response_count") or len(sorted_requests))
    if len(actual_records) != expected_count:
        error_records.append(
            {
                "scope": "actual",
                "status": "actual_count_mismatch",
                "expected_count": expected_count,
                "actual_count": len(actual_records),
            }
        )

    write_jsonl(actual_output_path, actual_records, overwrite=args.overwrite)
    write_jsonl(trace_path, trace_records, overwrite=args.overwrite)
    write_jsonl(error_path, error_records, overwrite=args.overwrite)

    success_count = sum(1 for record in actual_records if record.get("request_success") is True)
    failure_count = len(actual_records) - success_count

    manifest["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["fetch_state"] = state_name
    manifest["result_file_name"] = result_file_name
    manifest["raw_results_file"] = str(raw_results_path)
    manifest["actual_output_file"] = str(actual_output_path)
    manifest["trace_file"] = str(trace_path)
    manifest["error_file"] = str(error_path)
    manifest["result_record_count"] = len(result_records)
    manifest["actual_record_count"] = len(actual_records)
    manifest["actual_success_count"] = success_count
    manifest["actual_failure_count"] = failure_count
    manifest["fetch_error_count"] = len(error_records)
    manifest["last_fetch_utc"] = utc_now_iso()
    write_json(manifest_path, manifest)

    print(f"raw_results: {raw_results_path}")
    print(f"actual_output: {actual_output_path}")
    print(f"trace: {trace_path}")
    print(f"errors: {error_path}")
    print(f"actual_record_count: {len(actual_records)}")
    print(f"actual_success_count: {success_count}")
    print(f"actual_failure_count: {failure_count}")
    print(f"fetch_error_count: {len(error_records)}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a Gemini file-batch result and build ordered actual responses JSONL "
            "for post_sft_eval.py."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job", default="")
    parser.add_argument("--actual-output", default="actual/responses_35_flash.jsonl")
    parser.add_argument("--model-label", default="gemini_35_flash")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--raw-results", default="")
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
