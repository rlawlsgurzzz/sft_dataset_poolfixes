# datasets/test_sft_messages JSONL을 Gemini student batch 입력으로 변환한다.
# 각 row의 messages 중 system/user content만 그대로 전송하고 assistant label은 제외한다.
# Batch input JSONL에는 row별 key를 붙여 결과를 row_index로 복원 가능하게 만든다.
# dry-run은 실제 명령 목록과 sample_count만 출력한다.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types
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


REQUIRED_USER_PAYLOAD_KEYS = {
    "input",
    "commandAnalysis",
    "output_schema_example",
    "hard_constraints",
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line is not allowed")
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_number}: row root must be a JSON object")
            rows.append((line_number, data))
    return rows


def first_message_content(messages: list[Any], role: str, line_number: int) -> str:
    matches: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            matches.append(content)
    if not matches:
        raise ValueError(f"line {line_number}: missing {role!r} message content")
    if len(matches) > 1:
        raise ValueError(f"line {line_number}: multiple {role!r} messages are not supported")
    if not matches[0]:
        raise ValueError(f"line {line_number}: empty {role!r} message content")
    return matches[0]


def extract_student_parts(row: dict[str, Any], line_number: int) -> tuple[str, str, int]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"line {line_number}: messages must be a list")

    system_content = first_message_content(messages, "system", line_number)
    user_content = first_message_content(messages, "user", line_number)

    assistant_count = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistant_count += 1

    return system_content, user_content, assistant_count


def command_from_user_content(user_content: str, line_number: int) -> str:
    try:
        payload = json.loads(user_content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"line {line_number}: user content is not valid compact JSON: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number}: user payload root must be an object")

    missing = sorted(REQUIRED_USER_PAYLOAD_KEYS - set(payload.keys()))
    if missing:
        raise ValueError(f"line {line_number}: user payload missing keys: {missing}")

    input_value = payload.get("input")
    if not isinstance(input_value, dict):
        raise ValueError(f"line {line_number}: user payload input must be an object")

    command = input_value.get("command")
    if not isinstance(command, str):
        raise ValueError(f"line {line_number}: user payload input.command must be a string")

    return command


def build_generate_content_request(
    *,
    system_content: str,
    user_content: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    thinking_level: str,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "candidate_count": 1,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",
    }
    if thinking_level:
        generation_config["thinking_config"] = {"thinking_level": thinking_level}

    return {
        "system_instruction": {"parts": [{"text": system_content}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}],
            }
        ],
        "generation_config": generation_config,
    }


def select_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    start_index: int,
    sample_limit: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    if start_index < 1:
        raise ValueError("--start-index must be greater than zero")
    selected = rows[start_index - 1 :]
    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError("--sample-limit must be greater than zero")
        selected = selected[:sample_limit]
    return selected


def build_batch_files(args: argparse.Namespace) -> dict[str, Any]:
    messages_path = Path(args.messages)
    rows = read_jsonl(messages_path)
    selected_rows = select_rows(
        rows,
        start_index=args.start_index,
        sample_limit=args.sample_limit,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / run_id
    batch_input_path = output_dir / "batch_input.jsonl"
    manifest_path = output_dir / "manifest.json"

    batch_records: list[dict[str, Any]] = []
    manifest_requests: list[dict[str, Any]] = []
    commands: list[tuple[int, str]] = []
    assistant_label_count = 0

    for ordinal, (line_number, row) in enumerate(selected_rows, start=1):
        system_content, user_content, assistant_count = extract_student_parts(row, line_number)
        command = command_from_user_content(user_content, line_number)
        assistant_label_count += assistant_count

        request_key = f"row_{line_number:06d}"
        request = build_generate_content_request(
            system_content=system_content,
            user_content=user_content,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            thinking_level=args.thinking_level,
        )

        batch_records.append(
            {
                "key": request_key,
                "request": request,
            }
        )
        manifest_requests.append(
            {
                "request_key": request_key,
                "response_key": request_key,
                "ordinal_1_based": ordinal,
                "row_index": line_number,
                "source_line_number": line_number,
                "command": command,
                "system_sha256": sha256_text(system_content),
                "user_payload_sha256": sha256_text(user_content),
                "assistant_label_count_in_source": assistant_count,
            }
        )
        commands.append((line_number, command))

    write_jsonl(batch_input_path, batch_records)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "file_batch_student_messages",
        "model": args.model,
        "display_name": args.display_name,
        "messages": str(messages_path),
        "output_dir": str(output_dir),
        "batch_input_jsonl": str(batch_input_path),
        "manifest": str(manifest_path),
        "sample_limit": args.sample_limit,
        "start_index": args.start_index,
        "request_count": len(batch_records),
        "expected_response_count": len(batch_records),
        "source_line_count": len(rows),
        "selected_line_count": len(selected_rows),
        "assistant_label_count_in_source": assistant_label_count,
        "student_prompt_policy": (
            "Send the exact source system/user content from messages JSONL. "
            "Never send assistant labels."
        ),
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "candidate_count": 1,
            "max_output_tokens": args.max_tokens,
            "response_mime_type": "application/json",
            "thinking_level": args.thinking_level,
        },
        "requests": manifest_requests,
        "status": "built_not_submitted",
    }
    write_json(manifest_path, manifest)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "batch_input_path": batch_input_path,
        "commands": commands,
    }


def submit_batch(args: argparse.Namespace) -> dict[str, Any]:
    built = build_batch_files(args)
    manifest = built["manifest"]
    manifest_path: Path = built["manifest_path"]
    batch_input_path: Path = built["batch_input_path"]

    if args.dry_run:
        for line_number, command in built["commands"]:
            print(f"{line_number:04d}\t{command}")
        print(f"sample_count: {manifest['selected_line_count']}")
        return manifest

    client = genai.Client(api_key=get_api_key())

    uploaded_file = client.files.upload(
        file=str(batch_input_path),
        config=types.UploadFileConfig(
            display_name=f"{args.display_name}-input",
            mime_type="jsonl",
        ),
    )

    batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={"display_name": args.display_name},
    )

    manifest["status"] = "submitted"
    manifest["submitted_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["uploaded_file_name"] = getattr(uploaded_file, "name", None)
    manifest["uploaded_file"] = sdk_object_to_dict(uploaded_file)
    manifest["batch_job_name"] = getattr(batch_job, "name", None)
    manifest["batch_job"] = sdk_object_to_dict(batch_job)
    write_json(manifest_path, manifest)

    print(f"created_batch_job: {manifest['batch_job_name']}")
    print(f"manifest: {manifest_path}")
    print(f"batch_input_jsonl: {batch_input_path}")
    print(f"uploaded_file: {manifest['uploaded_file_name']}")
    print(f"request_count: {manifest['request_count']}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit Gemini Batch requests from student SFT messages JSONL. "
            "The assistant label is removed; exact system/user content is preserved."
        )
    )
    parser.add_argument("--messages", "--datasets", dest="messages", required=True)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--display-name", default="sft-student-batch")
    parser.add_argument("--output-root", default="raw_generations/gemini_student_batch")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-level", default="minimal")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Accepted for command compatibility. Validation is always performed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if args.start_index < 1:
        parser.error("--start-index must be greater than zero")
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be greater than zero")

    submit_batch(args)


if __name__ == "__main__":
    main()
