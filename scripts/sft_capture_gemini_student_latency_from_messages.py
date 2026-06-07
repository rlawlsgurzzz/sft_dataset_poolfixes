# Captures live Gemini latency for SFT student messages.
# Sends only system/user messages; assistant labels are never sent.
# Requires exactly --expected-count rows before any API call starts.
# Writes actual-style JSONL with latency_ms and a compact latency summary JSON.

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types
except Exception as exc:
    raise SystemExit(
        "google-genai is required. Install it with: py -3.11 -m pip install google-genai\n"
        f"Import error: {exc}"
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            rows.append((line_number, json.loads(text)))

    return rows


def extract_system_user(row_obj: dict[str, Any], line_number: int) -> tuple[str, str, str]:
    messages = row_obj.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"line {line_number}: missing messages list")

    system_contents: list[str] = []
    user_contents: list[str] = []
    assistant_seen = False

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"line {line_number}: message is not an object")

        role = message.get("role")
        content = message.get("content")

        if not isinstance(content, str):
            raise ValueError(f"line {line_number}: message content is not a string")

        if role == "system":
            system_contents.append(content)
        elif role == "user":
            user_contents.append(content)
        elif role == "assistant":
            assistant_seen = True
        else:
            raise ValueError(f"line {line_number}: unsupported role {role!r}")

    if len(system_contents) != 1:
        raise ValueError(f"line {line_number}: expected exactly one system message, got {len(system_contents)}")
    if len(user_contents) != 1:
        raise ValueError(f"line {line_number}: expected exactly one user message, got {len(user_contents)}")
    if not assistant_seen:
        raise ValueError(f"line {line_number}: assistant label is missing in source row")

    command = ""
    try:
        user_payload = json.loads(user_contents[0])
        command = str(user_payload.get("input", {}).get("command", ""))
    except Exception:
        command = ""

    return system_contents[0], user_contents[0], command


def build_generate_config(args: argparse.Namespace, system_instruction: str) -> types.GenerateContentConfig:
    config_kwargs: dict[str, Any] = {
        "system_instruction": system_instruction,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "candidate_count": 1,
        "max_output_tokens": args.max_tokens,
        "response_mime_type": "application/json",
    }

    if args.thinking_level:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=args.thinking_level)
        except Exception:
            # Some google-genai versions or models may not expose ThinkingConfig.
            # In that case, the request still proceeds without it.
            pass

    return types.GenerateContentConfig(**config_kwargs)


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    parts_text: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue

        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts_text.append(part_text)

    return "".join(parts_text)


def output_token_count(response: Any) -> int | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    for attr_name in ("candidates_token_count", "output_token_count"):
        value = getattr(usage, attr_name, None)
        if isinstance(value, int):
            return value

    return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)

    if low == high:
        return sorted_values[low]

    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_capture(args: argparse.Namespace) -> None:
    messages_path = Path(args.datasets)
    output_path = Path(args.actual_output)
    summary_path = Path(args.summary_output)
    latency_rows_path = Path(args.latency_rows_output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"summary already exists: {summary_path}")
    if latency_rows_path.exists() and not args.overwrite:
        raise FileExistsError(f"latency rows already exists: {latency_rows_path}")

    rows = read_jsonl(messages_path)

    if len(rows) != args.expected_count:
        raise SystemExit(
            f"expected exactly {args.expected_count} non-empty rows, got {len(rows)}: {messages_path}\n"
            "No API requests were sent."
        )

    prepared: list[dict[str, Any]] = []
    for row_index, (line_number, row_obj) in enumerate(rows, start=1):
        system_instruction, user_content, command = extract_system_user(row_obj, line_number)
        prepared.append(
            {
                "row_index": row_index,
                "source_line_number": line_number,
                "system_instruction": system_instruction,
                "user_content": user_content,
                "command": command,
            }
        )

    print(f"prepared_count: {len(prepared)}")
    for item in prepared:
        command = item["command"] or "(command unavailable)"
        print(f"{item['row_index']:04d}\t{command}")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.")

    client = genai.Client(api_key=api_key)

    ensure_parent(output_path)
    ensure_parent(latency_rows_path)

    output_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []

    for item in prepared:
        row_index = item["row_index"]
        request_key = f"row_{row_index:06d}"
        start_utc = utc_now_iso()
        start = time.perf_counter()

        request_success = False
        timeout = False
        raw_content = ""
        error_text = None
        output_tokens = None
        api_response_id = None

        try:
            config = build_generate_config(args, item["system_instruction"])
            response = client.models.generate_content(
                model=args.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=item["user_content"])],
                    )
                ],
                config=config,
            )
            raw_content = response_text(response)
            output_tokens = output_token_count(response)
            api_response_id_value = getattr(response, "response_id", None)
            if isinstance(api_response_id_value, str):
                api_response_id = api_response_id_value
            request_success = True
        except Exception as exc:
            error_text = repr(exc)
            timeout = "timeout" in error_text.lower()

        end = time.perf_counter()
        finished_utc = utc_now_iso()
        latency_ms = round((end - start) * 1000.0, 3)

        response_row = {
            "model_label": args.model_label,
            "row_index": row_index,
            "model": args.model,
            "base_url": None,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "request_success": request_success,
            "timeout": timeout,
            "latency_ms": latency_ms,
            "http_status": None,
            "output_tokens": output_tokens,
            "raw_content": raw_content,
            "api_response_id": api_response_id,
            "error": error_text,
            "request_started_at_utc": start_utc,
            "request_finished_at_utc": finished_utc,
            "request_key": request_key,
        }

        latency_row = {
            "row_index": row_index,
            "request_key": request_key,
            "request_success": request_success,
            "latency_ms": latency_ms,
            "output_tokens": output_tokens,
            "request_started_at_utc": start_utc,
            "request_finished_at_utc": finished_utc,
            "error": error_text,
            "command": item["command"],
        }

        output_rows.append(response_row)
        latency_rows.append(latency_row)

        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(response_row, ensure_ascii=False) + "\n")

        with latency_rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(latency_row, ensure_ascii=False) + "\n")

        print(
            f"row_index={row_index} success={request_success} "
            f"latency_ms={latency_ms} output_tokens={output_tokens}"
        )

        if args.sleep_sec > 0 and row_index < len(prepared):
            time.sleep(args.sleep_sec)

    successful_latencies = [
        float(row["latency_ms"])
        for row in latency_rows
        if row.get("request_success") is True and isinstance(row.get("latency_ms"), (int, float))
    ]

    summary = {
        "created_at_utc": utc_now_iso(),
        "messages_path": str(messages_path),
        "actual_output": str(output_path),
        "latency_rows_output": str(latency_rows_path),
        "model_label": args.model_label,
        "model": args.model,
        "expected_count": args.expected_count,
        "total_requests": len(latency_rows),
        "successful_requests": sum(1 for row in latency_rows if row.get("request_success") is True),
        "failed_requests": sum(1 for row in latency_rows if row.get("request_success") is not True),
        "avg_latency_ms": round(statistics.mean(successful_latencies), 3) if successful_latencies else None,
        "p50_latency_ms": round(percentile(successful_latencies, 0.50), 3) if successful_latencies else None,
        "p95_latency_ms": round(percentile(successful_latencies, 0.95), 3) if successful_latencies else None,
        "min_latency_ms": round(min(successful_latencies), 3) if successful_latencies else None,
        "max_latency_ms": round(max(successful_latencies), 3) if successful_latencies else None,
        "avg_output_tokens": (
            round(
                statistics.mean(
                    [
                        row["output_tokens"]
                        for row in latency_rows
                        if isinstance(row.get("output_tokens"), int)
                    ]
                ),
                3,
            )
            if any(isinstance(row.get("output_tokens"), int) for row in latency_rows)
            else None
        ),
    }

    write_json(summary_path, summary)

    print(f"actual_output: {output_path}")
    print(f"latency_rows_output: {latency_rows_path}")
    print(f"summary_output: {summary_path}")
    print(
        "latency_summary: "
        + json.dumps(
            {
                "avg_latency_ms": summary["avg_latency_ms"],
                "p50_latency_ms": summary["p50_latency_ms"],
                "p95_latency_ms": summary["p95_latency_ms"],
            },
            ensure_ascii=False,
        )
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture live Gemini latency for exactly 10 SFT student message rows."
    )
    parser.add_argument("--datasets", "--messages", required=True, help="Path to messages JSONL.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--model-label", default="gemini_31_flash_lite")
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--actual-output", required=True, help="New actual-style responses JSONL path.")
    parser.add_argument("--summary-output", required=True, help="New latency summary JSON path.")
    parser.add_argument("--latency-rows-output", required=True, help="New per-row latency JSONL path.")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-level", default="minimal", help="Use empty string to omit.")
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    run_capture(args)

#end


if __name__ == "__main__":
    main()
