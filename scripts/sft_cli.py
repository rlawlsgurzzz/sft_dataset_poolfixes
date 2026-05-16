# SFT dataset 관리 작업의 단일 진입점이다.
# request는 teacher LLM에 보낼 generation payload를 만든다.
# generate는 payload 생성 후 사용자 확인을 받고 teacher raw output을 저장한다.
# validate는 teacher raw output을 검증하고 accepted 저장 시 commandAnalysis를 계산해 추가한다.
# report는 accepted sample 기준으로 taxonomy coverage 문서를 재생성한다.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from sft_file_sequence import next_numbered_index, numbered_path
    from sft_generation_request import (
        DEFAULT_TARGET_SPLIT,
        VALID_SPLITS,
        build_mixed_generation_payload,
        make_default_output_path,
        write_json,
    )
    from sft_validator import validate_file
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_file_sequence import next_numbered_index, numbered_path
    from sft_generation_request import (
        DEFAULT_TARGET_SPLIT,
        VALID_SPLITS,
        build_mixed_generation_payload,
        make_default_output_path,
        write_json,
    )
    from sft_validator import validate_file


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY_PATH = DEFAULT_DATASET_ROOT / "config" / "taxonomy_sot.json"
DEFAULT_TEACHER_MODEL = "gemma-4-31b-it"


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


def build_payload_or_print_error(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        return build_mixed_generation_payload(
            mixed_requests=[
                {
                    "request": args.request,
                    "cycle_start_offset": 0,
                }
            ],
            dataset_root=Path(args.dataset_root),
            taxonomy_path=Path(args.taxonomy),
            target_split=getattr(args, "split", DEFAULT_TARGET_SPLIT),
        )
    except Exception as error:
        print(f"request failed: {error}")
        return None

PRINT_COMMAND_TEXT_POLICY_KEYS = (
    "target_split",
    "same_split_expression_pool_size",
    "existing_same_split_expression_count",
    "other_split_reserved_expression_count",
    "new_unique_command_texts_to_create",
    "samples_using_same_split_cycle",
)


def build_print_payload(payload: dict[str, Any]) -> dict[str, Any]:
    command_text_policy = payload.get("command_text_policy", {})

    if isinstance(command_text_policy, dict):
        compact_command_text_policy = {
            key: command_text_policy[key]
            for key in PRINT_COMMAND_TEXT_POLICY_KEYS
            if key in command_text_policy
        }
    else:
        compact_command_text_policy = {}

    return {
        "request": payload.get("request", {}),
        "target_split": payload.get("target_split"),
        "selected_bucket": payload.get("selected_bucket", {}),
        "existing_valid_paraphrase_samples": payload.get(
            "existing_valid_paraphrase_samples",
            [],
        ),
        "other_split_reserved_command_texts": payload.get(
            "other_split_reserved_command_texts",
            [],
        ),
        "command_text_policy": compact_command_text_policy,
    }


def print_payload_json(payload: dict[str, Any]) -> None:
    print(json.dumps(build_print_payload(payload), ensure_ascii=False, indent=2))


def run_request(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)

    payload = build_payload_or_print_error(args)
    if payload is None:
        return

    output_path = (
        Path(args.output)
        if args.output
        else make_default_output_path(dataset_root / "raw_generations", args.request)
    )

    write_json(output_path, payload)
    print(f"generation_request: {output_path}")

    if args.print_json:
        print_payload_json(payload)


def print_generation_plan(
    payload: dict[str, Any],
    request_path: Path,
    raw_output_path: Path,
    trace_output_path: Path,
) -> None:
    request_info = payload.get("request", {})
    target_split = payload.get("target_split", DEFAULT_TARGET_SPLIT)
    items = payload.get("mixed_generation_requests", [])

    print("request ok")
    print(f"count_to_generate: {payload.get('count_to_generate')}")
    print(f"target_split: {target_split}")
    print(f"path_count: {request_info.get('path_count')}")

    if isinstance(items, list):
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            item_request = item.get("request", {})
            policy = item.get("command_text_policy", {})
            sequence_contract = (
                policy.get("sequence_contract", {})
                if isinstance(policy, dict)
                else {}
            )

            print(f"item {index}:")
            print(f"  raw_request: {item_request.get('raw_request')}")
            print(f"  stable_path: {item_request.get('stable_path')}")
            if item_request.get("skill_stable_path"):
                print(f"  skill_stable_path: {item_request.get('skill_stable_path')}")
            print(f"  base_command_text: {item_request.get('base_command_text')}")
            print(f"  count_to_generate: {item.get('count_to_generate')}")
            print(f"  cycle_start_offset_used: {item.get('cycle_start_offset_used')}")

            if isinstance(policy, dict):
                print(
                    "  command_text_policy: "
                    f"pool={policy.get('same_split_expression_pool_size')}, "
                    f"existing_same_split={policy.get('existing_same_split_expression_count')}, "
                    f"reserved_other_split={policy.get('other_split_reserved_expression_count')}, "
                    f"new={policy.get('new_unique_command_texts_to_create')}, "
                    f"cycle={policy.get('samples_using_same_split_cycle')}, "
                    f"cycle_start_offset={sequence_contract.get('cycle_start_offset_0_based')}"
                )

    print(f"request_file: {request_path}")
    print(f"raw_generation_file: {raw_output_path}")
    print(f"trace_file: {trace_output_path}")


def run_generate(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    raw_dir = dataset_root / "raw_generations"

    payload = build_payload_or_print_error(args)
    if payload is None:
        return

    request_path = (
        Path(args.output)
        if args.output
        else make_default_output_path(raw_dir, args.request)
    )

    batch_index = next_numbered_index(raw_dir, "batch", "_raw.jsonl")
    raw_output_path = numbered_path(raw_dir, "batch", batch_index, "_raw.jsonl")
    trace_output_path = numbered_path(raw_dir, "batch", batch_index, "_trace.jsonl")

    print_generation_plan(
        payload=payload,
        request_path=request_path,
        raw_output_path=raw_output_path,
        trace_output_path=trace_output_path,
    )

    if args.print_json:
        print_payload_json(payload)

    answer = input("type y to continue, n to cancel: ").strip().lower()
    if answer != "y":
        print("cancelled")
        return

    write_json(request_path, payload)

    try:
        from sft_teacher_client import run_teacher_generation
    except ModuleNotFoundError as error:
        if error.name in {"google", "google.genai"}:
            print(
                "generate failed: missing package 'google-genai'. Install it with: python -m pip install google-genai"
            )
            return
        raise

    result = run_teacher_generation(
        input_path=request_path,
        output_path=raw_output_path,
        trace_path=trace_output_path,
        model_name=args.model,
        max_tokens=args.max_tokens,
        print_json=False,
        # print_json=args.print_json, 이면 llm 출력 결과도 모두 보임
        stream_output=args.stream_output,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_validate(args: argparse.Namespace) -> None:
    result = validate_file(
        input_path=Path(args.input),
        dataset_root=Path(args.dataset_root),
        taxonomy_path=Path(args.taxonomy),
        write_outputs=not args.dry_run,
    )

    if result.get("skipped"):
        print("validate skipped: input already validated")
        return

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.refresh_report and not args.dry_run:
        run_report(Path(args.dataset_root), Path(args.taxonomy))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic SFT dataset request/generate/validate/report CLI"
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))

    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report",
        help="Regenerate taxonomy coverage reports from accepted samples.",
    )
    report_parser.set_defaults(
        func=lambda args: run_report(Path(args.dataset_root), Path(args.taxonomy))
    )

    request_parser = subparsers.add_parser(
        "request",
        help="Build teacher LLM generation request payload.",
    )
    request_parser.add_argument(
        "request", help="Generation request path, e.g. c1-2-1-3-1-10.4"
    )
    request_parser.add_argument("--output", default="")
    request_parser.add_argument("--print-json", action="store_true")
    request_parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default=DEFAULT_TARGET_SPLIT,
        help="Target dataset split for generated samples.",
    )
    request_parser.set_defaults(func=run_request)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Build request payload and save teacher raw output after confirmation.",
    )
    generate_parser.add_argument(
        "request", help="Generation request path, e.g. c1-2-1-3-1-10.4"
    )
    generate_parser.add_argument(
        "--output", default="", help="Optional request payload path."
    )
    generate_parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    generate_parser.add_argument("--max-tokens", type=int, default=60000)
    generate_parser.add_argument("--print-json", action="store_true")
    generate_parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default=DEFAULT_TARGET_SPLIT,
        help="Target dataset split for generated samples.",
    )
    generate_parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Print teacher response chunks while generating.",
    )
    generate_parser.set_defaults(func=run_generate)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate teacher raw output, compute commandAnalysis, and split accepted/rejected.",
    )
    validate_parser.add_argument(
        "--input",
        required=True,
        help="Teacher raw output JSON/JSONL path.",
    )
    validate_parser.add_argument("--dry-run", action="store_true")
    validate_parser.add_argument("--refresh-report", action="store_true")
    validate_parser.set_defaults(func=run_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
