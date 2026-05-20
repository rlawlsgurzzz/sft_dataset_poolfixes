"""Build a full teacher prompt (SYSTEM + USER payload) from mixed requests.

Examples:
  python scripts/dump_full_prompt.py \
    --split train \
    --request c2-4-8-2-6-1.5 \
    --request c2-4-12-2-10-1.4 \
    --output full_prompt/sft_full_prompt_0001.txt
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from sft_generation_request import (
    DEFAULT_TARGET_SPLIT,
    VALID_SPLITS,
    build_mixed_generation_payload,
)


def extract_system_prompt(teacher_client_path: Path) -> str:
    src = teacher_client_path.read_text(encoding="utf-8")
    module = ast.parse(src)

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != "SYSTEM_PROMPT":
                continue

            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                return ast.literal_eval(value.func.value).strip()
            return ast.literal_eval(value)

    raise RuntimeError("SYSTEM_PROMPT not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full teacher prompt text from one or more mixed generation requests.",
    )
    parser.add_argument(
        "--request",
        action="append",
        required=True,
        help=(
            "Mixed request entry (repeatable), e.g. c2-4-8-2-6-1.5. "
            "Order matters."
        ),
    )
    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default=DEFAULT_TARGET_SPLIT,
        help="Target split.",
    )
    parser.add_argument(
        "--output",
        default="full_prompt/sft_full_prompt_0001.txt",
        help="Output full prompt txt path.",
    )
    parser.add_argument(
        "--system-output",
        default="sft_system_instruction.txt",
        help="Optional path for extracted system prompt.",
    )
    parser.add_argument(
        "--user-output",
        default="sft_user_contents_train.json",
        help="Optional path for user payload JSON.",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional dataset root override.",
    )
    parser.add_argument(
        "--taxonomy",
        default="",
        help="Optional taxonomy path override.",
    )
    parser.add_argument(
        "--teacher-client",
        default="scripts/sft_teacher_client.py",
        help="Path to teacher client file containing SYSTEM_PROMPT.",
    )
    return parser.parse_args()


def build_mixed_requests(raw_requests: list[str]) -> list[dict[str, Any]]:
    mixed_requests: list[dict[str, Any]] = []
    cycle_start_offset = 0

    for raw_request in raw_requests:
        request_text = raw_request.strip()
        if not request_text:
            raise ValueError("request cannot be empty")

        try:
            count = int(request_text.rsplit(".", 1)[1])
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"invalid request count suffix: {request_text}") from error

        if count < 0:
            raise ValueError(f"count suffix must be non-negative: {request_text}")

        mixed_requests.append(
            {
                "request": request_text,
                "cycle_start_offset": cycle_start_offset,
            }
        )
        cycle_start_offset += count

    return mixed_requests


def main() -> None:
    args = parse_args()

    system_prompt = extract_system_prompt(Path(args.teacher_client))
    mixed_requests = build_mixed_requests(args.request)

    payload_kwargs: dict[str, Any] = {
        "mixed_requests": mixed_requests,
        "target_split": args.split,
    }
    if args.dataset_root:
        payload_kwargs["dataset_root"] = Path(args.dataset_root)
    if args.taxonomy:
        payload_kwargs["taxonomy_path"] = Path(args.taxonomy)

    payload = build_mixed_generation_payload(**payload_kwargs)
    user_contents = json.dumps(payload, ensure_ascii=False, indent=2)

    combined_prompt = (
        "<SYSTEM_INSTRUCTION>\n"
        f"{system_prompt}\n"
        "</SYSTEM_INSTRUCTION>\n\n"
        "<USER_CONTENTS>\n"
        f"{user_contents}\n"
        "</USER_CONTENTS>\n"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined_prompt, encoding="utf-8")

    system_output_path = Path(args.system_output)
    system_output_path.parent.mkdir(parents=True, exist_ok=True)
    system_output_path.write_text(system_prompt, encoding="utf-8")

    user_output_path = Path(args.user_output)
    user_output_path.parent.mkdir(parents=True, exist_ok=True)
    user_output_path.write_text(user_contents, encoding="utf-8")

    print(f"wrote {output_path}")
    print(f"wrote {system_output_path}")
    print(f"wrote {user_output_path}")
    print("mixed_requests:")
    for item in mixed_requests:
        print(
            f"- request={item['request']} cycle_start_offset={item['cycle_start_offset']}"
        )


if __name__ == "__main__":
    main()
