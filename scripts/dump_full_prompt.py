"""Build a full teacher prompt text from one or two mixed generation requests.

Supported positional form:
  python scripts/dump_full_prompt.py train train c1-1-1-1-1-1.4 c3-2-12-13-13-1/9-6-8.5 full_prompt/sft_full_prompt_0001.txt

The second split token is accepted only for compatibility with the sft_cli dump form.
Each request count must be in 1..5. At most two request paths are allowed.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from sft_generation_request import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_TARGET_SPLIT,
        DEFAULT_TAXONOMY_PATH,
        VALID_SPLITS,
        build_mixed_generation_payload,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_generation_request import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_TARGET_SPLIT,
        DEFAULT_TAXONOMY_PATH,
        VALID_SPLITS,
        build_mixed_generation_payload,
    )

MAX_MIXED_REQUESTS = 2
MAX_REQUEST_COUNT = 5


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
            return str(ast.literal_eval(value)).strip()

    raise RuntimeError(f"SYSTEM_PROMPT not found: {teacher_client_path}")


def split_request_count(raw_request: str) -> tuple[str, int]:
    request = raw_request.strip()
    if not request:
        raise ValueError("request cannot be empty")

    if "." not in request:
        raise ValueError(f"request must end with .<count>: {raw_request}")

    prefix, count_text = request.rsplit(".", 1)
    if not prefix or not count_text.isdigit():
        raise ValueError(f"request must end with .<count>: {raw_request}")

    count = int(count_text)
    if count <= 0:
        raise ValueError(f"request count must be greater than zero: {raw_request}")
    if count > MAX_REQUEST_COUNT:
        raise ValueError(
            f"request count must be {MAX_REQUEST_COUNT} or less for dump: {raw_request}"
        )

    return prefix, count


def normalize_dump_items(split: str, items: Iterable[str]) -> tuple[list[str], Path]:
    tokens = [item.strip() for item in items if item.strip()]

    if tokens and tokens[0] in VALID_SPLITS:
        embedded_split = tokens.pop(0)
        if embedded_split != split:
            raise ValueError(
                f"dump split mismatch: first split is {split}, second split is {embedded_split}"
            )

    if len(tokens) < 2:
        raise ValueError(
            "dump requires at least one request and one output path: "
            "dump <split> [same_split] <request1> [request2] <output_path>"
        )

    output_path = Path(tokens[-1])
    requests = tokens[:-1]

    if len(requests) > MAX_MIXED_REQUESTS:
        raise ValueError(
            f"dump supports at most {MAX_MIXED_REQUESTS} request paths, got {len(requests)}"
        )

    prefixes: list[str] = []
    for request in requests:
        prefix, _count = split_request_count(request)
        prefixes.append(prefix)

    if len(prefixes) != len(set(prefixes)):
        raise ValueError("dump request paths must be distinct")

    return requests, output_path


def build_mixed_requests(raw_requests: list[str]) -> list[dict[str, Any]]:
    mixed_requests: list[dict[str, Any]] = []

    for request in raw_requests:
        split_request_count(request)
        mixed_requests.append(
            {
                "request": request,
                "cycle_start_offset": 0,
            }
        )

    return mixed_requests


def build_full_prompt_text(system_prompt: str, payload: dict[str, Any]) -> tuple[str, str]:
    user_contents = json.dumps(payload, ensure_ascii=False, indent=2)
    combined_prompt = (
        "<SYSTEM_INSTRUCTION>\n"
        f"{system_prompt}\n"
        "</SYSTEM_INSTRUCTION>\n\n"
        "<USER_CONTENTS>\n"
        f"{user_contents}\n"
        "</USER_CONTENTS>\n"
    )
    return combined_prompt, user_contents


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def run_dump_full_prompt(args: argparse.Namespace) -> None:
    split = getattr(args, "split", DEFAULT_TARGET_SPLIT)
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}: {split}")

    requests, output_path = normalize_dump_items(split, getattr(args, "items"))
    mixed_requests = build_mixed_requests(requests)

    dataset_root = Path(getattr(args, "dataset_root", DEFAULT_DATASET_ROOT))
    taxonomy_path = Path(getattr(args, "taxonomy", DEFAULT_TAXONOMY_PATH))
    teacher_client_path = Path(
        getattr(args, "teacher_client", "")
        or Path(__file__).resolve().parent / "sft_teacher_client.py"
    )

    system_prompt = extract_system_prompt(teacher_client_path)
    payload = build_mixed_generation_payload(
        mixed_requests=mixed_requests,
        dataset_root=dataset_root,
        taxonomy_path=taxonomy_path,
        target_split=split,
    )
    combined_prompt, user_contents = build_full_prompt_text(system_prompt, payload)

    system_output = getattr(args, "system_output", "")
    user_output = getattr(args, "user_output", "")

    write_text_file(output_path, combined_prompt)
    if system_output:
        write_text_file(Path(system_output), system_prompt)
    if user_output:
        write_text_file(Path(user_output), user_contents)

    print(f"wrote {output_path}")
    if system_output:
        print(f"wrote {system_output}")
    if user_output:
        print(f"wrote {user_output}")
    print("mixed_requests:")
    for item in mixed_requests:
        print(f"- request={item['request']} cycle_start_offset={item['cycle_start_offset']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build full teacher prompt text without calling the teacher API."
    )
    parser.add_argument(
        "split",
        choices=VALID_SPLITS,
        default=DEFAULT_TARGET_SPLIT,
        help="Target dataset split.",
    )
    parser.add_argument(
        "items",
        nargs="+",
        help=(
            "Use: [same_split] <request1> [request2] <output_path>. "
            "The optional same_split token exists for sft_cli-style commands."
        ),
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument(
        "--teacher-client",
        default=str(Path(__file__).resolve().parent / "sft_teacher_client.py"),
        help="Path to sft_teacher_client.py containing SYSTEM_PROMPT.",
    )
    parser.add_argument(
        "--system-output",
        default="",
        help="Optional path for extracted system prompt. Omit to skip.",
    )
    parser.add_argument(
        "--user-output",
        default="",
        help="Optional path for user payload JSON. Omit to skip.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_dump_full_prompt(args)
    except Exception as error:
        parser.exit(status=1, message=f"dump failed: {error}\n")


if __name__ == "__main__":
    main()
