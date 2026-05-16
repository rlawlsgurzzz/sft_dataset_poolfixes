#!/usr/bin/env python3
"""
reports/general_coverage.md의 leaf id 목록을 기준으로 accepted sample 중복을 계산한다.
중복 기준은 leaf 내부의 command_text + output.thinking + output.dialog exact match다.
같은 key가 N개 있으면 duplicate count는 N - 1로 계산한다.
raw sample, accepted sample, coverage 파일은 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEAF_LINE_RE = re.compile(r"^\s*(\d+(?:-\d+){5})\s+@\s+")


@dataclass(frozen=True)
class LeafEntry:
    path: str
    command_text: str
    expected_count: int
    edge_flags_text: str
    record_refs: tuple[str, ...]
    line_number: int


@dataclass(frozen=True)
class DuplicateStats:
    leaf: LeafEntry
    loaded_count: int
    missing_count: int
    duplicate_count: int
    duplicate_group_count: int
    max_group_size: int
    groups: dict[tuple[str, str, str], list[str]]


def detect_dataset_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).resolve()

    current = Path.cwd().resolve()
    script_path = Path(__file__).resolve()

    candidates = [
        current,
        script_path.parent,
        *current.parents,
        *script_path.parents,
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "reports" / "general_coverage.md").exists() and (
            candidate / "accepted"
        ).is_dir():
            return candidate

    raise FileNotFoundError(
        "dataset root를 찾지 못했다. --dataset-root로 레포 루트를 지정해라."
    )


def parse_quoted_command(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def parse_general_coverage(path: Path) -> list[LeafEntry]:
    leaves: list[LeafEntry] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\n")
            if not LEAF_LINE_RE.match(line):
                continue

            stripped = line.strip()
            parts = stripped.split(" @ ", 4)
            if len(parts) != 5:
                continue

            path_text, quoted_command, count_text, edge_flags_text, refs_text = parts

            try:
                expected_count = int(count_text)
            except ValueError:
                expected_count = 0

            record_refs = tuple(
                ref.strip()
                for ref in refs_text.split(",")
                if ref.strip()
            )

            leaves.append(
                LeafEntry(
                    path=path_text,
                    command_text=parse_quoted_command(quoted_command),
                    expected_count=expected_count,
                    edge_flags_text=edge_flags_text.strip(),
                    record_refs=record_refs,
                    line_number=line_number,
                )
            )

    return leaves


def iter_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: list[dict[str, Any]] = []

    if text.startswith("["):
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise ValueError(f"JSON array가 아니다: {path}")
        for item in loaded:
            if isinstance(item, dict):
                records.append(item)
        return records

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: JSON parse failed: {error}") from error
        if isinstance(item, dict):
            records.append(item)

    return records


def build_sample_lookup(accepted_dir: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}

    for jsonl_path in sorted(accepted_dir.rglob("*.jsonl")):
        for sample in iter_json_records(jsonl_path):
            sample_id = sample.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                continue
            key = f"{jsonl_path.name}_{sample_id}"
            lookup[key] = sample

    return lookup


def get_command_text(sample: dict[str, Any]) -> str:
    command_spec = sample.get("command_spec")
    if isinstance(command_spec, dict):
        command_text = command_spec.get("command_text")
        if isinstance(command_text, str):
            return command_text

    input_obj = sample.get("input")
    if isinstance(input_obj, dict):
        nested_input = input_obj.get("input")
        if isinstance(nested_input, dict):
            command = nested_input.get("command")
            if isinstance(command, str):
                return command

    return ""


def get_thinking(sample: dict[str, Any]) -> str:
    output = sample.get("output")
    if not isinstance(output, dict):
        return ""
    thinking = output.get("thinking")
    if isinstance(thinking, str):
        return thinking
    return json.dumps(thinking, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_dialog_key(sample: dict[str, Any]) -> str:
    output = sample.get("output")
    if not isinstance(output, dict):
        return "null"
    dialog = output.get("dialog")
    return json.dumps(dialog, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_duplicate_key(sample: dict[str, Any]) -> tuple[str, str, str]:
    return (
        get_command_text(sample),
        get_thinking(sample),
        get_dialog_key(sample),
    )


def compute_leaf_stats(
    leaves: list[LeafEntry],
    sample_lookup: dict[str, dict[str, Any]],
) -> list[DuplicateStats]:
    results: list[DuplicateStats] = []

    for leaf in leaves:
        key_to_refs: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        missing_count = 0

        for record_ref in leaf.record_refs:
            sample = sample_lookup.get(record_ref)
            if sample is None:
                missing_count += 1
                continue

            key_to_refs[make_duplicate_key(sample)].append(record_ref)

        duplicate_groups = {
            key: refs
            for key, refs in key_to_refs.items()
            if len(refs) > 1
        }

        duplicate_count = sum(len(refs) - 1 for refs in duplicate_groups.values())
        max_group_size = max((len(refs) for refs in duplicate_groups.values()), default=0)

        results.append(
            DuplicateStats(
                leaf=leaf,
                loaded_count=sum(len(refs) for refs in key_to_refs.values()),
                missing_count=missing_count,
                duplicate_count=duplicate_count,
                duplicate_group_count=len(duplicate_groups),
                max_group_size=max_group_size,
                groups=duplicate_groups,
            )
        )

    return results


def truncate(value: str, width: int) -> str:
    if width <= 0 or len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "…"


def print_summary(
    stats: list[DuplicateStats],
    *,
    only_duplicates: bool,
    command_width: int,
    details: bool,
) -> None:
    rows = [item for item in stats if not only_duplicates or item.duplicate_count > 0]

    print(
        "path | samples | loaded | missing | dup | dup_groups | max_group | command"
    )
    print("-" * 100)

    shown_total_samples = 0
    shown_total_loaded = 0
    shown_total_missing = 0
    shown_total_dup = 0
    shown_leaves_with_dup = 0

    for item in rows:
        leaf = item.leaf
        shown_total_samples += leaf.expected_count
        shown_total_loaded += item.loaded_count
        shown_total_missing += item.missing_count
        shown_total_dup += item.duplicate_count
        if item.duplicate_count > 0:
            shown_leaves_with_dup += 1

        print(
            f"{leaf.path} | "
            f"{leaf.expected_count} | "
            f"{item.loaded_count} | "
            f"{item.missing_count} | "
            f"{item.duplicate_count} | "
            f"{item.duplicate_group_count} | "
            f"{item.max_group_size} | "
            f"{truncate(leaf.command_text, command_width)}"
        )

        if details and item.groups:
            sorted_groups = sorted(
                item.groups.items(),
                key=lambda kv: (-len(kv[1]), kv[0]),
            )
            for group_index, ((command, thinking, dialog_key), refs) in enumerate(
                sorted_groups,
                start=1,
            ):
                print(
                    f"  group {group_index}: size={len(refs)}, dup={len(refs) - 1}"
                )
                print(f"    command: {command}")
                print(f"    thinking: {thinking}")
                print(f"    dialog: {dialog_key}")
                print(f"    refs: {', '.join(refs)}")

    all_total_samples = sum(item.leaf.expected_count for item in stats)
    all_total_loaded = sum(item.loaded_count for item in stats)
    all_total_missing = sum(item.missing_count for item in stats)
    all_total_dup = sum(item.duplicate_count for item in stats)
    all_leaves_with_dup = sum(1 for item in stats if item.duplicate_count > 0)

    print("-" * 100)
    if only_duplicates:
        print(
            "shown_total: "
            f"leaves={len(rows)}, "
            f"leaves_with_dup={shown_leaves_with_dup}, "
            f"samples={shown_total_samples}, "
            f"loaded={shown_total_loaded}, "
            f"missing={shown_total_missing}, "
            f"dup={shown_total_dup}"
        )

    print(
        "grand_total: "
        f"leaves={len(stats)}, "
        f"leaves_with_dup={all_leaves_with_dup}, "
        f"samples={all_total_samples}, "
        f"loaded={all_total_loaded}, "
        f"missing={all_total_missing}, "
        f"dup={all_total_dup}"
    )


def write_json_report(path: Path, stats: list[DuplicateStats]) -> None:
    payload: list[dict[str, Any]] = []

    for item in stats:
        groups_payload: list[dict[str, Any]] = []
        for (command, thinking, dialog_key), refs in sorted(
            item.groups.items(),
            key=lambda kv: (-len(kv[1]), kv[0]),
        ):
            groups_payload.append(
                {
                    "size": len(refs),
                    "duplicate_count": len(refs) - 1,
                    "command": command,
                    "thinking": thinking,
                    "dialog": json.loads(dialog_key),
                    "refs": refs,
                }
            )

        payload.append(
            {
                "path": item.leaf.path,
                "command": item.leaf.command_text,
                "expected_count": item.leaf.expected_count,
                "loaded_count": item.loaded_count,
                "missing_count": item.missing_count,
                "duplicate_count": item.duplicate_count,
                "duplicate_group_count": item.duplicate_group_count,
                "max_group_size": item.max_group_size,
                "groups": groups_payload,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "reports/general_coverage.md의 leaf별 accepted sample 안에서 "
            "command_text + output.thinking + output.dialog exact duplicate를 계산한다."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="레포 루트. 생략하면 현재 위치와 스크립트 위치에서 자동 탐색한다.",
    )
    parser.add_argument(
        "--coverage",
        default=None,
        help="general_coverage.md 경로. 생략하면 reports/general_coverage.md를 사용한다.",
    )
    parser.add_argument(
        "--accepted-dir",
        default=None,
        help="accepted 디렉터리 경로. 생략하면 accepted를 사용한다.",
    )
    parser.add_argument(
        "--only-duplicates",
        action="store_true",
        help="duplicate_count가 1 이상인 leaf만 출력한다.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="duplicate group별 command/thinking/dialog/ref 목록까지 출력한다.",
    )
    parser.add_argument(
        "--command-width",
        type=int,
        default=80,
        help="표 출력에서 command를 자를 길이. 0이면 자르지 않는다.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="상세 결과를 JSON 파일로 저장할 경로.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        dataset_root = detect_dataset_root(args.dataset_root)
        coverage_path = (
            Path(args.coverage).resolve()
            if args.coverage
            else dataset_root / "reports" / "general_coverage.md"
        )
        accepted_dir = (
            Path(args.accepted_dir).resolve()
            if args.accepted_dir
            else dataset_root / "accepted"
        )

        if not coverage_path.exists():
            raise FileNotFoundError(f"coverage 파일이 없다: {coverage_path}")
        if not accepted_dir.is_dir():
            raise FileNotFoundError(f"accepted 디렉터리가 없다: {accepted_dir}")

        leaves = parse_general_coverage(coverage_path)
        sample_lookup = build_sample_lookup(accepted_dir)
        stats = compute_leaf_stats(leaves, sample_lookup)

        print(f"dataset_root: {dataset_root}")
        print(f"coverage: {coverage_path}")
        print(f"accepted_dir: {accepted_dir}")
        print(
            "duplicate_key: command_spec.command_text/input.command "
            "+ output.thinking + output.dialog"
        )
        print()

        print_summary(
            stats,
            only_duplicates=args.only_duplicates,
            command_width=args.command_width,
            details=args.details,
        )

        if args.json_output:
            output_path = Path(args.json_output).resolve()
            write_json_report(output_path, stats)
            print(f"json_output: {output_path}")

        return 0

    except Exception as error:
        print(f"duplicate report failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
