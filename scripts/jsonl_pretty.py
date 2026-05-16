#!/usr/bin/env python3
"""Pretty-print repository JSON/JSONL samples to stdout.

Usage examples:
  python scripts/jsonl_pretty.py accepted_20260512_005252
  python scripts/jsonl_pretty.py accepted/accepted_20260512_005252.jsonl
  python scripts/jsonl_pretty.py seed_master_0001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DATA_DIRS = ("accepted", "rejected", "raw_generations")
JSON_EXTENSIONS = (".jsonl", ".json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: list[dict[str, Any]] = []

    if text.startswith(("[", "{")):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and isinstance(data.get("samples"), list):
            return [item for item in data["samples"] if isinstance(item, dict)]
        if isinstance(data, dict):
            return [data]

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if isinstance(item, dict):
            records.append(item)
        else:
            raise ValueError(f"JSONL line must be an object: {path}:{line_number}")

    return records


def iter_data_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirname in DATA_DIRS:
        directory = root / dirname
        if not directory.is_dir():
            continue
        for extension in JSON_EXTENSIONS:
            paths.extend(directory.glob(f"*{extension}"))
    return sorted(paths)


def resolve_files(root: Path, query: str) -> list[Path]:
    direct = Path(query)
    candidates: list[Path] = []

    if direct.is_absolute() and direct.is_file():
        return [direct]

    relative = root / direct
    if relative.is_file():
        return [relative]

    data_files = iter_data_files(root)
    query_path_name = direct.name

    for path in data_files:
        if path.name == query or path.name == query_path_name:
            candidates.append(path)

    if not candidates:
        for path in data_files:
            if path.stem == query or path.name.startswith(query):
                candidates.append(path)

    if not candidates:
        for path in data_files:
            if query in path.name:
                candidates.append(path)

    return sorted(dict.fromkeys(candidates))


def find_sample_by_id(root: Path, sample_id: str) -> tuple[Path, dict[str, Any]] | None:
    raw_dir = root / "raw_generations"
    raw_files = []
    if raw_dir.is_dir():
        for extension in JSON_EXTENSIONS:
            raw_files.extend(raw_dir.glob(f"*{extension}"))

    for path in sorted(raw_files):
        for record in read_json_records(path):
            if record.get("id") == sample_id:
                return path, record

    return None


def print_records(path: Path, records: list[dict[str, Any]], root: Path) -> None:
    rel_path = path.relative_to(root) if path.is_relative_to(root) else path
    print(f"# {rel_path}")
    print(f"# records: {len(records)}")
    for index, record in enumerate(records, start=1):
        print(f"\n# --- record {index}/{len(records)} id={record.get('id', '<missing>')} ---")
        print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pretty-print accepted/rejected/raw_generation JSONL files or one raw sample id."
    )
    parser.add_argument(
        "query",
        help="File path/name/prefix, or a sample id like seed_master_0001. Sample ids are searched in raw_generations first.",
    )
    args = parser.parse_args()

    root = repo_root()
    query = args.query.strip()

    file_paths = resolve_files(root, query)
    if file_paths:
        for file_index, file_path in enumerate(file_paths, start=1):
            if len(file_paths) > 1:
                print(f"# === file {file_index}/{len(file_paths)} ===")
            print_records(file_path, read_json_records(file_path), root)
            if file_index != len(file_paths):
                print("\n")
        return 0

    if query.startswith("seed_master_") or query.startswith("sample_"):
        found = find_sample_by_id(root, query)
        if found is not None:
            path, record = found
            print_records(path, [record], root)
            return 0

    print(f"No JSON/JSONL file or raw generation sample id matched: {query}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
