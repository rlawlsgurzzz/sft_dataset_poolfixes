# coverage markdown의 numeric path로 command slot pool을 점검한다.
# A_XX/E_XX와 일부 한국어 조사만 다른 command_text는 같은 family로 취급한다.
# command_style은 distinct paraphrase 판정에 사용하지 않는다.
# 출력 파일에는 payload용 existing_valid_paraphrase_samples와 family 요약을 기록한다.

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sft_coverage_report import (
        DEFAULT_DATASET_ROOT,
        get_report_sample_ref,
        load_accepted_samples,
    )
    from sft_generation_request import (
        SPLIT_EXPRESSION_POOL_LIMITS,
        VALID_SPLITS,
        sample_command_style,
        sample_command_text,
        sample_split,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_coverage_report import (
        DEFAULT_DATASET_ROOT,
        get_report_sample_ref,
        load_accepted_samples,
    )
    from sft_generation_request import (
        SPLIT_EXPRESSION_POOL_LIMITS,
        VALID_SPLITS,
        sample_command_style,
        sample_command_text,
        sample_split,
    )


UNIT_ID_RE = re.compile(r"(?<![A-Za-z0-9_])([AE])_\d{2}(?![A-Za-z0-9_])")
COVERAGE_ROW_RE = re.compile(
    r'^\s*'
    r'(?P<numeric_path>c?\d+(?:-\d+){5}(?:/\d+(?:-\d+){2})?)'
    r'\s+@\s+'
    r'"(?P<base_command_text>(?:\\.|[^"\\])*)"'
    r'\s+@\s+'
    r'(?P<count>\d+)'
    r'\s+@\s+'
    r'(?P<edge_flags>.*?)'
    r'\s+@\s+'
    r'(?P<refs>.*)'
    r'\s*$'
)


@dataclass(frozen=True)
class ParsedRequest:
    raw_request: str
    numeric_path: str
    count_to_generate: int


@dataclass(frozen=True)
class CoverageRow:
    source_report: Path
    numeric_path: str
    base_command_text: str
    sample_count: int
    edge_flags_text: str
    report_refs: tuple[str, ...]


@dataclass
class FamilyBucket:
    family_key: str
    selected_command_text: str
    selected_command_style: str
    report_refs: list[str]
    command_texts: list[str]


def normalize_numeric_path(value: str) -> str:
    value = value.strip()
    if value.startswith("c"):
        value = value[1:]
    return value


def parse_request(value: str) -> ParsedRequest:
    raw_request = value.strip()
    if not raw_request:
        raise ValueError("request is empty")

    if "." not in raw_request:
        raise ValueError(
            "request must end with .<count>, for example: 1-1-1-1-1-1.10"
        )

    path_text, count_text = raw_request.rsplit(".", 1)
    if not path_text or not count_text.isdigit():
        raise ValueError(
            "request must end with .<count>, for example: 1-1-1-1-1-1.10"
        )

    count_to_generate = int(count_text)
    if count_to_generate <= 0:
        raise ValueError("count must be greater than zero")

    return ParsedRequest(
        raw_request=raw_request,
        numeric_path=normalize_numeric_path(path_text),
        count_to_generate=count_to_generate,
    )


def normalize_unit_particles(text: str) -> str:
    # A_ID/E_ID 뒤의 은/는/이/가 차이는 같은 family로 취급한다.
    text = re.sub(
        r"(?P<unit>[AE]_ID)\s*(?:은|는|이|가)(?=$|[\s,.;!?])",
        r"\g<unit>{JOSA_ST}",
        text,
    )

    # 기존 예시의 E_ID을/E_ID를 차이도 같은 family로 취급한다.
    text = re.sub(
        r"(?P<unit>[AE]_ID)\s*(?:을|를)(?=$|[\s,.;!?])",
        r"\g<unit>{JOSA_OBJ}",
        text,
    )

    return text


def normalize_command_family(command_text: str) -> str:
    text = command_text.strip()
    text = UNIT_ID_RE.sub(lambda match: f"{match.group(1)}_ID", text)

    text = normalize_unit_particles(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+([.!?])", r"\1", text)

    return text.strip()


def parse_report_refs(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value or value == "{}":
        return tuple()

    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_coverage_row(line: str, source_report: Path) -> CoverageRow | None:
    match = COVERAGE_ROW_RE.match(line)
    if match is None:
        return None

    return CoverageRow(
        source_report=source_report,
        numeric_path=normalize_numeric_path(match.group("numeric_path")),
        base_command_text=match.group("base_command_text"),
        sample_count=int(match.group("count")),
        edge_flags_text=match.group("edge_flags").strip(),
        report_refs=parse_report_refs(match.group("refs")),
    )


def read_coverage_rows(report_path: Path) -> list[CoverageRow]:
    if not report_path.exists():
        return []

    rows: list[CoverageRow] = []
    for line in report_path.read_text(encoding="utf-8").splitlines():
        row = parse_coverage_row(line, report_path)
        if row is not None:
            rows.append(row)

    return rows


def find_coverage_row(dataset_root: Path, numeric_path: str) -> CoverageRow:
    reports_dir = dataset_root / "reports"
    candidate_reports = (
        [reports_dir / "skill_coverage.md", reports_dir / "general_coverage.md"]
        if "/" in numeric_path
        else [reports_dir / "general_coverage.md", reports_dir / "skill_coverage.md"]
    )

    matches: list[CoverageRow] = []
    for report_path in candidate_reports:
        for row in read_coverage_rows(report_path):
            if row.numeric_path == numeric_path:
                matches.append(row)

    if not matches:
        searched = ", ".join(path.as_posix() for path in candidate_reports)
        raise ValueError(
            f"coverage row not found for path '{numeric_path}'. searched: {searched}"
        )

    if len(matches) > 1:
        match_sources = ", ".join(match.source_report.as_posix() for match in matches)
        raise ValueError(
            f"multiple coverage rows found for path '{numeric_path}': {match_sources}"
        )

    return matches[0]


def build_sample_ref_index(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    for sample in samples:
        ref = get_report_sample_ref(sample)
        if ref in index:
            raise ValueError(f"duplicate report sample ref found: {ref}")
        index[ref] = sample

    return index


def collect_slot_samples(
    *,
    dataset_root: Path,
    row: CoverageRow,
) -> list[tuple[str, dict[str, Any]]]:
    samples = load_accepted_samples(dataset_root / "accepted")
    sample_index = build_sample_ref_index(samples)

    result: list[tuple[str, dict[str, Any]]] = []
    missing_refs: list[str] = []

    for ref in row.report_refs:
        sample = sample_index.get(ref)
        if sample is None:
            missing_refs.append(ref)
            continue
        result.append((ref, sample))

    if missing_refs:
        raise ValueError(
            "coverage report refs were not found in accepted samples: "
            + ", ".join(missing_refs)
        )

    return result


def build_family_buckets(
    *,
    slot_samples: list[tuple[str, dict[str, Any]]],
    target_split: str,
) -> OrderedDict[str, FamilyBucket]:
    buckets: OrderedDict[str, FamilyBucket] = OrderedDict()

    for report_ref, sample in slot_samples:
        if sample_split(sample) != target_split:
            continue

        command_text = sample_command_text(sample)
        command_style = sample_command_style(sample)
        family_key = normalize_command_family(command_text)

        bucket = buckets.get(family_key)
        if bucket is None:
            bucket = FamilyBucket(
                family_key=family_key,
                selected_command_text=command_text,
                selected_command_style=command_style,
                report_refs=[],
                command_texts=[],
            )
            buckets[family_key] = bucket

        bucket.report_refs.append(report_ref)
        if command_text not in bucket.command_texts:
            bucket.command_texts.append(command_text)

    return buckets


def build_existing_valid_paraphrase_samples(
    *,
    family_buckets: OrderedDict[str, FamilyBucket],
    pool_limit: int,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []

    for bucket in family_buckets.values():
        result.append(
            {
                "command_text": bucket.selected_command_text,
                "command_style": bucket.selected_command_style,
            }
        )

        if len(result) >= pool_limit:
            break

    return result


def build_command_text_policy_preview(
    *,
    target_split: str,
    existing_count: int,
    other_split_reserved_count: int,
    count_to_generate: int,
    pool_limit: int,
) -> dict[str, Any]:
    new_unique_count = max(0, min(count_to_generate, pool_limit - existing_count))
    cycle_count = max(0, count_to_generate - new_unique_count)

    return {
        "target_split": target_split,
        "same_split_expression_pool_size": pool_limit,
        "existing_same_split_expression_count": existing_count,
        "other_split_reserved_expression_count": other_split_reserved_count,
        "new_unique_command_texts_to_create": new_unique_count,
        "samples_using_same_split_cycle": cycle_count,
        "dedupe_rule_for_this_inspection": "command_text with A_XX/E_XX normalized to A_ID/E_ID; command_style ignored",
    }


def count_other_split_reserved_families(
    *,
    slot_samples: list[tuple[str, dict[str, Any]]],
    target_split: str,
) -> int:
    families: set[str] = set()

    for _, sample in slot_samples:
        if sample_split(sample) == target_split:
            continue
        families.add(normalize_command_family(sample_command_text(sample)))

    return len(families)


def render_output(
    *,
    parsed_request: ParsedRequest,
    target_split: str,
    row: CoverageRow,
    slot_samples: list[tuple[str, dict[str, Any]]],
    family_buckets: OrderedDict[str, FamilyBucket],
    pool_entries: list[dict[str, str]],
    pool_limit: int,
) -> str:
    same_split_sample_count = sum(
        1 for _, sample in slot_samples if sample_split(sample) == target_split
    )
    other_split_reserved_family_count = count_other_split_reserved_families(
        slot_samples=slot_samples,
        target_split=target_split,
    )
    command_text_policy_preview = build_command_text_policy_preview(
        target_split=target_split,
        existing_count=len(pool_entries),
        other_split_reserved_count=other_split_reserved_family_count,
        count_to_generate=parsed_request.count_to_generate,
        pool_limit=pool_limit,
    )

    payload_like = {
        "request": {
            "raw_request": parsed_request.raw_request,
            "count_to_generate": parsed_request.count_to_generate,
            "display_path": parsed_request.numeric_path,
            "base_command_text": row.base_command_text,
        },
        "target_split": target_split,
        "existing_valid_paraphrase_samples": pool_entries,
        "command_text_policy": command_text_policy_preview,
    }

    lines: list[str] = []
    lines.append("# Command Slot Pool Inspection")
    lines.append("")
    lines.append(f"- request: `{parsed_request.raw_request}`")
    lines.append(f"- normalized_path: `{parsed_request.numeric_path}`")
    lines.append(f"- target_split: `{target_split}`")
    lines.append(f"- coverage_source: `{row.source_report.as_posix()}`")
    lines.append(f"- base_command_text: `{row.base_command_text}`")
    lines.append(f"- coverage_row_sample_count: `{row.sample_count}`")
    lines.append(f"- same_split_sample_count: `{same_split_sample_count}`")
    lines.append(f"- distinct_family_count_in_same_split: `{len(family_buckets)}`")
    lines.append(f"- pool_limit: `{pool_limit}`")
    lines.append(f"- pool_entry_count: `{len(pool_entries)}`")
    lines.append("")
    lines.append("## Payload-like pool")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload_like, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append(
        "| family_index | selected_for_pool | family_sample_count | family_key | selected_command_text | selected_command_style | command_texts_in_family |"
    )
    lines.append("|---:|---|---:|---|---|---|---|")

    selected_family_keys = {
        normalize_command_family(entry["command_text"]) for entry in pool_entries
    }

    for index, bucket in enumerate(family_buckets.values(), start=1):
        selected_for_pool = "yes" if bucket.family_key in selected_family_keys else "no"
        command_texts = "<br>".join(bucket.command_texts)
        lines.append(
            "| "
            f"{index} | "
            f"{selected_for_pool} | "
            f"{len(bucket.report_refs)} | "
            f"`{bucket.family_key}` | "
            f"`{bucket.selected_command_text}` | "
            f"`{bucket.selected_command_style}` | "
            f"{command_texts} |"
        )

    lines.append("")
    lines.append("## Pool entry family counts")
    lines.append("")

    for index, entry in enumerate(pool_entries, start=1):
        family_key = normalize_command_family(entry["command_text"])
        bucket = family_buckets[family_key]
        lines.append(
            f"{index}. `{entry['command_text']}` / `{entry['command_style']}`"
        )
        lines.append(f"   - family_key: `{family_key}`")
        lines.append(f"   - family_sample_count: `{len(bucket.report_refs)}`")
        lines.append(f"   - family_command_texts:")
        for command_text in bucket.command_texts:
            lines.append(f"     - {command_text}")

    lines.append("")

    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect payload command_text pool for one coverage command slot. "
            "A_XX/E_XX-only variants are treated as the same family."
        )
    )
    parser.add_argument(
        "request",
        help="Numeric request with count. Example: 1-1-1-1-1-1.10 or c1-1-1-1-1-1.10",
    )
    parser.add_argument(
        "output",
        help="Output txt/md file path. Example: reports/command_slot_pool_train.txt",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Repository dataset root. Default: repository root inferred from script path.",
    )
    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="train",
        help="Target split used for pool construction.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    parsed_request = parse_request(args.request)
    target_split = args.split
    pool_limit = SPLIT_EXPRESSION_POOL_LIMITS[target_split]

    row = find_coverage_row(
        dataset_root=dataset_root,
        numeric_path=parsed_request.numeric_path,
    )
    slot_samples = collect_slot_samples(
        dataset_root=dataset_root,
        row=row,
    )
    family_buckets = build_family_buckets(
        slot_samples=slot_samples,
        target_split=target_split,
    )
    pool_entries = build_existing_valid_paraphrase_samples(
        family_buckets=family_buckets,
        pool_limit=pool_limit,
    )

    output_text = render_output(
        parsed_request=parsed_request,
        target_split=target_split,
        row=row,
        slot_samples=slot_samples,
        family_buckets=family_buckets,
        pool_entries=pool_entries,
        pool_limit=pool_limit,
    )

    output_path = Path(args.output)
    write_output(output_path, output_text)

    print(f"written: {output_path}")
    print(f"target_split: {target_split}")
    print(f"normalized_path: {parsed_request.numeric_path}")
    print(f"pool_entries: {len(pool_entries)}/{pool_limit}")
    print(f"distinct_families: {len(family_buckets)}")


if __name__ == "__main__":
    main()