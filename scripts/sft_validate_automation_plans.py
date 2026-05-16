# generation_automation 폴더의 자동화 plan 입력을 검증한다.
# 기본 실행은 auto_generation_plan_*.txt 전체를 검사한다.
# --plan 0001처럼 지정하면 auto_generation_plan_0001.txt 하나만 검사한다.
# 각 plan line은 "<split> <generation_request>" 형식이다.
# payload 생성, mix 구성, teacher 요청은 수행하지 않는다.

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sft_taxonomy import (
        DEFAULT_TAXONOMY_PATH,
        ParsedGenerationRequest,
        conflict_type_to_key,
        get_general_intent_family_matrix,
        get_order,
        load_taxonomy,
        parse_generation_request,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_taxonomy import (
        DEFAULT_TAXONOMY_PATH,
        ParsedGenerationRequest,
        conflict_type_to_key,
        get_general_intent_family_matrix,
        get_order,
        load_taxonomy,
        parse_generation_request,
    )


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTOMATION_DIR = DEFAULT_DATASET_ROOT / "generation_automation"
VALID_SPLITS = {"train", "validation", "test"}
PLAN_FILE_PATTERN = re.compile(r"^auto_generation_plan_(\d{4})\.txt$")


@dataclass(frozen=True)
class PlanEntry:
    plan_file: Path
    line_number: int
    split: str
    request: str


@dataclass(frozen=True)
class InvalidEntry:
    plan_file: Path
    line_number: int
    split: str
    raw_request: str
    numeric_request: str | None
    stable_request: str | None
    reason: str


def strip_inline_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_plan_line(plan_file: Path, line: str, line_number: int) -> PlanEntry | None:
    cleaned = strip_inline_comment(line)
    if not cleaned:
        return None

    parts = cleaned.split()
    if len(parts) != 2:
        raise ValueError("expected '<split> <generation_request>'")

    split, request = parts
    if split not in VALID_SPLITS:
        raise ValueError("split must be one of train, validation, test")

    return PlanEntry(
        plan_file=plan_file,
        line_number=line_number,
        split=split,
        request=request,
    )


def resolve_plan_path(plan: str, automation_dir: Path) -> Path:
    value = plan.strip()
    if not value:
        raise ValueError("empty plan value")

    if value.isdigit():
        return automation_dir / f"auto_generation_plan_{int(value):04d}.txt"

    path = Path(value)
    if path.exists():
        return path

    candidate = automation_dir / value
    if candidate.exists():
        return candidate

    if PLAN_FILE_PATTERN.match(value):
        return candidate

    raise FileNotFoundError(f"plan file not found: {plan}")


def discover_plan_files(automation_dir: Path, plan: str = "") -> list[Path]:
    if plan:
        plan_file = resolve_plan_path(plan, automation_dir)
        if not plan_file.exists():
            raise FileNotFoundError(f"plan file not found: {plan_file}")
        return [plan_file]

    plan_files = sorted(automation_dir.glob("auto_generation_plan_*.txt"))
    if not plan_files:
        raise FileNotFoundError(
            f"no auto_generation_plan_*.txt files found in {automation_dir}"
        )

    return plan_files


def read_plan_entries(plan_files: list[Path]) -> list[PlanEntry]:
    entries: list[PlanEntry] = []

    for plan_file in plan_files:
        lines = plan_file.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            try:
                entry = parse_plan_line(plan_file, line, line_number)
            except Exception as error:
                entries.append(
                    PlanEntry(
                        plan_file=plan_file,
                        line_number=line_number,
                        split="<invalid>",
                        request=f"<line-parse-error: {error}> {line.strip()}",
                    )
                )
                continue

            if entry is not None:
                entries.append(entry)

    return entries


def index_1(values: list[str], value: str, label: str) -> int:
    try:
        return values.index(value) + 1
    except ValueError as error:
        raise ValueError(f"{label} not found in taxonomy order: {value}") from error


def scenario_index(
    taxonomy: dict[str, Any],
    intent_family: str,
    scenario_family: str,
) -> int:
    matrix = get_general_intent_family_matrix(taxonomy, intent_family)
    scenario_map = matrix.get("allowed_scenario_family", {})
    if not isinstance(scenario_map, dict):
        raise ValueError(f"allowed_scenario_family must be object: {intent_family}")

    scenarios = list(scenario_map.keys())
    return index_1(scenarios, scenario_family, "scenario_family")


def to_numeric_request(parsed: ParsedGenerationRequest, taxonomy: dict[str, Any]) -> str:
    general = parsed.general_path

    numeric_parts = [
        index_1(get_order(taxonomy, "intent_family"), general.intent_family, "intent_family"),
        index_1(get_order(taxonomy, "actor_selection"), general.actor_selection, "actor_selection"),
        index_1(get_order(taxonomy, "target_selection"), general.target_selection, "target_selection"),
        index_1(get_order(taxonomy, "action_pattern"), general.action_pattern, "action_pattern"),
        scenario_index(taxonomy, general.intent_family, general.scenario_family),
        general.command_slot_index,
    ]

    request = "c" + "-".join(str(part) for part in numeric_parts)

    if parsed.skill_path is not None:
        skill = parsed.skill_path
        skill_parts = [
            index_1(get_order(taxonomy, "skill_family"), skill.skill_family, "skill_family"),
            index_1(get_order(taxonomy, "skill_target_kind"), skill.skill_target_kind, "skill_target_kind"),
            index_1(
                get_order(taxonomy, "conflict_type"),
                conflict_type_to_key(skill.conflict_type),
                "conflict_type",
            ),
        ]
        request += "/" + "-".join(str(part) for part in skill_parts)

    return f"{request}.{parsed.count}"


def to_stable_request(parsed: ParsedGenerationRequest) -> str:
    general = parsed.general_path
    request = f"{general.stable_path}#{general.command_slot_index}"

    if parsed.skill_path is not None:
        request += f"/{parsed.skill_path.stable_path}"

    return f"{request}.{parsed.count}"


def validate_entry(
    entry: PlanEntry,
    taxonomy: dict[str, Any],
) -> InvalidEntry | None:
    if entry.split not in VALID_SPLITS:
        return InvalidEntry(
            plan_file=entry.plan_file,
            line_number=entry.line_number,
            split=entry.split,
            raw_request=entry.request,
            numeric_request=None,
            stable_request=None,
            reason="invalid plan line",
        )

    try:
        parsed = parse_generation_request(entry.request, taxonomy)
        numeric_request = to_numeric_request(parsed, taxonomy)
        stable_request = to_stable_request(parsed)
    except Exception as error:
        return InvalidEntry(
            plan_file=entry.plan_file,
            line_number=entry.line_number,
            split=entry.split,
            raw_request=entry.request,
            numeric_request=None,
            stable_request=None,
            reason=f"request parse failed: {error}",
        )

    if parsed.count <= 0:
        return InvalidEntry(
            plan_file=entry.plan_file,
            line_number=entry.line_number,
            split=entry.split,
            raw_request=entry.request,
            numeric_request=numeric_request,
            stable_request=stable_request,
            reason="count must be positive",
        )

    return None


def print_invalid(invalid: InvalidEntry, automation_dir: Path) -> None:
    try:
        plan_name = invalid.plan_file.resolve().relative_to(
            automation_dir.resolve()
        ).as_posix()
    except ValueError:
        plan_name = invalid.plan_file.as_posix()

    print(
        f"INVALID {plan_name}:{invalid.line_number} "
        f"split={invalid.split} reason={invalid.reason}"
    )

    if invalid.numeric_request is not None:
        print(f"  numeric: {invalid.split} {invalid.numeric_request}")
    else:
        print("  numeric: <unavailable>")

    if invalid.stable_request is not None:
        print(f"  stable:  {invalid.split} {invalid.stable_request}")
    else:
        print("  stable:  <unavailable>")


def validate_plans(
    *,
    dataset_root: Path,
    taxonomy_path: Path,
    automation_dir: Path,
    plan: str = "",
) -> int:
    _ = dataset_root

    taxonomy = load_taxonomy(taxonomy_path)
    plan_files = discover_plan_files(automation_dir, plan)
    entries = read_plan_entries(plan_files)

    invalid_entries: list[InvalidEntry] = []
    for entry in entries:
        invalid = validate_entry(
            entry=entry,
            taxonomy=taxonomy,
        )
        if invalid is not None:
            invalid_entries.append(invalid)

    if invalid_entries:
        print(f"INVALID_COUNT: {len(invalid_entries)}")
        for invalid in invalid_entries:
            print_invalid(invalid, automation_dir)
        return 1

    if plan:
        print(f"OK: {plan_files[0].name} ({len(entries)} entries)")
    else:
        print(f"OK: {len(plan_files)} plan files, {len(entries)} entries")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate automation generation plan files."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--automation-dir", default=str(DEFAULT_AUTOMATION_DIR))
    parser.add_argument(
        "--plan",
        default="",
        help="Optional plan number or file name. Example: 0001, auto_generation_plan_0001.txt",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raise SystemExit(
        validate_plans(
            dataset_root=Path(args.dataset_root),
            taxonomy_path=Path(args.taxonomy),
            automation_dir=Path(args.automation_dir),
            plan=args.plan,
        )
    )


if __name__ == "__main__":
    main()