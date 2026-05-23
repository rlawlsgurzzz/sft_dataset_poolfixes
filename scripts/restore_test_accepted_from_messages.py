# Restores accepted test samples that were converted into SFT messages JSONL.
# Matches each messages row against accepted split=test samples by runtime user payload and assistant output.
# Writes a new JSONL preserving the full accepted sample fields, including metadata, command_spec, gold, and validator_result.
# Fails if any messages row has no exact accepted-sample match.
# Does not generate, evaluate, shuffle, or modify sample contents.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTED_DIR = DEFAULT_DATASET_ROOT / "accepted"
DEFAULT_SPLIT = "test"

OUTPUT_SCHEMA_EXAMPLE: dict[str, Any] = {
    "thinking": "현재 전장 상태와 명령 의미를 근거로 실행 가능한 행동만 선택한다.",
    "dialog": [
        {
            "unitId": "A_01",
            "text": "내가 맡아서 처리한다.",
        }
    ],
    "action": [
        {
            "unitId": "A_01",
            "sequence": [
                {
                    "type": "move",
                    "subtype": "approachOpponent",
                    "movementType": "direct",
                    "to": "E_01",
                },
                {
                    "type": "attack",
                    "target": "E_01",
                },
            ],
        }
    ],
}

HARD_CONSTRAINTS = [
    "JSON object 하나만 반환한다.",
    "최상위 key는 thinking, dialog, action 세 개만 사용한다.",
    "action.unitId는 commandAnalysis.allowedActors 안에서만 고른다.",
    "move.to는 그 어떤 경우에서도 액터 자신의 unitId가 될 수 없다.",
    "attack.target은 commandAnalysis.allowedAttackTargets 안에서만 고른다.",
    "skill target은 skill action에 한해서 일반 attack target보다 넓게 선택할 수 있다.",
    "skill target은 입력에 존재하는 unitId 중에서 고른다. 단, canBeTargeted가 false인 유닛은 skill target으로 사용하지 않는다.",
    "IsSkillOnSelf가 true일 때만 유일하게, actor 본인의 unitId를 skill target으로 사용한다.",
    "IsSkillOnOtherAlly가 true이면 actor 자신이 아닌 아군 unitId를 skill target으로 사용한다.",
    "IsSkillOnSelf와 IsSkillOnOtherAlly가 모두 false이면 적 unitId를 skill target으로 사용한다.",
    "canSkillTargetDead가 true이고 죽은 아군 대상 스킬이 명령 의미에 맞으면 commandAnalysis.deadAllies에서 target을 고른다.",
    "canSkillTargetDead가 false이면 죽은 유닛을 skill target으로 선택하지 않는다.",
    "move.to는 commandAnalysis.validMoveToUnits 안에서만 고른다.",
    "move.subtype은 approachOpponent, escape, help, holdFront 중 하나만 사용한다.",
    "move.movementType은 direct 또는 flank만 사용한다.",
    "dialog는 action actor당 하나만 출력한다.",
    "thinking과 dialog.text는 짧은 한국어로 쓴다.",
    "attack 또는 skill 뒤에는 wait을 붙이지 않는다.",
    "actor에게 skillDescription이 있고 명령에 스킬 지연 또는 스킬 금지 의도가 명시되면 skillControl을 반드시 사용한다.",
    "skillControl은 skillDescription이 있는 actor에게만 사용한다.",
    "미래 조건부 action은 만들지 않는다.",
]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"JSON array file must contain objects: {path}")
        return [item for item in data if isinstance(item, dict)]

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(item, dict):
            raise ValueError(f"JSONL row must be object at {path}:{line_number}")
        records.append(item)
    return records


def iter_jsonl_files(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Accepted path does not exist: {path}")
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    yield from sorted(path.glob(pattern))


def iter_accepted_samples(accepted_path: Path, recursive: bool) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in iter_jsonl_files(accepted_path, recursive=recursive):
        for row_index, sample in enumerate(read_json_records(path), start=1):
            yield path, row_index, sample


def require_mapping(value: Any, field_name: str, sample_ref: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{sample_ref} must contain object field {field_name}")
    return value


def accepted_runtime_user_object(sample: dict[str, Any], sample_ref: str) -> dict[str, Any]:
    source_input = require_mapping(sample.get("input"), "input", sample_ref)
    runtime_input = require_mapping(source_input.get("input"), "input.input", sample_ref)
    command_analysis = require_mapping(source_input.get("commandAnalysis"), "input.commandAnalysis", sample_ref)
    return {
        "input": runtime_input,
        "commandAnalysis": command_analysis,
        "output_schema_example": OUTPUT_SCHEMA_EXAMPLE,
        "hard_constraints": HARD_CONSTRAINTS,
    }


def accepted_key(sample: dict[str, Any], sample_ref: str) -> tuple[str, str]:
    user_object = accepted_runtime_user_object(sample, sample_ref)
    output = require_mapping(sample.get("output"), "output", sample_ref)
    return canonical_json(user_object), canonical_json(output)


def parse_messages_record(record: dict[str, Any], row_index: int) -> tuple[tuple[str, str], str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"messages row {row_index}: missing messages list")

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"messages row {row_index}: expected roles ['system','user','assistant'], got {roles}")

    user_content = messages[1].get("content") if isinstance(messages[1], dict) else None
    assistant_content = messages[2].get("content") if isinstance(messages[2], dict) else None
    if not isinstance(user_content, str) or not user_content.strip():
        raise ValueError(f"messages row {row_index}: empty user content")
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        raise ValueError(f"messages row {row_index}: empty assistant content")

    try:
        user_object = json.loads(user_content)
    except json.JSONDecodeError as error:
        raise ValueError(f"messages row {row_index}: user content is not valid JSON") from error
    try:
        output_object = json.loads(assistant_content)
    except json.JSONDecodeError as error:
        raise ValueError(f"messages row {row_index}: assistant content is not valid JSON") from error

    if not isinstance(user_object, dict):
        raise ValueError(f"messages row {row_index}: user content JSON must be object")
    if not isinstance(output_object, dict):
        raise ValueError(f"messages row {row_index}: assistant content JSON must be object")

    command = None
    runtime_input = user_object.get("input")
    if isinstance(runtime_input, dict):
        command = runtime_input.get("command")
    command_label = command if isinstance(command, str) else ""

    return (canonical_json(user_object), canonical_json(output_object)), command_label


def build_accepted_index(
    accepted_path: Path,
    split: str,
    recursive: bool,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[tuple[str, str], list[str]], int]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    refs: dict[tuple[str, str], list[str]] = defaultdict(list)
    total = 0

    for source_path, row_index, sample in iter_accepted_samples(accepted_path, recursive=recursive):
        if sample.get("split") != split:
            continue
        sample_id = sample.get("id")
        sample_ref = f"{source_path}:{sample_id if isinstance(sample_id, str) else row_index}"
        key = accepted_key(sample, sample_ref)
        index[key].append(sample)
        refs[key].append(sample_ref)
        total += 1

    if total == 0:
        raise ValueError(f"No accepted samples found for split={split} under {accepted_path}")
    return index, refs, total


def restore_accepted_samples_from_messages(
    messages_path: Path,
    accepted_path: Path,
    output_path: Path,
    split: str,
    recursive: bool,
    allow_reuse: bool,
) -> dict[str, Any]:
    message_records = read_json_records(messages_path)
    if not message_records:
        raise ValueError(f"No messages records found: {messages_path}")

    accepted_index, accepted_refs, accepted_total = build_accepted_index(
        accepted_path=accepted_path,
        split=split,
        recursive=recursive,
    )

    restored: list[dict[str, Any]] = []
    matched_refs: list[str] = []
    missing: list[dict[str, Any]] = []
    used_key_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row_index, record in enumerate(message_records, start=1):
        key, command_label = parse_messages_record(record, row_index)
        candidates = accepted_index.get(key, [])
        if not candidates:
            missing.append(
                {
                    "row_index": row_index,
                    "command": command_label,
                }
            )
            continue

        if allow_reuse:
            candidate_index = 0
        else:
            candidate_index = used_key_counts[key]
            if candidate_index >= len(candidates):
                missing.append(
                    {
                        "row_index": row_index,
                        "command": command_label,
                        "reason": "all exact-match accepted candidates already used",
                    }
                )
                continue

        restored.append(candidates[candidate_index])
        matched_refs.append(accepted_refs[key][candidate_index])
        used_key_counts[key] += 1

    if missing:
        preview = json.dumps(missing[:10], ensure_ascii=False, indent=2)
        raise ValueError(
            f"Failed to restore {len(missing)} messages rows from accepted split={split} samples.\n"
            f"First missing rows:\n{preview}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as dst:
        for sample in restored:
            dst.write(compact_json(sample) + "\n")

    duplicate_match_key_count = sum(1 for count in used_key_counts.values() if count > 1)
    report = {
        "messages_path": str(messages_path),
        "accepted_path": str(accepted_path),
        "output_path": str(output_path),
        "split": split,
        "messages_count": len(message_records),
        "accepted_split_sample_count": accepted_total,
        "restored_count": len(restored),
        "allow_reuse": allow_reuse,
        "unique_match_keys_used": len(used_key_counts),
        "duplicate_match_key_count": duplicate_match_key_count,
        "matched_source_refs": matched_refs,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".restore_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore full accepted test samples from a shuffled SFT messages JSONL by exact matching "
            "runtime user payload and assistant output."
        )
    )
    parser.add_argument("messages_path", type=Path, help="Shuffled test_sft_messages.jsonl.")
    parser.add_argument("accepted_path", type=Path, help="Accepted directory or accepted JSONL file.")
    parser.add_argument("output_path", type=Path, help="Output restored accepted JSONL path.")
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Accepted split to search. Default: {DEFAULT_SPLIT}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search accepted_path recursively when it is a directory.",
    )
    parser.add_argument(
        "--consume-duplicates",
        action="store_true",
        help=(
            "Use separate accepted candidates for duplicate identical messages. "
            "Default allows reusing the first exact-match accepted sample."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = restore_accepted_samples_from_messages(
        messages_path=args.messages_path,
        accepted_path=args.accepted_path,
        output_path=args.output_path,
        split=args.split,
        recursive=args.recursive,
        allow_reuse=not args.consume_duplicates,
    )

    print(f"[done] restored {report['restored_count']} accepted samples")
    print(f"[done] wrote {report['output_path']}")
    print(f"[done] wrote {report['output_path']}.restore_report.json")
    print(f"[done] accepted split sample count searched: {report['accepted_split_sample_count']}")


if __name__ == "__main__":
    main()
