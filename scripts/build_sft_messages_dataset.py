# accepted JSONL 샘플 전체를 섞어서 Unsloth/TRL SFT용 messages JSONL을 생성한다.
# 학습 입력에는 student runtime에서 실제로 쓰는 input/commandAnalysis만 포함하고,
# assistant label에는 accepted sample의 output만 compact JSON으로 저장한다.

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTED_DIR = DEFAULT_DATASET_ROOT / "accepted"
DEFAULT_OUTPUT_PATH = DEFAULT_DATASET_ROOT / "datasets" / "train_sft_messages.jsonl"
DEFAULT_TEACHER_CLIENT_PATH = Path(__file__).resolve().parent / "sft_teacher_client.py"
DEFAULT_SHUFFLE_SEED = 20260517
SPLIT_CHOICES = ("train", "validation", "test")

STUDENT_PROMPT_BEGIN = "[STUDENT_RUNTIME_SYSTEM_PROMPT_BEGIN]"
STUDENT_PROMPT_END = "[STUDENT_RUNTIME_SYSTEM_PROMPT_END]"

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
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_student_runtime_system_prompt(teacher_client_path: Path) -> str:
    source = teacher_client_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(STUDENT_PROMPT_BEGIN)}\s*(.*?)\s*{re.escape(STUDENT_PROMPT_END)}",
        re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(
            f"Could not find student runtime prompt markers in {teacher_client_path}"
        )
    return match.group(1).strip()


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
        if isinstance(item, dict):
            records.append(item)
    return records


def iter_accepted_samples(accepted_dir: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(accepted_dir.glob("*.jsonl")):
        for sample in read_json_records(path):
            yield path, sample


def require_mapping(value: Any, field_name: str, sample_ref: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{sample_ref} must contain object field {field_name}")
    return value


def build_runtime_user_message_from_accepted(sample: dict[str, Any], sample_ref: str) -> str:
    source_input = require_mapping(sample.get("input"), "input", sample_ref)
    runtime_input = require_mapping(source_input.get("input"), "input.input", sample_ref)
    command_analysis = require_mapping(
        source_input.get("commandAnalysis"), "input.commandAnalysis", sample_ref
    )

    user_message = {
        "input": runtime_input,
        "commandAnalysis": command_analysis,
        "output_schema_example": OUTPUT_SCHEMA_EXAMPLE,
        "hard_constraints": HARD_CONSTRAINTS,
    }
    return compact_json(user_message)


def build_sft_record(
    sample: dict[str, Any], sample_ref: str, student_runtime_system_prompt: str
) -> dict[str, Any]:
    output = require_mapping(sample.get("output"), "output", sample_ref)
    return {
        "messages": [
            {
                "role": "system",
                "content": student_runtime_system_prompt,
            },
            {
                "role": "user",
                "content": build_runtime_user_message_from_accepted(sample, sample_ref),
            },
            {
                "role": "assistant",
                "content": compact_json(output),
            },
        ]
    }


def convert_accepted_to_sft_messages(
    accepted_dir: Path,
    output_path: Path,
    teacher_client_path: Path,
    seed: int,
    split: str | None = None,
) -> int:
    if not accepted_dir.is_dir():
        raise FileNotFoundError(f"Accepted directory does not exist: {accepted_dir}")

    student_runtime_system_prompt = load_student_runtime_system_prompt(teacher_client_path)
    samples: list[tuple[Path, dict[str, Any]]] = list(iter_accepted_samples(accepted_dir))
    if split is not None:
        samples = [
            (source_path, sample)
            for source_path, sample in samples
            if sample.get("split") == split
        ]
    if not samples:
        if split is None:
            raise ValueError(f"No accepted samples found in {accepted_dir}")
        raise ValueError(f"No accepted samples found in {accepted_dir} for split={split}")

    random.Random(seed).shuffle(samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as dst:
        for index, (source_path, sample) in enumerate(samples, start=1):
            sample_id = sample.get("id")
            sample_ref = f"{source_path}:{sample_id if isinstance(sample_id, str) else index}"
            record = build_sft_record(sample, sample_ref, student_runtime_system_prompt)
            dst.write(compact_json(record) + "\n")

    return len(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one shuffled messages JSONL for SFT from every accepted JSONL sample."
    )
    parser.add_argument(
        "--accepted-dir",
        type=Path,
        default=DEFAULT_ACCEPTED_DIR,
        help=f"Accepted JSONL directory. Default: {DEFAULT_ACCEPTED_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output SFT JSONL path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--teacher-client",
        type=Path,
        default=DEFAULT_TEACHER_CLIENT_PATH,
        help=f"File containing student runtime prompt markers. Default: {DEFAULT_TEACHER_CLIENT_PATH}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Deterministic shuffle seed. Default: {DEFAULT_SHUFFLE_SEED}",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default=None,
        help="Only include records whose accepted sample split matches this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_accepted_to_sft_messages(
        accepted_dir=args.accepted_dir,
        output_path=args.output,
        teacher_client_path=args.teacher_client,
        seed=args.seed,
        split=args.split,
    )
    print(f"wrote {count} SFT records to {args.output}")


if __name__ == "__main__":
    main()
