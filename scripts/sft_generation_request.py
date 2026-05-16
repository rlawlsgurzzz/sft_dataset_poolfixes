# accepted 샘플에서 기준 command slot을 찾고 teacher 생성 요청 payload를 만든다.
# 숫자 path는 sft_taxonomy.py로 검증하고 내부 처리는 stable path 기준으로 수행한다.
# selected_bucket에는 분류 기준, edge, skill_case, 생성 계약을 포함한다.
# existing_valid_paraphrase_samples에는 요청 split의 cycle 가능 표현만 포함한다.
# other_split_reserved_command_texts에는 다른 split의 중복 금지 표현만 포함한다.
# commandAnalysis는 teacher 생성 대상이 아니며 validator가 accepted 저장 시 계산한다.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from sft_file_sequence import next_numbered_path

try:
    from sft_taxonomy import (
        DEFAULT_TAXONOMY_PATH,
        GeneralPath,
        SkillPath,
        get_selected_bucket_descriptions,
        load_taxonomy,
        parse_generation_request,
    )
    from sft_coverage_report import (
        DEFAULT_DATASET_ROOT,
        get_base_command_text,
        get_edge_flags,
        get_general_path,
        get_metadata,
        get_skill_path,
        load_accepted_samples,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_taxonomy import (
        DEFAULT_TAXONOMY_PATH,
        GeneralPath,
        SkillPath,
        get_selected_bucket_descriptions,
        load_taxonomy,
        parse_generation_request,
    )
    from sft_coverage_report import (
        DEFAULT_DATASET_ROOT,
        get_base_command_text,
        get_edge_flags,
        get_general_path,
        get_metadata,
        get_skill_path,
        load_accepted_samples,
    )


DEFAULT_REQUEST_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "raw_generations"
DEFAULT_TARGET_SPLIT = "train"
VALID_SPLITS = ("train", "validation", "test")

SPLIT_EXPRESSION_POOL_LIMITS: dict[str, int] = {
    "train": 7,
    "validation": 3,
    "test": 3,
}


DEFAULT_SKILL_FLAGS: dict[str, dict[str, bool]] = {
    "enemy_single_target_attack": {
        "is_skill_aoe": False,
        "can_skill_target_dead": False,
    },
    "self_buff": {"is_skill_aoe": False, "can_skill_target_dead": False},
    "ally_shield": {"is_skill_aoe": False, "can_skill_target_dead": False},
    "ally_heal": {"is_skill_aoe": False, "can_skill_target_dead": False},
    "ally_resurrection": {"is_skill_aoe": False, "can_skill_target_dead": True},
    "enemy_aoe_attack": {"is_skill_aoe": True, "can_skill_target_dead": False},
    "enemy_debuff": {"is_skill_aoe": False, "can_skill_target_dead": False},
    "mobility_skill": {"is_skill_aoe": False, "can_skill_target_dead": False},
    "no_skill": {"is_skill_aoe": False, "can_skill_target_dead": False},
}


# accepted sample에서 실제 command_text를 추출한다.
def sample_command_text(sample: dict[str, Any]) -> str:
    command_spec = sample.get("command_spec")
    if isinstance(command_spec, dict):
        command_text = command_spec.get("command_text")
        if isinstance(command_text, str) and command_text:
            return command_text

    input_obj = sample.get("input")
    if isinstance(input_obj, dict):
        nested = input_obj.get("input")
        if isinstance(nested, dict):
            command = nested.get("command")
            if isinstance(command, str) and command:
                return command

    return get_base_command_text(sample)


# accepted sample metadata에서 command_style을 추출한다.
def sample_command_style(sample: dict[str, Any]) -> str:
    metadata = get_metadata(sample)
    value = metadata.get("command_style")
    if isinstance(value, str) and value:
        return value
    return "direct_korean"


def normalize_target_split(value: str) -> str:
    if value not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}: {value}")
    return value


def sample_split(sample: dict[str, Any]) -> str:
    value = sample.get("split")
    if isinstance(value, str) and value:
        return value
    return DEFAULT_TARGET_SPLIT


def get_expression_pool_limit(target_split: str) -> int:
    target_split = normalize_target_split(target_split)
    return SPLIT_EXPRESSION_POOL_LIMITS[target_split]


def general_path_matches(sample: dict[str, Any], path: GeneralPath) -> bool:
    return get_general_path(sample) == (
        path.intent_family,
        path.actor_selection,
        path.target_selection,
        path.action_pattern,
        path.scenario_family,
    )


# 같은 general path 안에서 base command와 edge_flags가 같은 샘플들을 command slot으로 묶는다.
def group_command_slots(
    samples: list[dict[str, Any]],
    path: GeneralPath,
) -> list[tuple[str, tuple[str, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        if not general_path_matches(sample, path):
            continue
        key = (get_base_command_text(sample), get_edge_flags(sample))
        groups[key].append(sample)

    rows: list[tuple[str, tuple[str, ...], list[dict[str, Any]]]] = []
    for (base_command_text, edge_flags), grouped_samples in groups.items():
        rows.append((base_command_text, edge_flags, grouped_samples))

    return sorted(rows, key=lambda item: item[0])


# 요청 path의 command_slot_index에 대응하는 accepted command slot을 찾는다.
def get_command_slot_samples(
    samples: list[dict[str, Any]],
    path: GeneralPath,
) -> tuple[str, tuple[str, ...], list[dict[str, Any]]]:
    rows = group_command_slots(samples, path)
    index = path.command_slot_index

    if not rows:
        raise ValueError(
            f"No accepted command slots found for general path: {path.stable_path}"
        )
    if index < 1 or index > len(rows):
        raise ValueError(
            f"command_slot_index out of range for {path.stable_path}: {index}. "
            f"Available rows: {len(rows)}"
        )

    return rows[index - 1]


# skill path가 요청에 직접 없으면 같은 command slot의 accepted sample에서 하나로 추론한다.
def infer_skill_path_from_samples(samples: list[dict[str, Any]]) -> Optional[SkillPath]:
    skill_paths = {get_skill_path(sample) for sample in samples}
    skill_paths.discard(None)

    if not skill_paths:
        return None
    if len(skill_paths) > 1:
        raise ValueError(
            "Selected command slot contains multiple skill paths. Use an explicit skill override."
        )

    family, target_kind, conflict_key = next(iter(skill_paths))
    return SkillPath(
        skill_family=family,
        skill_target_kind=target_kind,
        conflict_type=None if conflict_key == "null" else conflict_key,
    )


# accepted sample의 skill_case boolean을 우선 사용하고 없으면 skill_family 기본값을 사용한다.
def infer_skill_flags_from_samples(
    samples: list[dict[str, Any]],
    skill_path: SkillPath,
) -> dict[str, bool]:
    for sample in samples:
        skill_case = sample.get("skill_case")
        if not isinstance(skill_case, dict):
            continue
        if skill_case.get("skill_family") != skill_path.skill_family:
            continue
        if skill_case.get("skill_target_kind") != skill_path.skill_target_kind:
            continue
        if skill_case.get("conflict_type") != skill_path.conflict_type:
            continue

        is_aoe = skill_case.get("is_skill_aoe")
        can_dead = skill_case.get("can_skill_target_dead")
        if isinstance(is_aoe, bool) and isinstance(can_dead, bool):
            return {
                "is_skill_aoe": is_aoe,
                "can_skill_target_dead": can_dead,
            }

    return dict(
        DEFAULT_SKILL_FLAGS.get(
            skill_path.skill_family,
            {"is_skill_aoe": False, "can_skill_target_dead": False},
        )
    )


# selected_bucket에 skill_case를 명시해 teacher가 skill 조건을 같은 축으로 생성하게 한다.
def attach_skill_case_fields(
    selected_bucket: dict[str, Any],
    skill_path: Optional[SkillPath],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_bucket = dict(selected_bucket)

    if skill_path is None:
        selected_bucket["skill_case"] = None
        return selected_bucket

    skill_case = selected_bucket.get("skill_case")
    if not isinstance(skill_case, dict):
        skill_case = {}

    flags = infer_skill_flags_from_samples(samples, skill_path)
    skill_case["skill_family"] = skill_path.skill_family
    skill_case["skill_target_kind"] = skill_path.skill_target_kind
    skill_case["is_skill_aoe"] = flags["is_skill_aoe"]
    skill_case["can_skill_target_dead"] = flags["can_skill_target_dead"]
    skill_case["conflict_type"] = skill_path.conflict_type
    selected_bucket["skill_case"] = skill_case
    return selected_bucket


# 요청 split 안에서 cycle 가능한 기존 표현 pool을 만든다.
def build_existing_paraphrase_samples(
    samples: list[dict[str, Any]],
    target_split: str,
) -> list[dict[str, str]]:
    target_split = normalize_target_split(target_split)
    pool_limit = get_expression_pool_limit(target_split)

    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []

    for sample in samples:
        if sample_split(sample) != target_split:
            continue

        command_text = sample_command_text(sample)
        command_style = sample_command_style(sample)
        key = (command_text, command_style)

        if key in seen:
            continue

        seen.add(key)
        result.append(
            {
                "command_text": command_text,
                "command_style": command_style,
            }
        )

        if len(result) >= pool_limit:
            break

    return result


# 다른 split에 이미 존재하는 표현을 중복 금지 목록으로 만든다.
def build_other_split_reserved_command_texts(
    samples: list[dict[str, Any]],
    target_split: str,
) -> list[dict[str, str]]:
    target_split = normalize_target_split(target_split)

    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, str]] = []

    for sample in samples:
        split = sample_split(sample)

        if split == target_split:
            continue

        command_text = sample_command_text(sample)
        command_style = sample_command_style(sample)
        key = (split, command_text, command_style)

        if key in seen:
            continue

        seen.add(key)
        result.append(
            {
                "split": split,
                "command_text": command_text,
                "command_style": command_style,
            }
        )

    return result


def build_command_text_sequence_contract(
    *,
    existing_count: int,
    new_unique_count: int,
    cycle_count: int,
    cycle_start_offset: int = 0,
) -> dict[str, Any]:
    new_unique_start = 1 if new_unique_count > 0 else None
    new_unique_end = new_unique_count if new_unique_count > 0 else None

    cycle_start = new_unique_count + 1 if cycle_count > 0 else None
    cycle_end = new_unique_count + cycle_count if cycle_count > 0 else None

    cycle_source_pool_size = existing_count + new_unique_count
    normalized_cycle_start_offset = (
        cycle_start_offset % cycle_source_pool_size
        if cycle_count > 0 and cycle_source_pool_size > 0
        else 0
    )
    cycle_reuse_plan: list[dict[str, Any]] = []

    if cycle_count > 0 and cycle_source_pool_size > 0:
        for offset in range(cycle_count):
            source_pool_index = (
                (normalized_cycle_start_offset + offset) % cycle_source_pool_size
            ) + 1
            output_index = new_unique_count + offset + 1

            if source_pool_index <= existing_count:
                source_kind = "existing_valid_paraphrase_samples"
                source_index_1_based = source_pool_index
            else:
                source_kind = "newly_created_unique_command_texts_in_this_response"
                source_index_1_based = source_pool_index - existing_count

            cycle_reuse_plan.append(
                {
                    "output_index_1_based": output_index,
                    "source_pool_index_1_based": source_pool_index,
                    "source_kind": source_kind,
                    "source_index_1_based": source_index_1_based,
                }
            )

    return {
        "output_order": "new_unique_first_then_cycle",
        "new_unique_output_range_1_based": (
            [new_unique_start, new_unique_end]
            if new_unique_start is not None and new_unique_end is not None
            else []
        ),
        "cycle_output_range_1_based": (
            [cycle_start, cycle_end]
            if cycle_start is not None and cycle_end is not None
            else []
        ),
        "cycle_source_pool_order": (
            "existing_valid_paraphrase_samples in payload order followed by "
            "newly_created_unique_command_texts in output order"
        ),
        "cycle_reuse_strategy": "round_robin_from_cycle_start_offset",
        "cycle_start_offset_0_based": normalized_cycle_start_offset,
        "cycle_source_pool_size_after_new_unique": cycle_source_pool_size,
        "cycle_reuse_plan_1_based": cycle_reuse_plan,
        "hard_fail_conditions": [
            "cycle samples appear before all new_unique samples are created",
            "a new_unique output reuses an existing command_text",
            "a cycle output ignores cycle_reuse_plan_1_based",
            "a cycle output repeatedly uses only source_pool_index_1_based=1 when cycle_reuse_plan_1_based specifies otherwise",
        ],
        "example": (
            "if existing=7,new=0,cycle=5,cycle_start_offset=5 then outputs 1-5 "
            "reuse source pool indexes [6,7,1,2,3]"
        ),
    }


# split별 표현 pool 생성/순환 정책을 payload에 명시한다.
def build_command_text_policy(
    existing_paraphrase_samples: list[dict[str, str]],
    other_split_reserved_command_texts: list[dict[str, str]],
    target_split: str,
    count_to_generate: int,
    cycle_start_offset: int = 0,
) -> dict[str, Any]:
    target_split = normalize_target_split(target_split)
    pool_limit = get_expression_pool_limit(target_split)

    existing_count = len(existing_paraphrase_samples)
    new_unique_count = max(0, min(count_to_generate, pool_limit - existing_count))
    cycle_count = max(0, count_to_generate - new_unique_count)

    return {
        "target_split": target_split,
        "same_split_expression_pool_size": pool_limit,
        "existing_same_split_expression_count": existing_count,
        "other_split_reserved_expression_count": len(
            other_split_reserved_command_texts
        ),
        "new_unique_command_texts_to_create": new_unique_count,
        "samples_using_same_split_cycle": cycle_count,
        "cycle_source": "existing_valid_paraphrase_samples plus newly created unique command_texts in this request",
        "sequence_contract": build_command_text_sequence_contract(
            existing_count=existing_count,
            new_unique_count=new_unique_count,
            cycle_count=cycle_count,
            cycle_start_offset=cycle_start_offset,
        ),
        "rules": [
            "같은 split 안에서 command_text 표현 pool을 관리한다.",
            "표현 pool이 가득 차기 전에는 새 command_text를 만든다.",
            "새 command_text는 existing_valid_paraphrase_samples와 exact duplicate이면 안 된다.",
            "출력 array의 앞쪽 new_unique_command_texts_to_create개 sample은 반드시 새 unique command_text로 만든다.",
            "새 unique command_text 생성이 모두 끝난 뒤에만 samples_using_same_split_cycle개 sample을 만든다.",
            "cycle source pool은 existing_valid_paraphrase_samples의 payload 순서 뒤에 이번 응답에서 만든 새 unique command_text를 output 순서대로 이어 붙인 목록이다.",
            "cycle sample은 sequence_contract.cycle_reuse_plan_1_based를 따라 source command_text를 재사용한다.",
            "cycle 구간에서 첫 번째 source command_text만 반복하거나 같은 command_text를 연속 반복하지 않는다.",
            "새 command_text는 other_split_reserved_command_texts와 exact duplicate이면 안 된다.",
            "표현 pool이 가득 찬 뒤에는 같은 split 표현 pool 안에서만 command_text를 순환 재사용할 수 있다.",
            "같은 요청에서 새로 만든 unique command_text도 이후 cycle source로 사용할 수 있다.",
            "other_split_reserved_command_texts는 cycle source가 아니다.",
            "command_text를 재사용하더라도 area_situation, gold, output은 새로 구성한다.",
            "다른 split의 command_text 표현 pool을 cycle 대상으로 섞지 않는다.",
        ],
    }


def build_runtime_generation_contract() -> dict[str, Any]:
    return {
        "teacher_role": "전장 시나리오와 정답 output을 생성한다.",
        "validator_role": "area_situation을 검증하고 commandAnalysis를 계산해 accepted 저장 시점에 추가한다.",
        "teacher_must_create": [
            "id",
            "split",
            "command_spec",
            "metadata",
            "skill_case",
            "gold",
            "input.input.command",
            "input.input.area_situation.allies",
            "input.input.area_situation.enemies",
            "output.thinking",
            "output.dialog",
            "output.action",
        ],
        "teacher_must_not_create": [
            "input.commandAnalysis",
            "commandAnalysis",
            "allowedActors",
            "allowedAttackTargets",
            "validMoveToUnits",
            "deadAllies",
            "invalidUnits",
            "actionPolicy",
            "validator_result",
            "source_ref",
        ],
        "validator_will_create_on_accept": [
            "input.commandAnalysis.analysisMode",
            "input.commandAnalysis.allowedActors",
            "input.commandAnalysis.allowedAttackTargets",
            "input.commandAnalysis.validMoveToUnits",
            "input.commandAnalysis.deadAllies",
            "input.commandAnalysis.invalidUnits",
            "input.commandAnalysis.actionPolicy",
        ],
        "runtime_student_input_after_accept": {
            "input": {
                "command": "한국어 명령",
                "area_situation": {
                    "allies": "A_01부터 A_06까지 정확히 6명",
                    "enemies": "E_01부터 E_06까지 정확히 6명",
                },
            },
            "commandAnalysis": "validator가 계산한 runtime_constraint_summary",
        },
    }


def build_area_situation_contract() -> dict[str, Any]:
    return {
        "allies_count": 6,
        "enemies_count": 6,
        "ally_ids": [f"A_{index:02d}" for index in range(1, 7)],
        "enemy_ids": [f"E_{index:02d}" for index in range(1, 7)],
        "ally_required_fields": [
            "unitId",
            "isAlive",
            "canBeTargeted",
            "isRanged",
            "hpRatio",
            "attackRatioToAvg",
            "engagedByOpponentCount",
            "teamFormationRole",
            "skillDescription",
            "IsSkillOnSelf",
            "IsSkillOnOtherAlly",
            "isSkillAoe",
            "canSkillTargetDead",
            "closestTargetableOpponent",
            "farthestTargetableOpponent",
            "closestAliveAlly",
            "farthestAliveAlly",
        ],
        "enemy_required_fields": [
            "unitId",
            "isAlive",
            "canBeTargeted",
            "isRanged",
            "hpRatio",
            "attackRatioToAvg",
            "engagedByOpponentCount",
            "teamFormationRole",
        ],
        "single_distance_fields": {
            "closestTargetableOpponent": "살아있고 canBeTargeted=true인 enemy unitId 또는 null",
            "farthestTargetableOpponent": "살아있고 canBeTargeted=true인 enemy unitId 또는 null",
            "closestAliveAlly": "자기 자신을 제외한 살아있는 ally unitId 또는 null",
            "farthestAliveAlly": "자기 자신을 제외한 살아있는 ally unitId 또는 null",
        },
        "single_distance_field_rules": [
            "네 필드는 배열이 아니라 unitId string 또는 null이다.",
            "후보가 없으면 null을 사용한다.",
            "후보가 1명뿐이면 closest와 farthest에 같은 unitId를 사용한다.",
            "후보가 2명 이상이면 closest와 farthest는 서로 다른 unitId를 사용한다.",
            "closestAliveAlly와 farthestAliveAlly에는 자기 자신의 unitId를 쓰지 않는다.",
            "죽은 unit은 네 필드에 들어가지 않는다.",
        ],
        "dead_unit_rules": [
            "이 게임에서는 죽은 unit도 보통 canBeTargeted=true일 수 있다.",
            "죽은 ally가 canBeTargeted=true이면 validator가 commandAnalysis.deadAllies에 포함한다.",
            "죽은 unit은 actor, attack target, move.to로 사용할 수 없다.",
            "canSkillTargetDead=true인 skill에서만 죽은 targetable ally를 skill target으로 사용할 수 있다.",
        ],
    }


def build_output_contract() -> dict[str, Any]:
    return {
        "assistant_output_top_level_keys": ["thinking", "dialog", "action"],
        "thinking": "짧은 한국어 판단 요약. 자세한 사고 과정 금지.",
        "dialog": [
            "action actor마다 정확히 하나만 생성한다.",
            "action에 없는 unitId를 넣지 않는다.",
            "여러 actor에게 완전히 같은 text를 반복하지 않는다.",
        ],
        "action": [
            "actor는 살아있는 ally만 가능하다.",
            "enemy는 actor가 될 수 없다.",
            "각 actor sequence는 최대 3개 action이다.",
            "실행 가능한 action이 없으면 dialog와 action은 빈 배열이다.",
        ],
        "skill": [
            "skill.description은 actor.skillDescription과 정확히 같아야 한다.",
            "IsSkillOnSelf=true이면 target은 actor 자신이다.",
            "IsSkillOnOtherAlly=true이면 target은 actor 자신이 아닌 ally다.",
            "IsSkillOnSelf=false이고 IsSkillOnOtherAlly=false이면 target은 enemy다.",
            "canSkillTargetDead=false이면 죽은 unit을 skill target으로 쓰지 않는다.",
            "isSkillAoe=true여도 output target은 중심 unitId 하나만 쓴다.",
        ],
        "wait_and_skill_control": [
            "wait은 명령에 대기, 지연, 타이밍 조절, 위치 유지 의미가 직접 있을 때만 사용한다.",
            "attack 또는 skill 뒤에는 wait을 붙이지 않는다.",
            "skillControl은 스킬 지연 또는 금지 의도가 명시될 때만 사용한다.",
            "조건부 명령은 current-state-only로 처리한다.",
        ],
    }


def build_generation_constraints(target_split: str = DEFAULT_TARGET_SPLIT) -> list[str]:
    target_split = normalize_target_split(target_split)

    return [
        "selected_bucket만 따른다.",
        f"각 sample.split은 반드시 {target_split}이다.",
        "새 command는 selected_bucket의 command slot 의미와 edge_flags를 유지한다.",
        "command_text_policy를 따른다.",
        "metadata.command_style은 direct_korean, casual_korean, elliptical_korean, tactical_korean, rough_korean 중 하나만 사용한다.",
        "metadata.command_style에 informal을 절대 사용하지 않는다.",
        "existing_valid_paraphrase_samples는 요청 split의 cycle 가능 표현 pool이다.",
        "other_split_reserved_command_texts는 다른 split의 중복 금지 표현 목록이다.",
        "새 command_text는 existing_valid_paraphrase_samples와 exact duplicate이면 안 된다.",
        "새 command_text는 other_split_reserved_command_texts와 exact duplicate이면 안 된다.",
        "표현 pool이 가득 찬 뒤에는 같은 split 표현 pool 안에서만 command_text를 순환 재사용할 수 있다.",
        "같은 요청에서 새로 만든 unique command_text도 이후 cycle source로 사용할 수 있다.",
        "other_split_reserved_command_texts는 cycle source가 아니다.",
        "command_text를 재사용하더라도 area_situation, gold, output은 복사하지 않고 새로 구성한다.",
        "taxonomy 밖 값을 만들지 않는다.",
        "source_ref를 생성하지 않는다.",
        "validator_result를 생성하지 않는다.",
        "input.commandAnalysis와 commandAnalysis 하위 필드를 절대 생성하지 않는다.",
        "input.input.area_situation을 완전한 전장 상태로 창작한다.",
        "area_situation.allies에는 A_01부터 A_06까지 정확히 6명의 ally를 만든다.",
        "area_situation.enemies에는 E_01부터 E_06까지 정확히 6명의 enemy를 만든다.",
        "아군 유닛에는 IsSkillOnOtherAlly와 단일 closest/farthest 필드를 사용한다.",
        "closestTargetableOpponent, farthestTargetableOpponent, closestAliveAlly, farthestAliveAlly는 배열이 아니라 unitId string 또는 null이며, 살아있는 유효 후보만 넣는다.",
        "죽은 unit도 보통 canBeTargeted=true일 수 있지만, 새 네 필드에는 죽은 unit을 넣지 않는다.",
        "output은 sanitizer가 고친 결과가 아니라 처음부터 raw valid assistant output이어야 한다.",
        "output은 학생 SLM runtime system prompt를 실제로 적용했을 때의 thinking/dialog/action 정답이어야 한다.",
    ]


# 숫자 요청과 accepted 샘플을 합쳐 teacher LLM 입력 payload를 만든다.
def build_generation_payload(
    raw_request: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    target_split: str = DEFAULT_TARGET_SPLIT,
    cycle_start_offset: int = 0,
) -> dict[str, Any]:
    target_split = normalize_target_split(target_split)

    taxonomy = load_taxonomy(taxonomy_path)
    parsed = parse_generation_request(raw_request, taxonomy)
    samples = load_accepted_samples(dataset_root / "accepted")

    base_command_text, edge_flags, command_slot_samples = get_command_slot_samples(
        samples=samples,
        path=parsed.general_path,
    )

    skill_path = parsed.skill_path

    if skill_path is None and parsed.general_path.intent_family == "skill":
        skill_path = infer_skill_path_from_samples(command_slot_samples)

    if skill_path is None and parsed.general_path.intent_family == "skill":
        raise ValueError(
            "Skill intent_family request requires skill_path or accepted samples with skill_case."
        )

    selected_bucket = get_selected_bucket_descriptions(
        taxonomy=taxonomy,
        general_path=parsed.general_path,
        skill_path=skill_path,
        edge_flags=list(edge_flags),
    )

    selected_bucket = attach_skill_case_fields(
        selected_bucket=selected_bucket,
        skill_path=skill_path,
        samples=command_slot_samples,
    )

    existing_paraphrase_samples = build_existing_paraphrase_samples(
        samples=command_slot_samples,
        target_split=target_split,
    )

    other_split_reserved_command_texts = build_other_split_reserved_command_texts(
        samples=command_slot_samples,
        target_split=target_split,
    )

    command_text_policy = build_command_text_policy(
        existing_paraphrase_samples=existing_paraphrase_samples,
        other_split_reserved_command_texts=other_split_reserved_command_texts,
        target_split=target_split,
        count_to_generate=parsed.count,
        cycle_start_offset=cycle_start_offset,
    )

    return {
        "request": {
            "raw_request": parsed.raw_request,
            "count_to_generate": parsed.count,
            "display_path": parsed.raw_request.split(".")[0].removeprefix("c"),
            "stable_path": parsed.general_path.stable_path,
            "skill_stable_path": None if skill_path is None else skill_path.stable_path,
            "command_slot_index": parsed.general_path.command_slot_index,
            "base_command_text": base_command_text,
        },
        "target_split": target_split,
        "selected_bucket": selected_bucket,
        "existing_valid_paraphrase_samples": existing_paraphrase_samples,
        "other_split_reserved_command_texts": other_split_reserved_command_texts,
        "command_text_policy": command_text_policy,
        "count_to_generate": parsed.count,
        "runtime_generation_contract": build_runtime_generation_contract(),
        "area_situation_contract": build_area_situation_contract(),
        "assistant_output_contract": build_output_contract(),
        "generation_constraints": build_generation_constraints(target_split),
    }


# mixed path request의 item에서 LLM이 읽어야 하는 path별 계약만 남긴다.
def build_mixed_generation_item(
    *,
    mix_item_index_1_based: int,
    payload: dict[str, Any],
    cycle_start_offset: int,
) -> dict[str, Any]:
    return {
        "mix_item_index_1_based": mix_item_index_1_based,
        "request": payload["request"],
        "target_split": payload["target_split"],
        "selected_bucket": payload["selected_bucket"],
        "existing_valid_paraphrase_samples": payload[
            "existing_valid_paraphrase_samples"
        ],
        "other_split_reserved_command_texts": payload[
            "other_split_reserved_command_texts"
        ],
        "command_text_policy": payload["command_text_policy"],
        "count_to_generate": payload["count_to_generate"],
        "cycle_start_offset_used": cycle_start_offset,
    }


# 서로 다른 두 path를 하나의 teacher payload 안에 넣는 mixed generation payload를 만든다.
def build_mixed_generation_payload(
    mixed_requests: list[dict[str, Any]],
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    target_split: str = DEFAULT_TARGET_SPLIT,
) -> dict[str, Any]:
    target_split = normalize_target_split(target_split)

    if not mixed_requests:
        raise ValueError("generation payload requires at least one path request")

    item_payloads: list[dict[str, Any]] = []
    total_count = 0

    for index, item in enumerate(mixed_requests, start=1):
        raw_request = item.get("request")
        if not isinstance(raw_request, str) or not raw_request:
            raise ValueError(f"mixed request item {index} has no request string")

        cycle_start_offset = int(item.get("cycle_start_offset", 0))

        payload = build_generation_payload(
            raw_request=raw_request,
            dataset_root=dataset_root,
            taxonomy_path=taxonomy_path,
            target_split=target_split,
            cycle_start_offset=cycle_start_offset,
        )

        item_payloads.append(
            build_mixed_generation_item(
                mix_item_index_1_based=index,
                payload=payload,
                cycle_start_offset=cycle_start_offset,
            )
        )
        total_count += int(payload["count_to_generate"])

    path_keys = [
        item["request"]["raw_request"].rsplit(".", 1)[0] for item in item_payloads
    ]
    if len(path_keys) != len(set(path_keys)):
        raise ValueError("generation payload path requests must be distinct")

    return {
        "request": {
            "count_to_generate": total_count,
            "path_count": len(item_payloads),
            "items": [
                {
                    "mix_item_index_1_based": item["mix_item_index_1_based"],
                    "raw_request": item["request"]["raw_request"],
                    "count_to_generate": item["count_to_generate"],
                    "stable_path": item["request"]["stable_path"],
                    "skill_stable_path": item["request"]["skill_stable_path"],
                    "cycle_start_offset_used": item["cycle_start_offset_used"],
                }
                for item in item_payloads
            ],
        },
        "target_split": target_split,
        "mixed_generation_requests": item_payloads,
        "count_to_generate": total_count,
        "mixed_output_contract": {
            "output_order": "generate all samples for mixed_generation_requests in array order; finish one item before starting the next item",
            "per_item_rule": "Each mixed_generation_requests item is an independent generation contract.",
            "count_rule": "The final JSON array length must equal top-level count_to_generate.",
        },
        "runtime_generation_contract": build_runtime_generation_contract(),
        "area_situation_contract": build_area_situation_contract(),
        "assistant_output_contract": build_output_contract(),
        "generation_constraints": [
            "mixed_generation_requests의 각 item을 독립된 generation contract로 처리한다.",
            "각 item의 selected_bucket, existing_valid_paraphrase_samples, other_split_reserved_command_texts, command_text_policy를 해당 item sample에만 적용한다.",
            "한 item의 계약 필드를 다른 item sample에 섞어 쓰지 않는다.",
            "출력 array는 mixed_output_contract.output_order를 따른다.",
            "각 sample.split은 자신이 속한 item의 target_split과 같아야 한다.",
            "각 item의 command_text_policy.sequence_contract.cycle_reuse_plan_1_based를 그대로 따른다.",
            "cycle source index를 다시 계산하지 않는다.",
            "command_text를 재사용하더라도 area_situation, gold, output은 복사하지 않고 새로 구성한다.",
            "taxonomy 밖 값을 만들지 않는다.",
            "source_ref를 생성하지 않는다.",
            "validator_result를 생성하지 않는다.",
            "input.commandAnalysis와 commandAnalysis 하위 필드를 절대 생성하지 않는다.",
            "input.input.area_situation을 완전한 전장 상태로 창작한다.",
            "output은 처음부터 raw valid assistant output이어야 한다.",
        ],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def make_default_output_path(output_dir: Path, raw_request: str) -> Path:
    return next_numbered_path(output_dir, "request", ".json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", help="Generation request, e.g. c1-2-1-3-1-10.4")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--output", default="")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--split", choices=VALID_SPLITS, default=DEFAULT_TARGET_SPLIT)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    payload = build_mixed_generation_payload(
        mixed_requests=[
            {
                "request": args.request,
                "cycle_start_offset": 0,
            }
        ],
        dataset_root=dataset_root,
        taxonomy_path=Path(args.taxonomy),
        target_split=args.split,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = make_default_output_path(
            dataset_root / "raw_generations", args.request
        )

    write_json(output_path, payload)
    print(f"generation_request: {output_path}")

    if args.print_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
