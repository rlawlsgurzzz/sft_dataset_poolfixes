# teacher master sample 후보를 검증해 accepted/rejected JSONL로 분류한다.
# raw teacher output에는 commandAnalysis를 허용하지 않는다.
# validator가 area_situation에서 commandAnalysis를 계산하고 accepted 저장 시 추가한다.
# 현재 기준은 Readme/gemma_ollama_test19.py의 단일 closest/farthest unitId 스키마다.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from sft_taxonomy import DEFAULT_TAXONOMY_PATH, load_taxonomy, validate_metadata_against_taxonomy
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from sft_taxonomy import DEFAULT_TAXONOMY_PATH, load_taxonomy, validate_metadata_against_taxonomy


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTED_DIR = DEFAULT_DATASET_ROOT / "accepted"
DEFAULT_REJECTED_DIR = DEFAULT_DATASET_ROOT / "rejected"
VALIDATION_INDEX_FILENAME = ".validated_inputs.json"

DEFAULT_WAIT_SECONDS = 2.0
DEFAULT_SKILL_CONTROL_DEFER_SECONDS = 5.0
MIN_WAIT_SECONDS = 1.0
MAX_WAIT_SECONDS = 10.0
MAX_ACTIONS_PER_ACTOR = 3

EXPECTED_ALLY_IDS = {f"A_{index:02d}" for index in range(1, 7)}
EXPECTED_ENEMY_IDS = {f"E_{index:02d}" for index in range(1, 7)}
TEAM_FORMATION_ROLES = {"frontline", "midline", "backline"}

ALLOWED_MOVE_SUBTYPES = {"approachOpponent", "escape", "help", "holdFront"}
ALLOWED_MOVEMENT_TYPES = {"direct", "flank"}
ALLOWED_ACTION_TYPES = {"move", "attack", "skill", "wait", "skillControl"}
EXPECTED_OUTPUT_KEYS = {"thinking", "dialog", "action"}

ALLY_UNIT_FIELDS = {
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
}

ENEMY_UNIT_FIELDS = {
    "unitId",
    "isAlive",
    "canBeTargeted",
    "isRanged",
    "hpRatio",
    "attackRatioToAvg",
    "engagedByOpponentCount",
    "teamFormationRole",
}


RAW_FORBIDDEN_KEYS = {
    "commandAnalysis",
    "allowedActors",
    "allowedAttackTargets",
    "validMoveToUnits",
    "deadAllies",
    "invalidUnits",
    "actionPolicy",
    "validator_result",
    "source_ref",
}


class ValidationContext:
    def __init__(self, sample: dict[str, Any], taxonomy: dict[str, Any]) -> None:
        self.sample = sample
        self.taxonomy = taxonomy
        self.errors: list[str] = []
        self.command_analysis: dict[str, Any] | None = None

    def add(self, code: str, detail: str = "") -> None:
        self.errors.append(code if not detail else f"{code}: {detail}")


def read_json_records_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], []

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    def add_json_value(value: Any, source_label: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    valid.append(item)
                else:
                    invalid.append(
                        {
                            "raw": item,
                            "failure_reasons": [f"SAMPLE_NOT_OBJECT at {source_label}[{index}]"],
                        }
                    )
            return

        if isinstance(value, dict) and isinstance(value.get("samples"), list):
            for index, item in enumerate(value["samples"], start=1):
                if isinstance(item, dict):
                    valid.append(item)
                else:
                    invalid.append(
                        {
                            "raw": item,
                            "failure_reasons": [f"SAMPLE_NOT_OBJECT at {source_label}.samples[{index}]"],
                        }
                    )
            return

        if isinstance(value, dict):
            valid.append(value)
            return

        invalid.append(
            {
                "raw": value,
                "failure_reasons": [f"ROOT_NOT_SAMPLE_OR_SAMPLE_LIST at {source_label}"],
            }
        )

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        add_json_value(parsed, "root")
        return valid, invalid

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as error:
            invalid.append(
                {
                    "raw": stripped,
                    "failure_reasons": [f"JSON_PARSE_FAILED at line {line_number}: {error}"],
                }
            )
            continue
        add_json_value(item, f"line {line_number}")

    return valid, invalid


def get_input_payload(sample: dict[str, Any]) -> dict[str, Any]:
    value = sample.get("input")
    return value if isinstance(value, dict) else {}


def get_battle_input(sample: dict[str, Any]) -> dict[str, Any]:
    input_payload = get_input_payload(sample)
    value = input_payload.get("input")
    return value if isinstance(value, dict) else {}


def get_area(sample: dict[str, Any]) -> dict[str, Any]:
    battle_input = get_battle_input(sample)
    value = battle_input.get("area_situation")
    return value if isinstance(value, dict) else {}


def get_units(sample: dict[str, Any], side: str) -> list[dict[str, Any]]:
    area = get_area(sample)
    value = area.get(side)
    return [unit for unit in value if isinstance(unit, dict)] if isinstance(value, list) else []


def get_unit_by_id(sample: dict[str, Any], unit_id: str) -> Optional[dict[str, Any]]:
    for side in ("allies", "enemies"):
        for unit in get_units(sample, side):
            if unit.get("unitId") == unit_id:
                return unit
    return None


def get_unit_side(sample: dict[str, Any], unit_id: str) -> Optional[str]:
    for unit in get_units(sample, "allies"):
        if unit.get("unitId") == unit_id:
            return "ally"
    for unit in get_units(sample, "enemies"):
        if unit.get("unitId") == unit_id:
            return "enemy"
    return None


def unit_id_set(sample: dict[str, Any], side: str) -> set[str]:
    return {unit.get("unitId") for unit in get_units(sample, side) if isinstance(unit.get("unitId"), str)}


def all_unit_ids(sample: dict[str, Any]) -> set[str]:
    return unit_id_set(sample, "allies") | unit_id_set(sample, "enemies")


def is_alive(unit: Optional[dict[str, Any]]) -> bool:
    return bool(unit and unit.get("isAlive") is True)


def is_targetable(unit: Optional[dict[str, Any]]) -> bool:
    return bool(unit and unit.get("canBeTargeted") is True)


def actor_has_skill(sample: dict[str, Any], actor_id: str) -> bool:
    unit = get_unit_by_id(sample, actor_id)
    return bool(unit and isinstance(unit.get("skillDescription"), str) and unit.get("skillDescription"))


def actor_skill_description(sample: dict[str, Any], actor_id: str) -> Optional[str]:
    unit = get_unit_by_id(sample, actor_id)
    if not unit:
        return None
    value = unit.get("skillDescription")
    return value if isinstance(value, str) and value else None


def can_actor_skill_target_dead(sample: dict[str, Any], actor_id: str) -> bool:
    unit = get_unit_by_id(sample, actor_id)
    return bool(unit and unit.get("canSkillTargetDead") is True)


def is_valid_skill_target(sample: dict[str, Any], actor_id: str, target_id: str) -> bool:
    actor = get_unit_by_id(sample, actor_id)
    target = get_unit_by_id(sample, target_id)
    if actor is None or target is None:
        return False

    target_side = get_unit_side(sample, target_id)
    is_self_skill = actor.get("IsSkillOnSelf") is True
    is_other_ally_skill = actor.get("IsSkillOnOtherAlly") is True

    if is_self_skill:
        if target_id != actor_id:
            return False
    elif is_other_ally_skill:
        if target_id == actor_id or target_side != "ally":
            return False
    else:
        if target_side != "enemy":
            return False

    if not is_targetable(target):
        return False
    if is_alive(target):
        return True
    return can_actor_skill_target_dead(sample, actor_id)


def output_action_entries(sample: dict[str, Any]) -> list[dict[str, Any]]:
    output = sample.get("output")
    if not isinstance(output, dict):
        return []
    action = output.get("action")
    return [entry for entry in action if isinstance(entry, dict)] if isinstance(action, list) else []


def iter_sequence_items(sample: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for action_entry in output_action_entries(sample):
        actor_id = action_entry.get("unitId")
        sequence = action_entry.get("sequence")
        if not isinstance(actor_id, str) or not isinstance(sequence, list):
            continue
        for seq_item in sequence:
            if isinstance(seq_item, dict):
                items.append((actor_id, seq_item))
    return items


def find_forbidden_raw_keys(value: Any, path: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in RAW_FORBIDDEN_KEYS:
                found.append(child_path)
            found.extend(find_forbidden_raw_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_forbidden_raw_keys(child, f"{path}[{index}]"))
    return found


def validate_master_shape(ctx: ValidationContext) -> None:
    sample = ctx.sample
    for key in ["id", "split", "command_spec", "metadata", "skill_case", "gold", "input", "output"]:
        if key not in sample:
            ctx.add("MISSING_MASTER_FIELD", key)

    for forbidden_path in find_forbidden_raw_keys(sample):
        ctx.add("RAW_FORBIDDEN_FIELD_PRESENT", forbidden_path)

    if not isinstance(sample.get("id"), str) or not sample.get("id"):
        ctx.add("INVALID_SAMPLE_ID")
    if sample.get("split") not in {"train", "validation", "test"}:
        ctx.add("INVALID_SPLIT")
    if not isinstance(sample.get("command_spec"), dict):
        ctx.add("COMMAND_SPEC_NOT_OBJECT")
    if not isinstance(sample.get("gold"), dict):
        ctx.add("GOLD_NOT_OBJECT")
    if not isinstance(get_input_payload(sample), dict):
        ctx.add("INPUT_NOT_OBJECT")

    command_spec = sample.get("command_spec")
    battle_input = get_battle_input(sample)
    if isinstance(command_spec, dict):
        command_text = command_spec.get("command_text")
        if not isinstance(command_text, str) or not command_text:
            ctx.add("COMMAND_TEXT_MISSING")
        slots = command_spec.get("slots")
        if not isinstance(slots, dict):
            ctx.add("COMMAND_SPEC_SLOTS_MISSING_OR_NOT_OBJECT")

        input_command = battle_input.get("command") if isinstance(battle_input, dict) else None
        if isinstance(command_text, str) and isinstance(input_command, str) and command_text != input_command:
            ctx.add("COMMAND_TEXT_INPUT_COMMAND_MISMATCH")

    if not battle_input:
        ctx.add("RUNTIME_INPUT_MISSING")
    elif not isinstance(battle_input.get("command"), str) or not battle_input.get("command"):
        ctx.add("INPUT_COMMAND_MISSING")

    area = get_area(sample)
    if not area:
        ctx.add("AREA_SITUATION_MISSING")
    else:
        if not isinstance(area.get("allies"), list):
            ctx.add("ALLIES_MISSING")
        if not isinstance(area.get("enemies"), list):
            ctx.add("ENEMIES_MISSING")


def validate_taxonomy(ctx: ValidationContext) -> None:
    for error in validate_metadata_against_taxonomy(ctx.sample, ctx.taxonomy):
        ctx.add("TAXONOMY_INVALID", error)

    skill_case = ctx.sample.get("skill_case")
    if isinstance(skill_case, dict):
        if not isinstance(skill_case.get("is_skill_aoe"), bool):
            ctx.add("SKILL_CASE_AOE_NOT_BOOL")
        if not isinstance(skill_case.get("can_skill_target_dead"), bool):
            ctx.add("SKILL_CASE_CAN_TARGET_DEAD_NOT_BOOL")

    has_skill_action = any(
        seq_item.get("type") == "skill"
        for _, seq_item in iter_sequence_items(ctx.sample)
    )
    if has_skill_action and not isinstance(skill_case, dict):
        ctx.add("SKILL_ACTION_REQUIRES_SKILL_CASE")


def validate_unit_common_fields(ctx: ValidationContext, unit: dict[str, Any], side: str) -> None:
    unit_id = unit.get("unitId")
    label = unit_id if isinstance(unit_id, str) else f"{side}_unknown"

    for field_name in ["isAlive", "canBeTargeted", "isRanged"]:
        if not isinstance(unit.get(field_name), bool):
            ctx.add("UNIT_BOOL_FIELD_INVALID", f"{label}.{field_name}")

    hp_ratio = unit.get("hpRatio")
    if isinstance(hp_ratio, bool) or not isinstance(hp_ratio, (int, float)):
        ctx.add("UNIT_HP_RATIO_NOT_NUMBER", label)
    elif hp_ratio < 0 or hp_ratio > 1:
        ctx.add("UNIT_HP_RATIO_OUT_OF_RANGE", f"{label}={hp_ratio}")

    attack_ratio = unit.get("attackRatioToAvg")
    if isinstance(attack_ratio, bool) or not isinstance(attack_ratio, (int, float)):
        ctx.add("UNIT_ATTACK_RATIO_NOT_NUMBER", label)
    elif attack_ratio <= 0:
        ctx.add("UNIT_ATTACK_RATIO_NOT_POSITIVE", f"{label}={attack_ratio}")

    engaged_count = unit.get("engagedByOpponentCount")
    if isinstance(engaged_count, bool) or not isinstance(engaged_count, int):
        ctx.add("UNIT_ENGAGED_COUNT_NOT_INT", label)
    elif engaged_count < 0:
        ctx.add("UNIT_ENGAGED_COUNT_NEGATIVE", f"{label}={engaged_count}")

    role = unit.get("teamFormationRole")
    if role not in TEAM_FORMATION_ROLES:
        ctx.add("UNIT_TEAM_FORMATION_ROLE_INVALID", f"{label}.{role}")


def validate_unit_key_sets(ctx: ValidationContext) -> None:
    allies_raw = get_area(ctx.sample).get("allies")
    enemies_raw = get_area(ctx.sample).get("enemies")
    if not isinstance(allies_raw, list) or not isinstance(enemies_raw, list):
        return

    if len(allies_raw) != 6:
        ctx.add("ALLY_COUNT_NOT_SIX", str(len(allies_raw)))
    if len(enemies_raw) != 6:
        ctx.add("ENEMY_COUNT_NOT_SIX", str(len(enemies_raw)))

    allies = get_units(ctx.sample, "allies")
    enemies = get_units(ctx.sample, "enemies")
    if len(allies) != len(allies_raw):
        ctx.add("ALLY_ITEM_NOT_OBJECT")
    if len(enemies) != len(enemies_raw):
        ctx.add("ENEMY_ITEM_NOT_OBJECT")

    ally_ids = unit_id_set(ctx.sample, "allies")
    enemy_ids = unit_id_set(ctx.sample, "enemies")
    if ally_ids != EXPECTED_ALLY_IDS:
        ctx.add("ALLY_IDS_NOT_EXACT_SIX", f"expected={sorted(EXPECTED_ALLY_IDS)}, actual={sorted(ally_ids)}")
    if enemy_ids != EXPECTED_ENEMY_IDS:
        ctx.add("ENEMY_IDS_NOT_EXACT_SIX", f"expected={sorted(EXPECTED_ENEMY_IDS)}, actual={sorted(enemy_ids)}")
    if ally_ids & enemy_ids:
        ctx.add("UNIT_ID_DUPLICATED_BETWEEN_SIDES", str(sorted(ally_ids & enemy_ids)))

    for side, units, expected_fields in [
        ("ally", allies, ALLY_UNIT_FIELDS),
        ("enemy", enemies, ENEMY_UNIT_FIELDS),
    ]:
        seen_ids: set[str] = set()
        for unit in units:
            unit_id = unit.get("unitId")
            label = unit_id if isinstance(unit_id, str) else f"{side}_unknown"
            if isinstance(unit_id, str):
                if unit_id in seen_ids:
                    ctx.add("UNIT_ID_DUPLICATED", unit_id)
                seen_ids.add(unit_id)
            else:
                ctx.add("UNIT_ID_NOT_STRING", side)

            keys = set(unit.keys())
            missing = expected_fields - keys
            extra = keys - expected_fields
            
            if missing:
                ctx.add("UNIT_FIELD_MISSING", f"{label}: {sorted(missing)}")
            if extra:
                ctx.add("UNIT_FIELD_UNEXPECTED", f"{label}: {sorted(extra)}")
            

            validate_unit_common_fields(ctx, unit, side)

            if side == "ally":
                if not isinstance(unit.get("skillDescription"), str):
                    ctx.add("ALLY_SKILL_DESCRIPTION_NOT_STRING", label)
                for field_name in ["IsSkillOnSelf", "IsSkillOnOtherAlly", "isSkillAoe", "canSkillTargetDead"]:
                    if not isinstance(unit.get(field_name), bool):
                        ctx.add("ALLY_SKILL_BOOL_FIELD_INVALID", f"{label}.{field_name}")


def validate_single_distance_pair(
    ctx: ValidationContext,
    unit_id: str,
    closest_key: str,
    farthest_key: str,
    candidates: set[str],
) -> None:
    unit = get_unit_by_id(ctx.sample, unit_id)
    if unit is None:
        return

    closest = unit.get(closest_key)
    farthest = unit.get(farthest_key)
    pair_label = f"{unit_id}.{closest_key}/{farthest_key}"

    if isinstance(closest, list) or isinstance(farthest, list):
        ctx.add("DISTANCE_FIELD_MUST_NOT_BE_ARRAY", pair_label)
        return

    if closest is not None and not isinstance(closest, str):
        ctx.add("DISTANCE_FIELD_NOT_STRING_OR_NULL", f"{unit_id}.{closest_key}")
    if farthest is not None and not isinstance(farthest, str):
        ctx.add("DISTANCE_FIELD_NOT_STRING_OR_NULL", f"{unit_id}.{farthest_key}")

    if not candidates:
        if closest is not None or farthest is not None:
            ctx.add("DISTANCE_FIELDS_SHOULD_BE_NULL_WITH_NO_CANDIDATE", pair_label)
        return

    if closest not in candidates:
        ctx.add("CLOSEST_DISTANCE_FIELD_INVALID_TARGET", f"{unit_id}.{closest_key}={closest}")
    if farthest not in candidates:
        ctx.add("FARTHEST_DISTANCE_FIELD_INVALID_TARGET", f"{unit_id}.{farthest_key}={farthest}")

    if len(candidates) == 1:
        only = next(iter(candidates))
        if closest != only or farthest != only:
            ctx.add("DISTANCE_FIELDS_SINGLE_CANDIDATE_MISMATCH", f"{pair_label}, expected={only}")
        return

    if closest == farthest:
        ctx.add("DISTANCE_FIELDS_MULTI_CANDIDATE_SHOULD_DIFFER", pair_label)


def validate_ally_distance_fields(ctx: ValidationContext) -> None:
    ally_ids = unit_id_set(ctx.sample, "allies")
    enemy_target_candidates = {
        unit.get("unitId")
        for unit in get_units(ctx.sample, "enemies")
        if isinstance(unit.get("unitId"), str) and is_alive(unit) and is_targetable(unit)
    }
    alive_ally_ids = {
        unit.get("unitId")
        for unit in get_units(ctx.sample, "allies")
        if isinstance(unit.get("unitId"), str) and is_alive(unit)
    }

    for ally_id in sorted(alive_ally_ids):
        validate_single_distance_pair(
            ctx=ctx,
            unit_id=ally_id,
            closest_key="closestTargetableOpponent",
            farthest_key="farthestTargetableOpponent",
            candidates=enemy_target_candidates,
        )
        validate_single_distance_pair(
            ctx=ctx,
            unit_id=ally_id,
            closest_key="closestAliveAlly",
            farthest_key="farthestAliveAlly",
            candidates=alive_ally_ids - {ally_id},
        )


def validate_area_situation(ctx: ValidationContext) -> None:
    validate_unit_key_sets(ctx)
    validate_ally_distance_fields(ctx)


def get_alive_allies(sample: dict[str, Any]) -> list[str]:
    return [
        unit["unitId"]
        for unit in get_units(sample, "allies")
        if isinstance(unit.get("unitId"), str) and is_alive(unit)
    ]


def get_valid_attack_targets(sample: dict[str, Any]) -> list[str]:
    return [
        unit["unitId"]
        for unit in get_units(sample, "enemies")
        if isinstance(unit.get("unitId"), str) and is_alive(unit) and is_targetable(unit)
    ]


def get_dead_targetable_allies(sample: dict[str, Any]) -> list[str]:
    return [
        unit["unitId"]
        for unit in get_units(sample, "allies")
        if isinstance(unit.get("unitId"), str) and not is_alive(unit) and is_targetable(unit)
    ]


def collect_invalid_runtime_units(sample: dict[str, Any]) -> list[dict[str, str]]:
    invalid_units: list[dict[str, str]] = []

    for unit in get_units(sample, "allies"):
        unit_id = unit.get("unitId")
        if not isinstance(unit_id, str):
            continue
        if not is_alive(unit):
            invalid_units.append({"unitId": unit_id, "side": "ally", "reason": "dead"})

    for unit in get_units(sample, "enemies"):
        unit_id = unit.get("unitId")
        if not isinstance(unit_id, str):
            continue
        reasons: list[str] = []
        if not is_alive(unit):
            reasons.append("dead")
        if not is_targetable(unit):
            reasons.append("untargetable")
        if reasons:
            invalid_units.append({"unitId": unit_id, "side": "enemy", "reason": "+".join(reasons)})

    return invalid_units


def build_command_analysis(sample: dict[str, Any]) -> dict[str, Any]:
    alive_allies = get_alive_allies(sample)
    valid_attack_targets = get_valid_attack_targets(sample)
    dead_targetable_allies = get_dead_targetable_allies(sample)
    valid_move_to_units = alive_allies + valid_attack_targets
    invalid_units = collect_invalid_runtime_units(sample)

    return {
        "analysisMode": "runtime_constraint_summary",
        "description": (
            "This object summarizes runtime-valid actors, targets, move destinations, "
            "and dead targetable allies. It does not parse or decide the user's intent."
        ),
        "allowedActors": alive_allies,
        "allowedAttackTargets": valid_attack_targets,
        "validMoveToUnits": valid_move_to_units,
        "deadAllies": dead_targetable_allies,
        "invalidUnits": invalid_units,
        "actionPolicy": {
            "maxActionsPerActor": MAX_ACTIONS_PER_ACTOR,
            "moveToKey": "to",
            "allowedMoveSubtypes": sorted(ALLOWED_MOVE_SUBTYPES),
            "movementTypes": ["direct", "flank"],
            "waitDurationMinSec": MIN_WAIT_SECONDS,
            "waitDurationMaxSec": MAX_WAIT_SECONDS,
            "defaultWaitDurationSec": DEFAULT_WAIT_SECONDS,
            "skillControlDurationMinSec": MIN_WAIT_SECONDS,
            "skillControlDurationMaxSec": MAX_WAIT_SECONDS,
            "defaultSkillControlDeferDurationSec": DEFAULT_SKILL_CONTROL_DEFER_SECONDS,
            "conditionMode": "current_state_only",
        },
    }


def validate_output_schema(ctx: ValidationContext) -> None:
    sample = ctx.sample
    output = sample.get("output")
    if not isinstance(output, dict):
        ctx.add("OUTPUT_NOT_OBJECT")
        return

    if set(output.keys()) != EXPECTED_OUTPUT_KEYS:
        ctx.add("TOP_LEVEL_OUTPUT_KEYS_INVALID", str(sorted(output.keys())))
    if not isinstance(output.get("thinking"), str):
        ctx.add("THINKING_NOT_STRING")

    dialog = output.get("dialog")
    if not isinstance(dialog, list):
        ctx.add("DIALOG_NOT_ARRAY")
        dialog = []

    action = output.get("action")
    if not isinstance(action, list):
        ctx.add("ACTION_NOT_ARRAY")
        action = []

    seen_action_actors: set[str] = set()
    for action_entry in action:
        if not isinstance(action_entry, dict):
            ctx.add("ACTION_ITEM_NOT_OBJECT")
            continue
        if set(action_entry.keys()) != {"unitId", "sequence"}:
            ctx.add("INVALID_ACTION_KEYS", str(sorted(action_entry.keys())))

        actor_id = action_entry.get("unitId")
        sequence = action_entry.get("sequence")
        if not isinstance(actor_id, str):
            ctx.add("ACTION_UNIT_ID_NOT_STRING")
        else:
            if actor_id in seen_action_actors:
                ctx.add("ACTION_DUPLICATED_ACTOR", actor_id)
            seen_action_actors.add(actor_id)

        if not isinstance(sequence, list):
            ctx.add("SEQUENCE_NOT_ARRAY", str(actor_id))
            continue
        if len(sequence) > MAX_ACTIONS_PER_ACTOR:
            ctx.add("SEQUENCE_TOO_LONG", str(actor_id))

        for seq_item in sequence:
            if not isinstance(seq_item, dict):
                ctx.add("SEQUENCE_ITEM_NOT_OBJECT", str(actor_id))
                continue
            validate_sequence_schema(ctx, actor_id if isinstance(actor_id, str) else "", seq_item)

    for dialog_entry in dialog:
        if not isinstance(dialog_entry, dict):
            ctx.add("DIALOG_ITEM_NOT_OBJECT")
            continue
        if set(dialog_entry.keys()) != {"unitId", "text"}:
            ctx.add("INVALID_DIALOG_KEYS", str(sorted(dialog_entry.keys())))
        if not isinstance(dialog_entry.get("unitId"), str):
            ctx.add("DIALOG_UNIT_ID_NOT_STRING")
        if not isinstance(dialog_entry.get("text"), str):
            ctx.add("DIALOG_TEXT_NOT_STRING")


def validate_sequence_schema(ctx: ValidationContext, actor_id: str, seq_item: dict[str, Any]) -> None:
    action_type = seq_item.get("type")
    if action_type not in ALLOWED_ACTION_TYPES:
        ctx.add("UNKNOWN_ACTION_TYPE", str(action_type))
        return

    if action_type == "move":
        if set(seq_item.keys()) != {"type", "subtype", "movementType", "to"}:
            ctx.add("INVALID_MOVE_KEYS", str(sorted(seq_item.keys())))
        if seq_item.get("subtype") not in ALLOWED_MOVE_SUBTYPES:
            ctx.add("INVALID_MOVE_SUBTYPE", str(seq_item.get("subtype")))
        if seq_item.get("movementType") not in ALLOWED_MOVEMENT_TYPES:
            ctx.add("INVALID_MOVEMENT_TYPE", str(seq_item.get("movementType")))
        if not isinstance(seq_item.get("to"), str):
            ctx.add("MOVE_TO_NOT_STRING", actor_id)
        return

    if action_type == "attack":
        if set(seq_item.keys()) != {"type", "target"}:
            ctx.add("INVALID_ATTACK_KEYS", str(sorted(seq_item.keys())))
        if not isinstance(seq_item.get("target"), str):
            ctx.add("ATTACK_TARGET_NOT_STRING", actor_id)
        return

    if action_type == "skill":
        if set(seq_item.keys()) != {"type", "description", "target"}:
            ctx.add("INVALID_SKILL_KEYS", str(sorted(seq_item.keys())))
        if not isinstance(seq_item.get("description"), str):
            ctx.add("SKILL_DESCRIPTION_NOT_STRING", actor_id)
        if not isinstance(seq_item.get("target"), str):
            ctx.add("SKILL_TARGET_NOT_STRING", actor_id)
        return

    if action_type == "wait":
        if set(seq_item.keys()) != {"type", "durationSec"}:
            ctx.add("INVALID_WAIT_KEYS", str(sorted(seq_item.keys())))
        duration = seq_item.get("durationSec")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            ctx.add("WAIT_DURATION_NOT_NUMBER", actor_id)
        elif duration < MIN_WAIT_SECONDS or duration > MAX_WAIT_SECONDS:
            ctx.add("WAIT_DURATION_OUT_OF_RANGE", str(duration))
        return

    if action_type == "skillControl":
        mode = seq_item.get("mode")
        if mode == "defer":
            if set(seq_item.keys()) != {"type", "mode", "durationSec"}:
                ctx.add("INVALID_SKILL_CONTROL_DEFER_KEYS", str(sorted(seq_item.keys())))
            duration = seq_item.get("durationSec")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                ctx.add("SKILL_CONTROL_DURATION_NOT_NUMBER", actor_id)
            elif duration < MIN_WAIT_SECONDS or duration > MAX_WAIT_SECONDS:
                ctx.add("SKILL_CONTROL_DURATION_OUT_OF_RANGE", str(duration))
        elif mode == "forbid":
            if set(seq_item.keys()) != {"type", "mode"}:
                ctx.add("INVALID_SKILL_CONTROL_FORBID_KEYS", str(sorted(seq_item.keys())))
        else:
            ctx.add("INVALID_SKILL_CONTROL_MODE", str(mode))


def validate_runtime(ctx: ValidationContext) -> None:
    sample = ctx.sample
    command_analysis = ctx.command_analysis or build_command_analysis(sample)

    ally_ids = unit_id_set(sample, "allies")
    enemy_ids = unit_id_set(sample, "enemies")
    all_ids = all_unit_ids(sample)
    allowed_actors = set(command_analysis.get("allowedActors", []))
    allowed_attack_targets = set(command_analysis.get("allowedAttackTargets", []))
    valid_move_to_units = set(command_analysis.get("validMoveToUnits", []))

    for action_entry in output_action_entries(sample):
        actor_id = action_entry.get("unitId")
        if not isinstance(actor_id, str):
            continue

        actor_unit = get_unit_by_id(sample, actor_id)
        if actor_id not in ally_ids:
            ctx.add("ACTOR_NOT_ALLY", actor_id)
        if not is_alive(actor_unit):
            ctx.add("ACTOR_DEAD", actor_id)
        if actor_id not in allowed_actors:
            ctx.add("ACTOR_OUTSIDE_ALLOWED_ACTORS", actor_id)

        sequence = action_entry.get("sequence")
        if not isinstance(sequence, list):
            continue

        for seq_item in sequence:
            if not isinstance(seq_item, dict):
                continue
            action_type = seq_item.get("type")

            if action_type == "move":
                to_id = seq_item.get("to")
                if not isinstance(to_id, str):
                    continue
                if to_id == actor_id:
                    ctx.add("MOVE_TO_SELF", actor_id)
                if to_id not in all_ids:
                    ctx.add("MOVE_TO_NOT_FOUND", to_id)
                if to_id not in valid_move_to_units:
                    ctx.add("MOVE_TO_OUTSIDE_VALID_MOVE_TO_UNITS", to_id)

                target_unit = get_unit_by_id(sample, to_id)
                target_side = get_unit_side(sample, to_id)
                if target_side == "enemy" and (not is_alive(target_unit) or not is_targetable(target_unit)):
                    ctx.add("MOVE_TO_INVALID_ENEMY", to_id)
                if target_side == "ally" and not is_alive(target_unit):
                    ctx.add("MOVE_TO_DEAD_ALLY", to_id)

            elif action_type == "attack":
                target = seq_item.get("target")
                if not isinstance(target, str):
                    continue
                target_unit = get_unit_by_id(sample, target)
                if target not in enemy_ids:
                    ctx.add("ATTACK_TARGET_NOT_ENEMY", target)
                if not is_alive(target_unit):
                    ctx.add("ATTACK_TARGET_DEAD", target)
                if not is_targetable(target_unit):
                    ctx.add("ATTACK_TARGET_UNTARGETABLE", target)
                if target not in allowed_attack_targets:
                    ctx.add("ATTACK_TARGET_OUTSIDE_ALLOWED_TARGETS", target)

            elif action_type == "skill":
                target = seq_item.get("target")
                description = seq_item.get("description")
                if not actor_has_skill(sample, actor_id):
                    ctx.add("SKILL_ACTOR_HAS_NO_SKILL", actor_id)
                    continue

                expected_description = actor_skill_description(sample, actor_id)
                if description != expected_description:
                    ctx.add("SKILL_DESCRIPTION_MISMATCH", actor_id)
                if not isinstance(target, str) or target not in all_ids:
                    ctx.add("SKILL_TARGET_NOT_FOUND", str(target))
                elif not is_valid_skill_target(sample, actor_id, target):
                    ctx.add("SKILL_TARGET_INVALID", f"{actor_id}->{target}")

            elif action_type == "skillControl":
                if not actor_has_skill(sample, actor_id):
                    ctx.add("SKILL_CONTROL_ACTOR_HAS_NO_SKILL", actor_id)


def validate_policy(ctx: ValidationContext) -> None:
    sample = ctx.sample
    action_actor_ids: list[str] = []

    for action_entry in output_action_entries(sample):
        actor_id = action_entry.get("unitId")
        if isinstance(actor_id, str):
            action_actor_ids.append(actor_id)
        sequence = action_entry.get("sequence")
        if not isinstance(sequence, list):
            continue

        for index, seq_item in enumerate(sequence):
            if not isinstance(seq_item, dict):
                continue
            if seq_item.get("type") == "wait" and index > 0:
                previous = sequence[index - 1]
                if isinstance(previous, dict) and previous.get("type") == "attack":
                    ctx.add("WAIT_AFTER_ATTACK", str(actor_id))
                if isinstance(previous, dict) and previous.get("type") == "skill":
                    ctx.add("WAIT_AFTER_SKILL", str(actor_id))

    output = sample.get("output")
    if not isinstance(output, dict):
        return
    dialog = output.get("dialog")
    if not isinstance(dialog, list):
        return

    action_actor_set = set(action_actor_ids)
    dialog_ids: list[str] = []
    dialog_texts: list[str] = []
    for dialog_entry in dialog:
        if not isinstance(dialog_entry, dict):
            continue
        unit_id = dialog_entry.get("unitId")
        text = dialog_entry.get("text")
        if isinstance(unit_id, str):
            dialog_ids.append(unit_id)
            if unit_id not in action_actor_set:
                ctx.add("DIALOG_UNIT_MISMATCH", unit_id)
        if isinstance(text, str):
            dialog_texts.append(text)

    if not action_actor_ids and dialog:
        ctx.add("EMPTY_ACTION_DIALOG_NOT_EMPTY")

    for actor_id in action_actor_set:
        if dialog_ids.count(actor_id) != 1:
            ctx.add("DIALOG_MISSING_OR_DUPLICATED_FOR_ACTOR", actor_id)

    for dialog_id in set(dialog_ids):
        if dialog_ids.count(dialog_id) > 1:
            ctx.add("DIALOG_DUPLICATED_UNIT", dialog_id)

    if len(action_actor_set) > 1:
        for text in set(dialog_texts):
            if text and dialog_texts.count(text) > 1:
                ctx.add("DIALOG_TEXT_DUPLICATED_ACROSS_ACTORS", text)


def sequence_action_types(sample: dict[str, Any]) -> list[str]:
    return [seq_item.get("type") for _, seq_item in iter_sequence_items(sample) if isinstance(seq_item.get("type"), str)]


def used_targets(sample: dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    for _, seq_item in iter_sequence_items(sample):
        for key in ["target", "to"]:
            value = seq_item.get(key)
            if isinstance(value, str):
                targets.add(value)
    return targets


def list_key_as_set(value: dict[str, Any], key: str) -> tuple[bool, set[str]]:
    if key not in value:
        return False, set()
    item = value.get(key)
    if not isinstance(item, list):
        return True, set()
    return True, {entry for entry in item if isinstance(entry, str)}


def validate_semantic_gold(ctx: ValidationContext) -> None:
    sample = ctx.sample
    gold = sample.get("gold")
    if not isinstance(gold, dict):
        return
    
    expected_action_pattern = gold.get("expected_action_pattern")
    allowed_action_patterns = set(ctx.taxonomy.get("orders", {}).get("action_pattern", []))

    if not isinstance(expected_action_pattern, str):
        ctx.add("GOLD_EXPECTED_ACTION_PATTERN_NOT_STRING")
    elif expected_action_pattern not in allowed_action_patterns:
        ctx.add("GOLD_EXPECTED_ACTION_PATTERN_UNKNOWN", str(expected_action_pattern))

    metadata = sample.get("metadata")
    metadata_action_pattern = metadata.get("action_pattern") if isinstance(metadata, dict) else None

    if (
        isinstance(metadata_action_pattern, str)
        and isinstance(expected_action_pattern, str)
        and metadata_action_pattern != expected_action_pattern
    ):
        ctx.add(
            "GOLD_ACTION_PATTERN_METADATA_MISMATCH",
            f"metadata={metadata_action_pattern} gold={expected_action_pattern}",
        )

    action_actors = [entry.get("unitId") for entry in output_action_entries(sample) if isinstance(entry.get("unitId"), str)]
    action_actor_set = set(action_actors)
    action_type_set = set(sequence_action_types(sample))
    target_set = used_targets(sample)

    _, required_actors = list_key_as_set(gold, "required_actors")
    allowed_actors_present, allowed_actors = list_key_as_set(gold, "allowed_actors")
    _, forbidden_actors = list_key_as_set(gold, "forbidden_actors")

    if required_actors and not required_actors.issubset(action_actor_set):
        ctx.add("SEMANTIC_ACTOR_MISMATCH", f"missing={sorted(required_actors - action_actor_set)}")
    if allowed_actors_present and not action_actor_set.issubset(allowed_actors):
        ctx.add("SEMANTIC_ACTOR_OUTSIDE_ALLOWED", f"extra={sorted(action_actor_set - allowed_actors)}")
    if forbidden_actors & action_actor_set:
        ctx.add("SEMANTIC_FORBIDDEN_ACTOR_USED", f"used={sorted(forbidden_actors & action_actor_set)}")

    _, required_types = list_key_as_set(gold, "required_action_types")
    allowed_types_present, allowed_types = list_key_as_set(gold, "allowed_action_types")
    _, forbidden_types = list_key_as_set(gold, "forbidden_action_types")

    if required_types and not required_types.issubset(action_type_set):
        ctx.add("SEMANTIC_ACTION_TYPE_MISMATCH", f"missing={sorted(required_types - action_type_set)}")
    if allowed_types_present and not action_type_set.issubset(allowed_types):
        ctx.add("SEMANTIC_ACTION_TYPE_OUTSIDE_ALLOWED", f"extra={sorted(action_type_set - allowed_types)}")
    if forbidden_types & action_type_set:
        ctx.add("SEMANTIC_FORBIDDEN_ACTION_TYPE_USED", f"used={sorted(forbidden_types & action_type_set)}")

    empty_action_allowed = gold.get("empty_action_allowed")
    if empty_action_allowed is False and not action_actors:
        ctx.add("SEMANTIC_UNEXPECTED_EMPTY_ACTION")
    if empty_action_allowed is True and action_actors and gold.get("expected_action_pattern") == "empty_action_expected":
        ctx.add("SEMANTIC_EMPTY_ACTION_MISMATCH")

    targets = gold.get("targets")
    if isinstance(targets, dict):
        _, required_targets = list_key_as_set(targets, "required")
        allowed_targets_present, allowed_targets = list_key_as_set(targets, "allowed")
        _, forbidden_targets = list_key_as_set(targets, "forbidden")

        if required_targets and not required_targets.issubset(target_set):
            ctx.add("SEMANTIC_TARGET_MISMATCH", f"missing={sorted(required_targets - target_set)}")
        if allowed_targets_present and not target_set.issubset(allowed_targets):
            ctx.add("SEMANTIC_TARGET_OUTSIDE_ALLOWED", f"extra={sorted(target_set - allowed_targets)}")
        if forbidden_targets & target_set:
            ctx.add("SEMANTIC_FORBIDDEN_TARGET_USED", f"used={sorted(forbidden_targets & target_set)}")

    validate_gold_action_details(ctx, gold)


def validate_gold_action_details(ctx: ValidationContext, gold: dict[str, Any]) -> None:
    for actor_id, seq_item in iter_sequence_items(ctx.sample):
        action_type = seq_item.get("type")

        if action_type == "move" and isinstance(gold.get("move"), dict):
            move_gold = gold["move"]
            if move_gold.get("actor") and actor_id != move_gold.get("actor"):
                continue
            if move_gold.get("required_subtype") and seq_item.get("subtype") != move_gold.get("required_subtype"):
                ctx.add("SEMANTIC_MOVE_SUBTYPE_MISMATCH", actor_id)
            if move_gold.get("required_to") and seq_item.get("to") != move_gold.get("required_to"):
                ctx.add("SEMANTIC_MOVE_TO_MISMATCH", actor_id)

        if action_type == "attack" and isinstance(gold.get("attack"), dict):
            attack_gold = gold["attack"]
            if attack_gold.get("actor") and actor_id != attack_gold.get("actor"):
                continue
            if attack_gold.get("required_target") and seq_item.get("target") != attack_gold.get("required_target"):
                ctx.add("SEMANTIC_ATTACK_TARGET_MISMATCH", actor_id)

        if action_type == "skill" and isinstance(gold.get("skill"), dict):
            skill_gold = gold["skill"]
            if skill_gold.get("actor") and actor_id != skill_gold.get("actor"):
                continue
            if skill_gold.get("required_target") and seq_item.get("target") != skill_gold.get("required_target"):
                ctx.add("SEMANTIC_SKILL_TARGET_MISMATCH", actor_id)
            if skill_gold.get("description_exact") and seq_item.get("description") != skill_gold.get("description_exact"):
                ctx.add("SEMANTIC_SKILL_DESCRIPTION_MISMATCH", actor_id)

        if action_type == "skillControl" and isinstance(gold.get("skillControl"), dict):
            control_gold = gold["skillControl"]
            if control_gold.get("actor") and actor_id != control_gold.get("actor"):
                continue
            if control_gold.get("required_mode") and seq_item.get("mode") != control_gold.get("required_mode"):
                ctx.add("SEMANTIC_SKILL_CONTROL_MODE_MISMATCH", actor_id)


def validate_sample_with_analysis(sample: dict[str, Any], taxonomy: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    ctx = ValidationContext(sample, taxonomy)
    validate_master_shape(ctx)
    validate_taxonomy(ctx)
    validate_area_situation(ctx)

    if not ctx.errors:
        ctx.command_analysis = build_command_analysis(sample)

    validate_output_schema(ctx)
    validate_runtime(ctx)
    validate_policy(ctx)
    validate_semantic_gold(ctx)
    return ctx.errors, ctx.command_analysis


def validate_sample(sample: dict[str, Any], taxonomy: dict[str, Any]) -> list[str]:
    errors, _ = validate_sample_with_analysis(sample, taxonomy)
    return errors


def normalize_sample_with_result(
    sample: dict[str, Any],
    passed: bool,
    errors: list[str],
    command_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(sample)
    if passed and command_analysis is not None:
        input_payload = result.setdefault("input", {})
        if isinstance(input_payload, dict):
            input_payload["commandAnalysis"] = command_analysis
    result["validator_result"] = {
        "passed": passed,
        "failure_reasons": errors,
    }
    return result


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation_index(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / VALIDATION_INDEX_FILENAME
    if not path.exists():
        return {"validated_inputs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"validated_inputs": {}}
    if not isinstance(data, dict):
        return {"validated_inputs": {}}
    if not isinstance(data.get("validated_inputs"), dict):
        data["validated_inputs"] = {}
    return data


def save_validation_index(dataset_root: Path, index: dict[str, Any]) -> None:
    path = dataset_root / VALIDATION_INDEX_FILENAME
    path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_file(
    input_path: Path,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    write_outputs: bool = True,
) -> dict[str, Any]:
    input_hash = file_sha256(input_path)
    if write_outputs:
        validation_index = load_validation_index(dataset_root)
        validated_inputs = validation_index["validated_inputs"]
        if input_hash in validated_inputs:
            return {
                "input_path": str(input_path),
                "input_hash": input_hash,
                "skipped": True,
                "reason": "INPUT_ALREADY_VALIDATED",
                "previous_result": validated_inputs[input_hash],
            }
    else:
        validation_index = {"validated_inputs": {}}

    taxonomy = load_taxonomy(taxonomy_path)
    samples, parse_failures = read_json_records_with_errors(input_path)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for failure in parse_failures:
        rejected.append(
            {
                "id": "parse_failed",
                "raw": failure.get("raw"),
                "validator_result": {
                    "passed": False,
                    "failure_reasons": failure.get("failure_reasons", ["JSON_PARSE_FAILED"]),
                },
            }
        )

    for sample in samples:
        errors, command_analysis = validate_sample_with_analysis(sample, taxonomy)
        if errors:
            rejected.append(normalize_sample_with_result(sample, False, errors))
        else:
            accepted.append(normalize_sample_with_result(sample, True, [], command_analysis))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accepted_path = dataset_root / "accepted" / f"3accepted_{timestamp}.jsonl"
    rejected_path = dataset_root / "rejected" / f"3rejected_{timestamp}.jsonl"

    result = {
        "input_path": str(input_path),
        "input_hash": input_hash,
        "skipped": False,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_path": str(accepted_path) if accepted else None,
        "rejected_path": str(rejected_path) if rejected else None,
    }

    if write_outputs:
        append_jsonl(accepted_path, accepted)
        append_jsonl(rejected_path, rejected)
        validation_index["validated_inputs"][input_hash] = result
        save_validation_index(dataset_root, validation_index)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Teacher output JSON/JSONL file path.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = validate_file(
        input_path=Path(args.input),
        dataset_root=Path(args.dataset_root),
        taxonomy_path=Path(args.taxonomy),
        write_outputs=not args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
