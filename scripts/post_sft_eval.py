# 저장된 responses.jsonl과 accepted test JSONL을 로컬에서 채점한다.
# output 파싱, schema 검사, runtime contract 검사, SAC, gold 비교를 수행한다.
# taxonomy breakdown은 accepted sample의 metadata 중 지정된 5개 축만 사용한다.
# general path, scenario_family, edge_flags, skill_case 기준 breakdown은 생성하지 않는다.
# 서버 호출, generation, malformed examples, sample별 상세 JSON 저장은 수행하지 않는다.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_OUTPUT_KEYS = {"thinking", "dialog", "action"}
ALLOWED_MOVE_SUBTYPES = {"approachOpponent", "escape", "help", "holdFront"}
ALLOWED_MOVEMENT_TYPES = {"direct", "flank"}
ALLOWED_ACTION_TYPES = {"move", "attack", "skill", "wait", "skillControl"}
MIN_WAIT_SECONDS = 1.0
MAX_WAIT_SECONDS = 10.0
MAX_ACTIONS_PER_ACTOR = 3

GENERAL_METADATA_FIELDS = [
    "intent_family",
    "command_style",
    "actor_selection",
    "target_selection",
    "action_pattern",
]
BREAKDOWN_FIELDS = [
    "intent_family",
    "command_style",
    "actor_selection",
    "target_selection",
    "action_pattern",
]
SUMMARY_RATE_FIELDS = [
    "request_success_rate",
    "json_parse_success_rate",
    "top_level_schema_success_rate",
    "strict_schema_success_rate",
    "contract_violation_rate",
    "sac_success_rate",
    "exact_json_match_rate",
    "action_exact_match_rate",
    "semantic_or_action_valid_match_rate",
    "dialog_validity_rate",
    "empty_action_correctness_rate",
    "timeout_rate",
]
PER_ROW_FIELDS = [
    "model_name",
    "row_index",
    "request_success",
    "timeout",
    "latency_ms",
    "output_tokens",
    "json_parse_ok",
    "top_level_schema_ok",
    "strict_schema_ok",
    "contract_ok",
    "contract_violation_count",
    "sac_ok",
    "exact_json_match",
    "action_exact_match",
    "semantic_or_action_valid_match",
    "dialog_valid",
    "empty_action_correct",
    "intent_family",
    "command_style",
    "actor_selection",
    "target_selection",
    "action_pattern",
]


@dataclass
class PreparedSample:
    row_index: int
    runtime_input: dict[str, Any]
    command_analysis: dict[str, Any]
    gold_output: dict[str, Any]
    metadata: dict[str, Any]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}") from error
        if not isinstance(item, dict):
            raise ValueError(f"JSONL row must be object at {path}:{line_no}")
        records.append(item)
    return records



def validate_metadata(row_index: int, metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError(f"row {row_index}: accepted sample missing metadata object")

    missing: list[str] = []
    for field in GENERAL_METADATA_FIELDS:
        value = metadata.get(field)
        if not isinstance(value, str) or not value:
            missing.append(field)

    if missing:
        raise ValueError(f"row {row_index}: metadata is missing required taxonomy fields: {', '.join(missing)}")
    return metadata


def build_runtime_user_message_from_accepted(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_input = sample.get("input")
    if not isinstance(source_input, dict):
        raise ValueError("accepted sample missing input object")
    runtime_input = source_input.get("input")
    command_analysis = source_input.get("commandAnalysis")
    if not isinstance(runtime_input, dict):
        raise ValueError("accepted sample missing input.input object")
    if not isinstance(command_analysis, dict):
        raise ValueError("accepted sample missing input.commandAnalysis object")
    return runtime_input, command_analysis


def parse_messages_record(row_index: int, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    messages = record["messages"]
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"row {row_index}: invalid messages roles: {roles}")

    user = messages[1].get("content")
    assistant = messages[2].get("content")
    if not isinstance(user, str) or not isinstance(assistant, str):
        raise ValueError(f"row {row_index}: messages content must be strings")

    user_obj = json.loads(user)
    gold = json.loads(assistant)
    runtime_input = user_obj.get("input")
    command_analysis = user_obj.get("commandAnalysis")
    if not isinstance(runtime_input, dict) or not isinstance(command_analysis, dict):
        raise ValueError(f"row {row_index}: messages user content missing input/commandAnalysis")

    metadata = validate_metadata(row_index, record.get("metadata"))
    return runtime_input, command_analysis, gold, metadata


def prepare_samples(path: Path) -> list[PreparedSample]:
    records = read_json_records(path)
    samples: list[PreparedSample] = []

    for row_index, record in enumerate(records, start=1):
        if isinstance(record.get("messages"), list):
            # A pure SFT messages JSONL does not contain taxonomy metadata.
            # It is accepted only if metadata was explicitly preserved on the same row.
            runtime_input, command_analysis, gold, metadata = parse_messages_record(row_index, record)
        else:
            runtime_input, command_analysis = build_runtime_user_message_from_accepted(record)
            gold = record.get("output")
            if not isinstance(gold, dict):
                raise ValueError(f"row {row_index}: accepted sample missing output object")
            metadata = validate_metadata(row_index, record.get("metadata"))

        samples.append(
            PreparedSample(
                row_index=row_index,
                runtime_input=runtime_input,
                command_analysis=command_analysis,
                gold_output=gold,
                metadata=metadata,
            )
        )

    if not samples:
        raise ValueError(f"No samples found: {path}")
    return samples


def parse_model_output(raw_text: str | None) -> tuple[bool, dict[str, Any] | None]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        return False, None
    try:
        parsed = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        return False, None
    if not isinstance(parsed, dict):
        return False, None
    return True, parsed


def is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def check_output_schema(parsed: dict[str, Any] | None) -> tuple[bool, bool, list[str]]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return False, False, ["OUTPUT_NOT_OBJECT"]

    top_level_ok = set(parsed.keys()) == EXPECTED_OUTPUT_KEYS
    if not top_level_ok:
        errors.append("TOP_LEVEL_OUTPUT_KEYS_INVALID")
    if not isinstance(parsed.get("thinking"), str):
        errors.append("THINKING_NOT_STRING")

    dialog = parsed.get("dialog")
    if not isinstance(dialog, list):
        errors.append("DIALOG_NOT_ARRAY")
        dialog = []

    action = parsed.get("action")
    if not isinstance(action, list):
        errors.append("ACTION_NOT_ARRAY")
        action = []

    seen_actors: set[str] = set()
    for action_entry in action:
        if not isinstance(action_entry, dict):
            errors.append("ACTION_ITEM_NOT_OBJECT")
            continue
        if set(action_entry.keys()) != {"unitId", "sequence"}:
            errors.append("INVALID_ACTION_KEYS")
        actor_id = action_entry.get("unitId")
        sequence = action_entry.get("sequence")
        if not isinstance(actor_id, str):
            errors.append("ACTION_UNIT_ID_NOT_STRING")
        elif actor_id in seen_actors:
            errors.append("ACTION_DUPLICATED_ACTOR")
        else:
            seen_actors.add(actor_id)
        if not isinstance(sequence, list):
            errors.append("SEQUENCE_NOT_ARRAY")
            continue
        if len(sequence) > MAX_ACTIONS_PER_ACTOR:
            errors.append("SEQUENCE_TOO_LONG")
        for seq_item in sequence:
            if not isinstance(seq_item, dict):
                errors.append("SEQUENCE_ITEM_NOT_OBJECT")
                continue
            errors.extend(check_sequence_schema(seq_item))

    for dialog_entry in dialog:
        if not isinstance(dialog_entry, dict):
            errors.append("DIALOG_ITEM_NOT_OBJECT")
            continue
        if set(dialog_entry.keys()) != {"unitId", "text"}:
            errors.append("INVALID_DIALOG_KEYS")
        if not isinstance(dialog_entry.get("unitId"), str):
            errors.append("DIALOG_UNIT_ID_NOT_STRING")
        if not isinstance(dialog_entry.get("text"), str):
            errors.append("DIALOG_TEXT_NOT_STRING")

    return top_level_ok, len(errors) == 0, errors


def check_sequence_schema(seq_item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    action_type = seq_item.get("type")
    if action_type not in ALLOWED_ACTION_TYPES:
        return ["UNKNOWN_ACTION_TYPE"]
    if action_type == "move":
        if set(seq_item.keys()) != {"type", "subtype", "movementType", "to"}:
            errors.append("INVALID_MOVE_KEYS")
        if seq_item.get("subtype") not in ALLOWED_MOVE_SUBTYPES:
            errors.append("INVALID_MOVE_SUBTYPE")
        if seq_item.get("movementType") not in ALLOWED_MOVEMENT_TYPES:
            errors.append("INVALID_MOVEMENT_TYPE")
        if not isinstance(seq_item.get("to"), str):
            errors.append("MOVE_TO_NOT_STRING")
    elif action_type == "attack":
        if set(seq_item.keys()) != {"type", "target"}:
            errors.append("INVALID_ATTACK_KEYS")
        if not isinstance(seq_item.get("target"), str):
            errors.append("ATTACK_TARGET_NOT_STRING")
    elif action_type == "skill":
        if set(seq_item.keys()) != {"type", "description", "target"}:
            errors.append("INVALID_SKILL_KEYS")
        if not isinstance(seq_item.get("description"), str):
            errors.append("SKILL_DESCRIPTION_NOT_STRING")
        if not isinstance(seq_item.get("target"), str):
            errors.append("SKILL_TARGET_NOT_STRING")
    elif action_type == "wait":
        if set(seq_item.keys()) != {"type", "durationSec"}:
            errors.append("INVALID_WAIT_KEYS")
        duration = seq_item.get("durationSec")
        if not is_number(duration):
            errors.append("WAIT_DURATION_NOT_NUMBER")
        elif duration < MIN_WAIT_SECONDS or duration > MAX_WAIT_SECONDS:
            errors.append("WAIT_DURATION_OUT_OF_RANGE")
    elif action_type == "skillControl":
        mode = seq_item.get("mode")
        if mode == "defer":
            if set(seq_item.keys()) != {"type", "mode", "durationSec"}:
                errors.append("INVALID_SKILL_CONTROL_DEFER_KEYS")
            duration = seq_item.get("durationSec")
            if not is_number(duration):
                errors.append("SKILL_CONTROL_DURATION_NOT_NUMBER")
            elif duration < MIN_WAIT_SECONDS or duration > MAX_WAIT_SECONDS:
                errors.append("SKILL_CONTROL_DURATION_OUT_OF_RANGE")
        elif mode == "forbid":
            if set(seq_item.keys()) != {"type", "mode"}:
                errors.append("INVALID_SKILL_CONTROL_FORBID_KEYS")
        else:
            errors.append("INVALID_SKILL_CONTROL_MODE")
    return errors


def get_units(runtime_input: dict[str, Any], side: str) -> list[dict[str, Any]]:
    area = runtime_input.get("area_situation")
    if not isinstance(area, dict):
        return []
    value = area.get(side)
    return [unit for unit in value if isinstance(unit, dict)] if isinstance(value, list) else []


def get_unit_by_id(runtime_input: dict[str, Any], unit_id: str) -> dict[str, Any] | None:
    for side in ("allies", "enemies"):
        for unit in get_units(runtime_input, side):
            if unit.get("unitId") == unit_id:
                return unit
    return None


def get_unit_side(runtime_input: dict[str, Any], unit_id: str) -> str | None:
    for unit in get_units(runtime_input, "allies"):
        if unit.get("unitId") == unit_id:
            return "ally"
    for unit in get_units(runtime_input, "enemies"):
        if unit.get("unitId") == unit_id:
            return "enemy"
    return None


def is_alive(unit: dict[str, Any] | None) -> bool:
    return bool(unit and unit.get("isAlive") is True)


def is_targetable(unit: dict[str, Any] | None) -> bool:
    return bool(unit and unit.get("canBeTargeted") is True)


def unit_ids(runtime_input: dict[str, Any], side: str) -> set[str]:
    return {unit.get("unitId") for unit in get_units(runtime_input, side) if isinstance(unit.get("unitId"), str)}


def all_unit_ids(runtime_input: dict[str, Any]) -> set[str]:
    return unit_ids(runtime_input, "allies") | unit_ids(runtime_input, "enemies")


def actor_has_skill(runtime_input: dict[str, Any], actor_id: str) -> bool:
    unit = get_unit_by_id(runtime_input, actor_id)
    return bool(unit and isinstance(unit.get("skillDescription"), str) and unit.get("skillDescription"))


def actor_skill_description(runtime_input: dict[str, Any], actor_id: str) -> str | None:
    unit = get_unit_by_id(runtime_input, actor_id)
    value = unit.get("skillDescription") if unit else None
    return value if isinstance(value, str) and value else None


def can_actor_skill_target_dead(runtime_input: dict[str, Any], actor_id: str) -> bool:
    unit = get_unit_by_id(runtime_input, actor_id)
    return bool(unit and unit.get("canSkillTargetDead") is True)


def is_valid_skill_target(runtime_input: dict[str, Any], actor_id: str, target_id: str) -> bool:
    actor = get_unit_by_id(runtime_input, actor_id)
    target = get_unit_by_id(runtime_input, target_id)
    if actor is None or target is None:
        return False

    target_side = get_unit_side(runtime_input, target_id)
    if actor.get("IsSkillOnSelf") is True:
        if target_id != actor_id:
            return False
    elif actor.get("IsSkillOnOtherAlly") is True:
        if target_id == actor_id or target_side != "ally":
            return False
    else:
        if target_side != "enemy":
            return False

    if not is_targetable(target):
        return False
    if is_alive(target):
        return True
    return can_actor_skill_target_dead(runtime_input, actor_id)


def output_action_entries(output: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(output, dict):
        return []
    action = output.get("action")
    return [entry for entry in action if isinstance(entry, dict)] if isinstance(action, list) else []


def iter_sequence_items(output: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for entry in output_action_entries(output):
        actor_id = entry.get("unitId")
        sequence = entry.get("sequence")
        if not isinstance(actor_id, str) or not isinstance(sequence, list):
            continue
        for seq_item in sequence:
            if isinstance(seq_item, dict):
                items.append((actor_id, seq_item))
    return items


def check_runtime_contract(parsed: dict[str, Any] | None, runtime_input: dict[str, Any], command_analysis: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return False, ["NO_PARSED_OUTPUT"]

    ally_ids = unit_ids(runtime_input, "allies")
    enemy_ids = unit_ids(runtime_input, "enemies")
    all_ids = all_unit_ids(runtime_input)
    allowed_actors = set(command_analysis.get("allowedActors", [])) if isinstance(command_analysis.get("allowedActors"), list) else set()
    allowed_attack_targets = set(command_analysis.get("allowedAttackTargets", [])) if isinstance(command_analysis.get("allowedAttackTargets"), list) else set()
    valid_move_to_units = set(command_analysis.get("validMoveToUnits", [])) if isinstance(command_analysis.get("validMoveToUnits"), list) else set()

    for action_entry in output_action_entries(parsed):
        actor_id = action_entry.get("unitId")
        if not isinstance(actor_id, str):
            continue
        actor_unit = get_unit_by_id(runtime_input, actor_id)
        if actor_id not in ally_ids:
            errors.append("ACTOR_NOT_ALLY")
        if not is_alive(actor_unit):
            errors.append("ACTOR_DEAD")
        if actor_id not in allowed_actors:
            errors.append("ACTOR_OUTSIDE_ALLOWED_ACTORS")

        sequence = action_entry.get("sequence")
        if not isinstance(sequence, list):
            continue
        for index, seq_item in enumerate(sequence):
            if not isinstance(seq_item, dict):
                continue
            action_type = seq_item.get("type")
            if action_type == "move":
                to_id = seq_item.get("to")
                if not isinstance(to_id, str):
                    continue
                target_unit = get_unit_by_id(runtime_input, to_id)
                target_side = get_unit_side(runtime_input, to_id)
                if to_id == actor_id:
                    errors.append("MOVE_TO_SELF")
                if to_id not in all_ids:
                    errors.append("MOVE_TO_NOT_FOUND")
                if to_id not in valid_move_to_units:
                    errors.append("MOVE_TO_OUTSIDE_VALID_MOVE_TO_UNITS")
                if target_side == "enemy" and (not is_alive(target_unit) or not is_targetable(target_unit)):
                    errors.append("MOVE_TO_INVALID_ENEMY")
                if target_side == "ally" and not is_alive(target_unit):
                    errors.append("MOVE_TO_DEAD_ALLY")
            elif action_type == "attack":
                target = seq_item.get("target")
                if not isinstance(target, str):
                    continue
                target_unit = get_unit_by_id(runtime_input, target)
                if target not in enemy_ids:
                    errors.append("ATTACK_TARGET_NOT_ENEMY")
                if not is_alive(target_unit):
                    errors.append("ATTACK_TARGET_DEAD")
                if not is_targetable(target_unit):
                    errors.append("ATTACK_TARGET_UNTARGETABLE")
                if target not in allowed_attack_targets:
                    errors.append("ATTACK_TARGET_OUTSIDE_ALLOWED_TARGETS")
            elif action_type == "skill":
                target = seq_item.get("target")
                description = seq_item.get("description")
                if not actor_has_skill(runtime_input, actor_id):
                    errors.append("SKILL_ACTOR_HAS_NO_SKILL")
                    continue
                if description != actor_skill_description(runtime_input, actor_id):
                    errors.append("SKILL_DESCRIPTION_MISMATCH")
                if not isinstance(target, str) or target not in all_ids:
                    errors.append("SKILL_TARGET_NOT_FOUND")
                elif not is_valid_skill_target(runtime_input, actor_id, target):
                    errors.append("SKILL_TARGET_INVALID")
            elif action_type == "skillControl":
                if not actor_has_skill(runtime_input, actor_id):
                    errors.append("SKILL_CONTROL_ACTOR_HAS_NO_SKILL")

            if action_type == "wait" and index > 0:
                previous = sequence[index - 1]
                if isinstance(previous, dict) and previous.get("type") in {"attack", "skill"}:
                    errors.append("WAIT_AFTER_ATTACK_OR_SKILL")

    dialog_valid, dialog_errors = check_dialog_validity(parsed)
    if not dialog_valid:
        errors.extend(dialog_errors)

    return len(errors) == 0, errors


def check_dialog_validity(parsed: dict[str, Any] | None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return False, ["NO_PARSED_OUTPUT"]
    action_actor_ids = [entry.get("unitId") for entry in output_action_entries(parsed) if isinstance(entry.get("unitId"), str)]
    action_actor_set = set(action_actor_ids)
    dialog = parsed.get("dialog")
    if not isinstance(dialog, list):
        return False, ["DIALOG_NOT_ARRAY"]
    dialog_ids: list[str] = []
    dialog_texts: list[str] = []
    for entry in dialog:
        if not isinstance(entry, dict):
            errors.append("DIALOG_ITEM_NOT_OBJECT")
            continue
        unit_id = entry.get("unitId")
        text = entry.get("text")
        if isinstance(unit_id, str):
            dialog_ids.append(unit_id)
            if unit_id not in action_actor_set:
                errors.append("DIALOG_UNIT_MISMATCH")
        if isinstance(text, str):
            dialog_texts.append(text)
    if not action_actor_ids and dialog:
        errors.append("EMPTY_ACTION_DIALOG_NOT_EMPTY")
    for actor_id in action_actor_set:
        if dialog_ids.count(actor_id) != 1:
            errors.append("DIALOG_MISSING_OR_DUPLICATED_FOR_ACTOR")
    for dialog_id in set(dialog_ids):
        if dialog_ids.count(dialog_id) > 1:
            errors.append("DIALOG_DUPLICATED_UNIT")
    if len(action_actor_set) > 1:
        for text in set(dialog_texts):
            if text and dialog_texts.count(text) > 1:
                errors.append("DIALOG_TEXT_DUPLICATED_ACROSS_ACTORS")
    return len(errors) == 0, errors


def actor_set(output: dict[str, Any] | None) -> set[str]:
    return {entry.get("unitId") for entry in output_action_entries(output) if isinstance(entry.get("unitId"), str)}


def target_set(output: dict[str, Any] | None) -> set[str]:
    targets: set[str] = set()
    for _, seq_item in iter_sequence_items(output):
        for key in ("target", "to"):
            value = seq_item.get(key)
            if isinstance(value, str):
                targets.add(value)
    return targets


def action_type_signature(output: dict[str, Any] | None) -> list[tuple[str, tuple[str, ...]]]:
    signature: list[tuple[str, tuple[str, ...]]] = []
    for entry in output_action_entries(output):
        actor_id = entry.get("unitId")
        sequence = entry.get("sequence")
        if not isinstance(actor_id, str) or not isinstance(sequence, list):
            continue
        types = tuple(item.get("type") for item in sequence if isinstance(item, dict) and isinstance(item.get("type"), str))
        signature.append((actor_id, types))
    return sorted(signature)


def compare_with_gold(parsed: dict[str, Any] | None, gold: dict[str, Any], contract_ok: bool) -> dict[str, bool]:
    exact_json_match = parsed == gold if parsed is not None else False
    parsed_action = parsed.get("action") if isinstance(parsed, dict) else None
    gold_action = gold.get("action") if isinstance(gold, dict) else None
    action_exact_match = parsed_action == gold_action
    actor_set_match = actor_set(parsed) == actor_set(gold)
    target_set_match = target_set(parsed) == target_set(gold)
    action_type_sequence_match = action_type_signature(parsed) == action_type_signature(gold)
    semantic_or_action_valid_match = bool(action_exact_match or (contract_ok and actor_set_match and target_set_match and action_type_sequence_match))
    empty_action_correct = bool(isinstance(parsed_action, list) and isinstance(gold_action, list) and ((not parsed_action and not gold_action) or bool(parsed_action) == bool(gold_action)))
    return {
        "exact_json_match": exact_json_match,
        "action_exact_match": action_exact_match,
        "actor_set_match": actor_set_match,
        "target_set_match": target_set_match,
        "action_type_sequence_match": action_type_sequence_match,
        "semantic_or_action_valid_match": semantic_or_action_valid_match,
        "empty_action_correct": empty_action_correct,
    }


def evaluate_one_sample(sample: PreparedSample, model_name: str, request_success: bool, timeout: bool, latency_ms: float, output_tokens: int | None, raw_text: str | None) -> dict[str, Any]:
    parse_ok, parsed = parse_model_output(raw_text)
    top_level_schema_ok, strict_schema_ok, schema_errors = check_output_schema(parsed)
    contract_ok, contract_errors = check_runtime_contract(parsed, sample.runtime_input, sample.command_analysis)
    dialog_valid, _ = check_dialog_validity(parsed)
    gold_cmp = compare_with_gold(parsed, sample.gold_output, contract_ok)
    metadata = sample.metadata

    return {
        "model_name": model_name,
        "row_index": sample.row_index,
        "request_success": request_success,
        "timeout": timeout,
        "latency_ms": latency_ms,
        "output_tokens": output_tokens if output_tokens is not None else "",
        "json_parse_ok": parse_ok,
        "top_level_schema_ok": top_level_schema_ok,
        "strict_schema_ok": strict_schema_ok,
        "contract_ok": contract_ok,
        "contract_violation_count": len(contract_errors),
        "sac_ok": bool(strict_schema_ok and contract_ok),
        "exact_json_match": gold_cmp["exact_json_match"],
        "action_exact_match": gold_cmp["action_exact_match"],
        "semantic_or_action_valid_match": gold_cmp["semantic_or_action_valid_match"],
        "dialog_valid": dialog_valid,
        "empty_action_correct": gold_cmp["empty_action_correct"],
        "intent_family": metadata.get("intent_family", "unknown"),
        "command_style": metadata.get("command_style", "unknown"),
        "actor_selection": metadata.get("actor_selection", "unknown"),
        "target_selection": metadata.get("target_selection", "unknown"),
        "action_pattern": metadata.get("action_pattern", "unknown"),
        "_schema_error_count": len(schema_errors),
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def bool_rate(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get(field) is True) / len(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float)) and row.get("request_success") is True]
    tokens = [float(row["output_tokens"]) for row in rows if isinstance(row.get("output_tokens"), int)]
    total = len(rows)
    return {
        "total_samples": total,
        "request_success_rate": bool_rate(rows, "request_success"),
        "json_parse_success_rate": bool_rate(rows, "json_parse_ok"),
        "top_level_schema_success_rate": bool_rate(rows, "top_level_schema_ok"),
        "strict_schema_success_rate": bool_rate(rows, "strict_schema_ok"),
        "contract_violation_rate": 1 - bool_rate(rows, "contract_ok") if rows else None,
        "sac_success_rate": bool_rate(rows, "sac_ok"),
        "exact_json_match_rate": bool_rate(rows, "exact_json_match"),
        "action_exact_match_rate": bool_rate(rows, "action_exact_match"),
        "semantic_or_action_valid_match_rate": bool_rate(rows, "semantic_or_action_valid_match"),
        "dialog_validity_rate": bool_rate(rows, "dialog_valid"),
        "empty_action_correctness_rate": bool_rate(rows, "empty_action_correct"),
        "avg_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "avg_output_tokens": mean(tokens),
        "p95_output_tokens": percentile(tokens, 0.95),
        "timeout_rate": bool_rate(rows, "timeout"),
    }


def breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return {key: summarize_rows(value) for key, value in sorted(groups.items())}


def breakdown_validation(rows: list[dict[str, Any]], breakdowns: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    result: dict[str, Any] = {}
    for name, group_summary in breakdowns.items():
        sample_sum = 0
        unknown_like_groups: list[str] = []
        for group_name, summary in group_summary.items():
            if isinstance(summary, dict):
                count = summary.get("total_samples")
                if isinstance(count, int):
                    sample_sum += count
            if str(group_name).startswith("unknown") or str(group_name) == "metadata:unknown/unknown/unknown/unknown/unknown/unknown/unknown":
                unknown_like_groups.append(str(group_name))
        result[name] = {
            "group_count": len(group_summary),
            "total_samples_sum": sample_sum,
            "expected_total_samples": total,
            "sum_matches_total": sample_sum == total,
            "unknown_like_groups": unknown_like_groups,
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PER_ROW_FIELDS})


def read_response_records(path: Path) -> list[dict[str, Any]]:
    records = read_json_records(path)
    if not records:
        raise ValueError(f"No response rows found: {path}")
    return records


def first_string_value(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def extract_raw_text(record: dict[str, Any]) -> str | None:
    raw = first_string_value(record, ("raw_content", "raw_text", "content", "output", "response"))
    if raw is not None:
        return raw

    message = record.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    choices = record.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    return None


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def float_or_default(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def bool_or_default(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def index_responses_by_row(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for fallback_index, record in enumerate(records, start=1):
        row_index = int_or_none(record.get("row_index"))
        if row_index is None:
            row_index = fallback_index
        if row_index in indexed:
            raise ValueError(f"Duplicate response row_index: {row_index}")
        indexed[row_index] = record
    return indexed


def infer_model_name(records: list[dict[str, Any]], responses_path: Path, explicit_name: str) -> str:
    if explicit_name:
        return explicit_name
    for record in records:
        label = first_string_value(record, ("model_label", "model_name", "name"))
        if label:
            return label
    if responses_path.parent.name:
        return responses_path.parent.name
    return responses_path.stem


def evaluate_response_file(samples: list[PreparedSample], responses_path: Path, model_name: str) -> list[dict[str, Any]]:
    records = read_response_records(responses_path)
    response_by_row = index_responses_by_row(records)
    rows: list[dict[str, Any]] = []

    missing_rows = [sample.row_index for sample in samples if sample.row_index not in response_by_row]
    if missing_rows:
        preview = ", ".join(str(value) for value in missing_rows[:10])
        raise ValueError(f"responses file is missing row_index values: {preview}")

    for sample in samples:
        response = response_by_row[sample.row_index]
        raw_text = extract_raw_text(response)
        request_success = bool_or_default(response.get("request_success"), raw_text is not None)
        timeout = bool_or_default(response.get("timeout"), False)
        latency_ms = float_or_default(response.get("latency_ms"), 0.0)
        output_tokens = int_or_none(response.get("output_tokens"))
        if output_tokens is None and isinstance(raw_text, str):
            output_tokens = len(raw_text)

        row = evaluate_one_sample(
            sample=sample,
            model_name=model_name,
            request_success=request_success,
            timeout=timeout,
            latency_ms=latency_ms,
            output_tokens=output_tokens,
            raw_text=raw_text,
        )
        rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one saved responses.jsonl file against accepted test samples.")
    parser.add_argument("responses_path", help="Saved responses.jsonl from one model/adapter run.")
    parser.add_argument("test_path", help="Accepted test JSONL path. Pure messages JSONL is accepted only if metadata is preserved per row.")
    parser.add_argument("output_dir", help="Directory where report files will be written.")
    parser.add_argument("--model-name", default="", help="Override model name in reports. Defaults to model_label in responses.jsonl.")
    parser.add_argument("--baseline-name", default="", help="Stored in summary only. Deltas are not computed in single-response-file mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_path = Path(args.responses_path)
    test_path = Path(args.test_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = prepare_samples(test_path)
    response_records = read_response_records(responses_path)
    model_name = infer_model_name(response_records, responses_path, args.model_name)

    rows = evaluate_response_file(samples=samples, responses_path=responses_path, model_name=model_name)

    per_model_csv = output_dir / "test_per_model_results.csv"
    write_csv(per_model_csv, rows)

    model_summaries = {model_name: summarize_rows(rows)}
    model_breakdowns = {
        "by_intent_family": breakdown(rows, "intent_family"),
        "by_command_style": breakdown(rows, "command_style"),
        "by_actor_selection": breakdown(rows, "actor_selection"),
        "by_target_selection": breakdown(rows, "target_selection"),
        "by_action_pattern": breakdown(rows, "action_pattern"),
    }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_saved_responses",
        "taxonomy_mode": "metadata_5_axis",
        "skill_override_evaluation": False,
        "test_path": str(test_path),
        "responses_path": str(responses_path),
        "total_samples": len(samples),
        "total_responses": len(response_records),
        "model_name": model_name,
        "baseline_name": args.baseline_name or None,
        "model_summaries": model_summaries,
        "breakdown": {model_name: model_breakdowns},
        "breakdown_validation": {model_name: breakdown_validation(rows, model_breakdowns)},
        "excluded_outputs": [
            "general_path breakdown",
            "scenario_family breakdown",
            "edge_flags breakdown",
            "skill override breakdown",
            "skill_case breakdown",
            "malformed output examples",
            "split/sample id별 결과",
        ],
    }

    summary_path = output_dir / "test_summary_report.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] model={model_name}")
    print(f"[done] samples={len(samples)} responses={len(response_records)}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {per_model_csv}")
    print("[done] breakdowns=" + ", ".join(model_breakdowns.keys()))


if __name__ == "__main__":
    main()
