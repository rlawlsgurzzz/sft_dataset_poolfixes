# SFT taxonomy SOT를 로드하고 생성 요청 경로를 stable path로 변환한다.
# 일반 valid matrix와 skill valid matrix를 LLM 호출 전에 검사한다.
# edge_flags, command_style, skill_case enum 위반을 조기에 차단한다.
# coverage report와 generation request 코드가 공유하는 순수 유틸리티 모듈이다.

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "config" / "taxonomy_sot.json"
REQUEST_PATTERN = re.compile(
    r"^c(?P<general>\d+-\d+-\d+-\d+-\d+-\d+)(?:/(?P<skill>\d+-\d+-\d+))?\.(?P<count>\d+)$"
)
STABLE_REQUEST_PATTERN = re.compile(
    r"^(?P<general>[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)#(?P<slot>\d+)(?:/(?P<skill>[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+|\.null)))?\.(?P<count>\d+)$"
)


@dataclass(frozen=True)
class GeneralPath:
    intent_family: str
    actor_selection: str
    target_selection: str
    action_pattern: str
    scenario_family: str
    command_slot_index: int

    @property
    def stable_path(self) -> str:
        return ".".join(
            [
                self.intent_family,
                self.actor_selection,
                self.target_selection,
                self.action_pattern,
                self.scenario_family,
            ]
        )


@dataclass(frozen=True)
class SkillPath:
    skill_family: str
    skill_target_kind: str
    conflict_type: Optional[str]

    @property
    def stable_path(self) -> str:
        conflict = "null" if self.conflict_type is None else self.conflict_type
        return ".".join([self.skill_family, self.skill_target_kind, conflict])


@dataclass(frozen=True)
class ParsedGenerationRequest:
    raw_request: str
    general_path: GeneralPath
    skill_path: Optional[SkillPath]
    count: int


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")

    return data


def load_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> dict[str, Any]:
    taxonomy = load_json_file(path)
    validate_taxonomy_shape(taxonomy)
    return taxonomy


def get_order(taxonomy: dict[str, Any], key: str) -> list[str]:
    orders = taxonomy.get("orders", {})
    values = orders.get(key)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"taxonomy.orders.{key} must be a string list.")
    return values


def get_by_1_based_index(values: list[str], index_text: str, label: str) -> str:
    index = int(index_text)
    if index < 1 or index > len(values):
        raise ValueError(f"{label} index out of range: {index}")
    return values[index - 1]


def normalize_conflict_type(value: str) -> Optional[str]:
    return None if value == "null" else value


def conflict_type_to_key(value: Optional[str]) -> str:
    return "null" if value is None else value


def parse_generation_request(
    raw_request: str,
    taxonomy: dict[str, Any],
) -> ParsedGenerationRequest:
    stripped = raw_request.strip()

    if stripped.startswith("c"):
        return parse_numeric_generation_request(stripped, taxonomy)

    return parse_stable_generation_request(stripped, taxonomy)

def parse_numeric_generation_request(
    raw_request: str,
    taxonomy: dict[str, Any],
) -> ParsedGenerationRequest:
    match = REQUEST_PATTERN.match(raw_request)

    if not match:
        raise ValueError(
            "Invalid numeric request format. "
            "Expected: c1-2-1-3-1-10.4 or c1-2-1-3-1-10/2-3-1.7"
        )

    general_numbers = match.group("general").split("-")
    skill_numbers = match.group("skill")
    count = int(match.group("count"))

    if count < 1:
        raise ValueError("count must be at least 1.")

    intent_family = get_by_1_based_index(
        get_order(taxonomy, "intent_family"),
        general_numbers[0],
        "intent_family",
    )

    actor_selection = get_by_1_based_index(
        get_order(taxonomy, "actor_selection"),
        general_numbers[1],
        "actor_selection",
    )

    target_selection = get_by_1_based_index(
        get_order(taxonomy, "target_selection"),
        general_numbers[2],
        "target_selection",
    )

    action_pattern = get_by_1_based_index(
        get_order(taxonomy, "action_pattern"),
        general_numbers[3],
        "action_pattern",
    )

    scenario_family = get_scenario_family_by_intent_family_index(
        taxonomy=taxonomy,
        intent_family=intent_family,
        index_text=general_numbers[4],
    )

    command_slot_index = int(general_numbers[5])
    if command_slot_index < 1:
        raise ValueError("command_slot_index must be at least 1.")

    general_path = GeneralPath(
        intent_family=intent_family,
        actor_selection=actor_selection,
        target_selection=target_selection,
        action_pattern=action_pattern,
        scenario_family=scenario_family,
        command_slot_index=command_slot_index,
    )

    skill_path = None

    if skill_numbers:
        skill_parts = skill_numbers.split("-")

        skill_family = get_by_1_based_index(
            get_order(taxonomy, "skill_family"),
            skill_parts[0],
            "skill_family",
        )

        skill_target_kind = get_by_1_based_index(
            get_order(taxonomy, "skill_target_kind"),
            skill_parts[1],
            "skill_target_kind",
        )

        conflict_value = get_by_1_based_index(
            get_order(taxonomy, "conflict_type"),
            skill_parts[2],
            "conflict_type",
        )

        skill_path = SkillPath(
            skill_family=skill_family,
            skill_target_kind=skill_target_kind,
            conflict_type=normalize_conflict_type(conflict_value),
        )

    parsed = ParsedGenerationRequest(
        raw_request=raw_request,
        general_path=general_path,
        skill_path=skill_path,
        count=count,
    )

    validate_generation_request(parsed, taxonomy)
    return parsed


def parse_stable_generation_request(
    raw_request: str,
    taxonomy: dict[str, Any],
) -> ParsedGenerationRequest:
    match = STABLE_REQUEST_PATTERN.match(raw_request)

    if not match:
        raise ValueError(
            "Invalid stable request format. "
            "Expected: attack.explicit_actor.explicit_enemy_target.attack_only.simple_clear_target#1.4 "
            "or skill.explicit_actor.explicit_ally_target.skill_only.ally_skill_valid_target#1/ally_shield.ally_alive.null.4"
        )

    general_parts = match.group("general").split(".")
    skill_text = match.group("skill")
    command_slot_index = int(match.group("slot"))
    count = int(match.group("count"))

    if command_slot_index < 1:
        raise ValueError("command_slot_index must be at least 1.")

    if count < 1:
        raise ValueError("count must be at least 1.")

    general_path = GeneralPath(
        intent_family=general_parts[0],
        actor_selection=general_parts[1],
        target_selection=general_parts[2],
        action_pattern=general_parts[3],
        scenario_family=general_parts[4],
        command_slot_index=command_slot_index,
    )

    skill_path = None

    if skill_text:
        skill_parts = skill_text.split(".")
        conflict_type = normalize_conflict_type(skill_parts[2])

        skill_path = SkillPath(
            skill_family=skill_parts[0],
            skill_target_kind=skill_parts[1],
            conflict_type=conflict_type,
        )

    parsed = ParsedGenerationRequest(
        raw_request=raw_request,
        general_path=general_path,
        skill_path=skill_path,
        count=count,
    )

    validate_generation_request(parsed, taxonomy)
    return parsed


def get_scenario_family_by_intent_family_index(
    taxonomy: dict[str, Any],
    intent_family: str,
    index_text: str,
) -> str:
    matrix = get_general_intent_family_matrix(taxonomy, intent_family)
    scenario_map = matrix.get("allowed_scenario_family", {})
    if not isinstance(scenario_map, dict):
        raise ValueError(f"general_valid_matrix.{intent_family}.allowed_scenario_family must be an object.")

    values = list(scenario_map.keys())
    return get_by_1_based_index(values, index_text, "scenario_family")


def get_general_intent_family_matrix(taxonomy: dict[str, Any], intent_family: str) -> dict[str, Any]:
    matrix = taxonomy.get("general_valid_matrix", {})
    intent_family_matrix = matrix.get(intent_family)
    if not isinstance(intent_family_matrix, dict):
        raise ValueError(f"Unknown intent_family in general_valid_matrix: {intent_family}")
    return intent_family_matrix


def validate_generation_request(
    request: ParsedGenerationRequest,
    taxonomy: dict[str, Any],
) -> None:
    validate_general_path(request.general_path, taxonomy)

    if request.skill_path is not None:
        if request.general_path.intent_family != "skill":
            raise ValueError("Skill override can only be used with intent_family=skill.")

        validate_skill_path(request.skill_path, taxonomy)


def validate_general_path(path: GeneralPath, taxonomy: dict[str, Any]) -> None:
    intent_matrix = get_general_intent_family_matrix(taxonomy, path.intent_family)

    checks = [
        ("allowed_actor_selection", path.actor_selection),
        ("allowed_target_selection", path.target_selection),
        ("allowed_action_pattern", path.action_pattern),
        ("allowed_scenario_family", path.scenario_family),
    ]

    for matrix_key, value in checks:
        allowed = intent_matrix.get(matrix_key, {})

        if not isinstance(allowed, dict):
            raise ValueError(
                f"general_valid_matrix.{path.intent_family}.{matrix_key} must be an object."
            )

        if value not in allowed:
            raise ValueError(
                f"Invalid general path: {path.stable_path}. "
                f"{value} is not allowed in {path.intent_family}.{matrix_key}."
            )


def validate_skill_path(path: SkillPath, taxonomy: dict[str, Any]) -> None:
    skill_matrix = taxonomy.get("skill_valid_matrix", {})
    family_matrix = skill_matrix.get(path.skill_family)
    if not isinstance(family_matrix, dict):
        raise ValueError(f"Unknown skill_family in skill_valid_matrix: {path.skill_family}")

    target_kind_map = family_matrix.get("allowed_skill_target_kind", {})
    if not isinstance(target_kind_map, dict) or path.skill_target_kind not in target_kind_map:
        raise ValueError(
            f"Invalid skill path: {path.stable_path}. skill_target_kind is not allowed."
        )

    conflict_key = conflict_type_to_key(path.conflict_type)
    conflict_map = family_matrix.get("allowed_conflict_type", {})
    if not isinstance(conflict_map, dict) or conflict_key not in conflict_map:
        raise ValueError(
            f"Invalid skill path: {path.stable_path}. conflict_type is not allowed."
        )


def validate_metadata_against_taxonomy(
    sample: dict[str, Any],
    taxonomy: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return ["metadata must be an object"]

    intent_family = metadata.get("intent_family")
    actor_selection = metadata.get("actor_selection")
    target_selection = metadata.get("target_selection")
    action_pattern = metadata.get("action_pattern")
    scenario_family = metadata.get("scenario_family")
    command_style = metadata.get("command_style")
    edge_flags = metadata.get("edge_flags", [])

    if not isinstance(intent_family, str):
        errors.append("metadata.intent_family must be a string")
    if not isinstance(actor_selection, str):
        errors.append("metadata.actor_selection must be a string")
    if not isinstance(target_selection, str):
        errors.append("metadata.target_selection must be a string")
    if not isinstance(action_pattern, str):
        errors.append("metadata.action_pattern must be a string")
    if not isinstance(scenario_family, str):
        errors.append("metadata.scenario_family must be a string")
    if not isinstance(command_style, str):
        errors.append("metadata.command_style must be a string")

    command_styles = taxonomy.get("global_targets", {}).get("command_style", {})
    if isinstance(command_style, str) and command_style not in command_styles:
        errors.append(f"unknown command_style: {command_style}")

    if not isinstance(edge_flags, list) or not all(isinstance(flag, str) for flag in edge_flags):
        errors.append("metadata.edge_flags must be a string list")
    else:
        allowed_edge_flags = set(taxonomy.get("edge_flags", []))
        for flag in edge_flags:
            if flag not in allowed_edge_flags:
                errors.append(f"unknown edge_flag: {flag}")

    if not errors:
        try:
            validate_general_path(
                GeneralPath(
                    intent_family=intent_family,
                    actor_selection=actor_selection,
                    target_selection=target_selection,
                    action_pattern=action_pattern,
                    scenario_family=scenario_family,
                    command_slot_index=1,
                ),
                taxonomy,
            )
        except ValueError as error:
            errors.append(str(error))

    skill_case = sample.get("skill_case")
    if skill_case is None:
        if intent_family == "skill":
            errors.append("skill intent_family sample must have non-null skill_case")
        return errors

    if not isinstance(skill_case, dict):
        errors.append("skill_case must be null or an object")
        return errors

    skill_family = skill_case.get("skill_family")
    skill_target_kind = skill_case.get("skill_target_kind")
    conflict_type = skill_case.get("conflict_type")

    if not isinstance(skill_family, str):
        errors.append("skill_case.skill_family must be a string")
    if not isinstance(skill_target_kind, str):
        errors.append("skill_case.skill_target_kind must be a string")
    if conflict_type is not None and not isinstance(conflict_type, str):
        errors.append("skill_case.conflict_type must be null or a string")

    if not errors:
        try:
            validate_skill_path(
                SkillPath(
                    skill_family=skill_family,
                    skill_target_kind=skill_target_kind,
                    conflict_type=conflict_type,
                ),
                taxonomy,
            )
        except ValueError as error:
            errors.append(str(error))

    return errors


def get_selected_bucket_descriptions(
    taxonomy: dict[str, Any],
    general_path: GeneralPath,
    skill_path: Optional[SkillPath] = None,
    edge_flags: Optional[list[str]] = None,
) -> dict[str, Any]:
    descriptions = taxonomy.get("descriptions", {})
    selected: dict[str, Any] = {
        "intent_family": general_path.intent_family,
        "intent_family_description": descriptions.get("intent_family", {}).get(general_path.intent_family, ""),
        "actor_selection": general_path.actor_selection,
        "actor_selection_description": descriptions.get("actor_selection", {}).get(general_path.actor_selection, ""),
        "target_selection": general_path.target_selection,
        "target_selection_description": descriptions.get("target_selection", {}).get(general_path.target_selection, ""),
        "action_pattern": general_path.action_pattern,
        "action_pattern_description": descriptions.get("action_pattern", {}).get(general_path.action_pattern, ""),
        "scenario_family": general_path.scenario_family,
        "scenario_family_description": descriptions.get("scenario_family", {}).get(general_path.scenario_family, ""),
        "edge_flags": edge_flags or [],
        "edge_flag_descriptions": {},
    }

    edge_descriptions = descriptions.get("edge_flags", {})
    for flag in selected["edge_flags"]:
        selected["edge_flag_descriptions"][flag] = edge_descriptions.get(flag, "")

    if skill_path is not None:
        conflict_key = conflict_type_to_key(skill_path.conflict_type)
        selected["skill_case"] = {
            "skill_family": skill_path.skill_family,
            "skill_family_description": descriptions.get("skill_family", {}).get(skill_path.skill_family, ""),
            "skill_target_kind": skill_path.skill_target_kind,
            "skill_target_kind_description": descriptions.get("skill_target_kind", {}).get(skill_path.skill_target_kind, ""),
            "conflict_type": skill_path.conflict_type,
            "conflict_type_description": descriptions.get("conflict_type", {}).get(conflict_key, ""),
        }

    return selected


def validate_taxonomy_shape(taxonomy: dict[str, Any]) -> None:
    required_top_level = {
        "schema_version",
        "global_targets",
        "descriptions",
        "orders",
        "general_valid_matrix",
        "skill_targets",
        "skill_valid_matrix",
        "edge_flags",
    }
    missing = required_top_level - set(taxonomy.keys())
    if missing:
        raise ValueError(f"taxonomy_sot.json missing keys: {sorted(missing)}")

    if not isinstance(taxonomy.get("edge_flags"), list):
        raise ValueError("taxonomy.edge_flags must be a list.")

    for intent_family in get_order(taxonomy, "intent_family"):
        get_general_intent_family_matrix(taxonomy, intent_family)

    for family in get_order(taxonomy, "skill_family"):
        skill_matrix = taxonomy.get("skill_valid_matrix", {})
        if family not in skill_matrix:
            raise ValueError(f"skill_valid_matrix missing skill_family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", help="Generation request path, e.g. c1-2-1-3-1-10.4")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    args = parser.parse_args()

    taxonomy = load_taxonomy(Path(args.taxonomy))
    parsed = parse_generation_request(args.request, taxonomy)

    result = {
        "raw_request": parsed.raw_request,
        "count": parsed.count,
        "general_path": {
            "intent_family": parsed.general_path.intent_family,
            "actor_selection": parsed.general_path.actor_selection,
            "target_selection": parsed.general_path.target_selection,
            "action_pattern": parsed.general_path.action_pattern,
            "scenario_family": parsed.general_path.scenario_family,
            "command_slot_index": parsed.general_path.command_slot_index,
            "stable_path": parsed.general_path.stable_path,
        },
        "skill_path": None
        if parsed.skill_path is None
        else {
            "skill_family": parsed.skill_path.skill_family,
            "skill_target_kind": parsed.skill_path.skill_target_kind,
            "conflict_type": parsed.skill_path.conflict_type,
            "stable_path": parsed.skill_path.stable_path,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
