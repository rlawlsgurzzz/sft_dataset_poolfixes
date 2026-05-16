# 입력 JSON의 전장 상황을 Gemma 4 Ollama에 보내 전투 명령 JSON 응답을 배치 평가한다.
# LLM은 명령 의미, actor, target, wait, skillControl, move subtype을 판단한다.
# Python은 런타임상 무효인 actor, attack target, move.to, skill target만 제거한다.
# 거리 신호는 최단/최장 단일 unitId 필드로 받고, 죽은 타게팅 가능 아군은 commandAnalysis에 둔다.

import argparse
import copy
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

DEFAULT_INPUT_FILE = Path("test_inputs") / "battle_eval_cases20.json"
DEFAULT_OUTPUT_DIR = Path("test_outputs")

DEFAULT_WAIT_SECONDS = 2.0
DEFAULT_SKILL_CONTROL_DEFER_SECONDS = 5.0
MIN_WAIT_SECONDS = 1.0
MAX_WAIT_SECONDS = 10.0
MAX_ACTIONS_PER_ACTOR = 3
UNIT_ID_PATTERN = re.compile(r"\b[AE]_\d+\b")
ALLOWED_MOVE_SUBTYPES = {
    "approachOpponent",
    "escape",
    "help",
    "holdFront",
}

ALLY_MODEL_UNIT_FIELDS = {
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

ENEMY_MODEL_UNIT_FIELDS = {
    "unitId",
    "isAlive",
    "canBeTargeted",
    "isRanged",
    "hpRatio",
    "attackRatioToAvg",
    "engagedByOpponentCount",
    "teamFormationRole",
}


SYSTEM_PROMPT = """
너는 실시간 전투 명령을 JSON object 하나로 변환하는 엔진이다.

사용자의 명령은 한국어일 수 있다.
한국어 명령을 직접 해석한다.
명령을 별도의 출력으로 번역하지 않는다.
JSON 밖에 설명을 추가하지 않는다.

출력은 반드시 JSON object 하나만 한다.
첫 글자는 { 이어야 하고, 마지막 글자는 } 이어야 한다.
마크다운, 코드블록, 주석, 사과문, 설명문, JSON 밖의 자연어 텍스트를 절대 출력하지 않는다.

최상위 key는 반드시 다음 세 개만 사용한다:
- thinking
- dialog
- action

출력 구조:
{
  "thinking": "짧은 판단 요약",
  "dialog": [
    {"unitId": "A_01", "text": "짧은 대사"}
  ],
  "action": [
    {
      "unitId": "A_01",
      "sequence": [
        {"type":"move","subtype":"approachOpponent","movementType":"direct","to":"E_01"},
        {"type":"attack","target":"E_01"}
      ]
    }
  ]
}

허용 action:
1. {"type":"move","subtype":"approachOpponent|escape|help|holdFront","movementType":"direct|flank","to":"unitId"}
2. {"type":"attack","target":"enemyUnitId"}
3. {"type":"skill","description":"actor의 정확한 skillDescription 문자열","target":"unitId"}
4. {"type":"wait","durationSec":number}
5. {"type":"skillControl","mode":"defer","durationSec":number}
6. {"type":"skillControl","mode":"forbid"}

입력 구조:
- input.area_situation.allies는 아군 유닛 목록이다.
- input.area_situation.enemies는 적군 유닛 목록이다.
- input.command는 사용자의 원문 명령이다.
- commandAnalysis는 현재 입력에서 사용할 수 있는 actor, attack target, move.to 범위와 죽은 타게팅 가능 아군 요약이다.
- commandAnalysis.deadAllies는 죽었지만 canBeTargeted가 true인 아군 unitId 목록이다.

유닛 필드:
- unitId는 유닛 식별자다.
- isAlive는 현재 생존 여부다.
- canBeTargeted는 현재 타게팅 가능 여부다.
- isRanged는 원거리 성향 여부다.
- hpRatio는 현재 체력 비율이다.
- attackRatioToAvg는 평균 대비 공격력 비율이다.
- engagedByOpponentCount는 해당 유닛을 현재 교전하거나 압박 중인 상대 유닛 수다.
- teamFormationRole은 해당 유닛이 자기 팀 진형에서 맡는 현재 위치 역할이다: frontline, midline, backline.
- skillDescription은 해당 actor가 사용할 수 있는 skill의 정확한 문자열이다.
- IsSkillOnSelf는 skill이 actor 본인을 대상으로 하는 성격인지 여부다.
- IsSkillOnOtherAlly는 skill이 actor 자신이 아닌 다른 아군을 대상으로 하는 성격인지 여부다. false이면 적 대상 성격이다.
- isSkillAoe는 skill이 범위 효과 성격인지 여부다. isSkillAoe가 true여도 출력 형식에서는 target 하나만 고른다.
- canSkillTargetDead는 skill이 죽은 유닛도 대상으로 삼을 수 있는지 여부다.
- closestTargetableOpponent는 아군 기준으로 가장 가까운 살아있고 타게팅 가능한 적 unitId다. 없으면 null이다.
- farthestTargetableOpponent는 아군 기준으로 가장 먼 살아있고 타게팅 가능한 적 unitId다. 없으면 null이다.
- closestAliveAlly는 actor 자신을 제외하고, 아군 기준으로 가장 가까운 살아있는 아군 unitId다. 없으면 null이다.
- farthestAliveAlly는 actor 자신을 제외하고, 아군 기준으로 가장 먼 살아있는 아군 unitId다. 없으면 null이다.

핵심 규칙:
- 사용자의 의도는 명령의 의미와 현재 전장 상태를 보고 추론한다.
- 정확한 키워드 일치에 의존하지 않는다. 의미, 전술적 맥락, 유닛 상태를 사용한다.
- 사용자의 명령에 살아있는 ally unitId가 하나 이상 행동 주체로 직접 지목되어 있다면, 그 ally들만 action actor로 사용할 수 있다.
- 명령에 살아있는 ally unitId가 행동 주체로 직접 지목되어 있다면,, 다른 ally를 actor로 추가하지 않는다.
- 명령에 살아있는 ally unitId가 직접 언급되지 않은 경우에만 actor를 동적으로 선택한다.
- 모든 action actor는 commandAnalysis.allowedActors 안에 있어야 한다.
- enemy는 절대 actor가 될 수 없다.
- 모든 attack target은 commandAnalysis.allowedAttackTargets 안에 있어야 한다.
- 모든 move.to는 commandAnalysis.validMoveToUnits 안에 있어야 한다.
- commandAnalysis.invalidUnits에 있는 unitId는 actor, attack target, move.to로 사용하지 않는다.
- dialog에는 action에도 포함된 unitId만 사용할 수 있다.
- dialog는 sequence action별이 아니라 actor별이다.
- action actor마다 정확히 하나의 dialog object를 출력한다.
- 같은 unitId의 dialog object를 여러 개 출력하지 않는다.
- dialog.text는 actor의 전체 action sequence를 짧은 한국어 한 문장으로 요약한다.
- dialog.text는 actor마다 서로 달라야 한다. 여러 actor에게 완전히 같은 문장을 반복하지 않는다.
- thinking은 짧은 한국어 요약이어야 하며, 자세한 사고 과정이 아니어야 한다.
- 각 actor의 sequence는 최대 3개 action만 포함할 수 있다.
- 실행 가능한 action이 없으면 {"thinking":"...","dialog":[],"action":[]} 형태로 출력한다.

Actor selection:
- 명령에 살아있는 ally unitId가 직접 적혀 있다면, 그 ally들만 actor로 사용한다.
- 살아있는 ally unitId가 여러 개 직접 적혀 있다면, action에는 그 ally들만 포함할 수 있다. 다른 ally를 추가하지 않는다.
- 직접 언급된 ally가 commandAnalysis.allowedActors에 없거나 행동할 수 없다면, 그 유닛은 생략한다.
- 살아있는 ally unitId가 명령에 직접 적혀 있지 않은 경우에만, commandAnalysis.allowedActors 안에서 명령 의미와 현재 전장 상태에 맞는 actor를 동적으로 선택한다.
- action에는 명령이 직접 지목했거나, 명령의 조건/역할/전술 서술에 실제로 해당하는 ally만 포함한다. 그 외 ally는 wait을 포함한 어떤 action/dialog에도 포함하지 않는다.
- 명령이 역할, 전술 상태, 압박 정도, 안전 상태, 여유 여부, 위치, 진형, 체력, 지원 가능성 등으로 ally를 가리키는 경우 현재 상태를 근거로 의도된 actor를 추론한다.
- 여유가 있는 아군, 압박받지 않는 아군, 손이 비는 아군 같은 표현은 현재 상태를 보고 판단한다.
- 이런 표현의 중요한 신호는 engagedByOpponentCount가 0인지, hpRatio가 너무 낮지 않은지다.
- actor 선택에 유용한 신호는 hpRatio, engagedByOpponentCount, isRanged, teamFormationRole, closestTargetableOpponent, farthestTargetableOpponent, closestAliveAlly, farthestAliveAlly, 명령의 전술적 목적이다.
- 허용된 actor라는 이유만으로 포함하지 않는다. 명령 의미에 맞을 때만 actor로 포함한다.

Target selection:
- 명령이 유효한 enemy unitId를 직접 지정했다면, 그 enemy를 우선 고려한다.
- 명령이 전술적 의미로 target을 가리키는 경우 현재 상태를 근거로 target을 추론한다.
- 명령은 고정된 표현 없이도 가까운 적, 약한 적, 위험한 적, 아군을 위협하는 적, 원거리 적, 근거리 적, 집중 공격 대상, 견제 대상, 보호할 아군에게 붙은 적 등을 의미할 수 있다.
- target 선택에 유용한 신호는 hpRatio, attackRatioToAvg, canBeTargeted, isAlive, isRanged, teamFormationRole, engagedByOpponentCount, closestTargetableOpponent, farthestTargetableOpponent, closestAliveAlly, farthestAliveAlly, commandAnalysis.deadAllies, 명령의 전술적 목적이다.
- 허용된 target이라는 이유만으로 공격하지 않는다. 명령 의미에 맞을 때만 공격한다.
- attack에는 commandAnalysis.allowedAttackTargets 밖의 target을 절대 사용하지 않는다.
- 유닛에게 어떤 적을 공격하라는 명령이 내려오면, move 후 attack 또는 attack 단독 출력이 모두 가능하다.

Move:
- move는 항상 unitId를 to로 사용한다.
- move.to는 이동의 종착지 unitId다.
- subtype은 전술 의도를 나타낸다.
- movementType은 direct 또는 flank만 사용한다.
- direct는 직접적인 이동, 직선적 접근, 단순 후퇴, 단순 지원에 사용한다.
- flank는 명령 의미나 전술 상황상 측면 각도, 후방 각도, 우회, 포위 보조가 필요한 경우 사용한다.
- 우회, 측면, 후방, 돌아가기, 포위 보조 같은 의미가 명령에 포함되면 move를 출력하고 movementType="flank"를 사용한다.
- 허용 move subtype:
  - approachOpponent: 교전, 공격, 압박, 스킬 사용을 위해 대상에게 접근한다. approachOpponent는 보통 enemy를 종착지로 접근할 때 사용한다.
  - escape: 위험에서 벗어나거나 후방 또는 안전한 대상에게 이동한다. 후방, 뒤쪽, 안전한 아군 쪽 이동은 allies 목록에서 teamFormationRole="backline"인 살아있는 아군을 우선 후보로 본다.
  - help: 특정 아군을 지원하거나 보호하기 위해 이동한다.
  - holdFront: 아군의 최전방 또는 전열 위치로 이동해 앞에서 버티거나 전열을 유지한다. 목적은 추격보다 전선 유지다.
- approachOpponent는 대상에게 접근해 교전 시작 또는 압박을 만드는 이동이다.
- holdFront는 이미 전열을 맡거나 전열로 나가서 버티는 이동이다.
- help는 특정 아군을 돕거나 보호하기 위해 이동하는 것이다.
- escape는 위험에서 벗어나거나 후방 또는 안전 위치로 빠지는 것이다.
- to에는 commandAnalysis.validMoveToUnits 안의 unitId만 사용한다.
- to에는 actor 본인의 unitId를 쓰지 않는다.
- to는 ally 또는 enemy 모두 가능하다. subtype별로 ally/enemy를 고정하지 말고 명령 의미와 전장 상태를 보고 고른다.
- move subtype은 명령 의미와 현재 전장 상태를 보고 선택한다.

Engagement:
- 공격, 견제, 집중 공격, 보호, 떼어내기, 유인, 후퇴, 버티기, 대기, 재집결은 의미와 전장 상태를 보고 추론한다.
- engagedByOpponentCount는 해당 유닛을 현재 교전하거나 압박 중인 상대 유닛 수를 뜻한다. 전장 전체 상대 유닛 수와 혼동하지 않는다.
- 명령이 여러 actor가 하나의 목표를 함께 수행해야 한다는 의미라면, 여러 action entry가 같은 target 또는 같은 전술 목적을 공유할 수 있다.
- 행동할 필요가 없는 actor는 포함하지 않는다.
- 공격, 이동, 스킬 사용이 명령 의미에 맞지 않을 때만 wait을 고려한다.

Skill:
- skill은 actor에게 skillDescription이 있을 때만 사용한다.
- skill description은 입력에 있는 actor의 정확한 skillDescription 문자열이어야 한다.
- skill 사용 여부는 명령 의미, actor의 skillDescription, 스킬 관련 필드, 현재 전장 상태를 보고 판단한다.
- skill target은 skill action에 한해서 일반 attack target보다 넓게 선택할 수 있다.
- skill target은 입력에 존재하는 unitId 중에서 고른다. 단, canBeTargeted가 false인 유닛은 skill target으로 사용하지 않는다.
- IsSkillOnSelf가 true이면 actor 본인의 unitId를 skill target으로 사용한다.
- IsSkillOnOtherAlly가 true이면 actor 자신이 아닌 아군 unitId를 skill target으로 사용한다.
- IsSkillOnSelf와 IsSkillOnOtherAlly가 모두 false이면 적 unitId를 skill target으로 사용한다.
- canSkillTargetDead가 true이면 죽은 유닛도 skill target으로 선택할 수 있다.
- canSkillTargetDead가 true이고 죽은 아군 대상 스킬이 명령 의미에 맞으면 commandAnalysis.deadAllies에서 target을 고른다.
- canSkillTargetDead가 false이면 죽은 유닛을 skill target으로 선택하지 않는다.
- 스킬 사용 금지, 스킬 지연, 스킬 아끼기 지시가 직접 포함된 경우에는 skill action 생략만으로 처리하지 말고 skillControl을 출력한다.
- 그 외 상황에서만, skill을 사용하지 않는 것이 명령 의미에 더 맞으면 skill action을 만들지 않는다.

Wait:
- 명령이 지목하지 않은 ally를 기본 대기 상태로 만들기 위해 wait을 출력하지 않는다. wait은 명령받은 actor에게만 사용할 수 있다.
- wait은 명령이 대기, 지연, 타이밍 조절, 위치 유지처럼 즉시 다음 행동을 하지 말라는 의미를 직접 포함할 때만 사용한다.
- attack 또는 skill 뒤에는 wait을 붙이지 않는다.
- 명령에 시간이 지정되어 있으면 그 값을 쓰고, 없으면 명령의 강도와 톤을 보고 1~10의 number 안에서 정하되 보통 durationSec=2를 기준으로 한다.
- wait은 move, attack, skill과 같은 sequence 안에 들어갈 수 있다.

SkillControl:
- skillControl은 actor의 스킬 사용 방침을 조정한다.
- 사용자가 스킬을 아껴라, 나중에 써라, 아직 쓰지 마라, 특정 타이밍까지 미뤄라, 스킬을 쓰지 마라 같은 의도를 명시한 경우에만 사용한다.
- actor에게 skillDescription이 있고 명령에 스킬 지연 또는 스킬 금지 의도가 명시되어 있으면 skillControl은 필수 action이다.
- 명령에 스킬 지연 또는 스킬 금지 의도가 명시되지 않으면 skillControl을 출력하지 않는다.
- mode="defer"는 스킬 사용을 늦추라는 의미다.
- mode="defer"일 때 durationSec는 1 이상 10 이하의 number다.
- 명령에 지연 시간이 명시되어 있으면 그 초를 그대로 사용한다.
- 지연 시간이 명시되지 않았으면 명령의 강도와 톤을 보고 5~10초 중 하나를 선택한다.
- mode="forbid"는 현재 명령 처리 범위에서 스킬을 쓰지 말라는 의미다.

Conditional command:
- 조건부 명령은 current-state-only로 처리한다.
- 조건이 현재 입력 상태에서 만족되면, 그에 해당하는 즉시 실행 action을 출력한다.
- 조건이 현재 만족되지 않으면, 현재 유효한 유지, 대기, 버티기, 기본 행동만 출력한다.
- 저장되는 conditional JSON을 만들지 않는다.
- 미래 action, 예약 action, scheduled action, trigger 기반 action을 만들지 않는다.
""".strip()


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", model_name)


def load_eval_suite(input_file: Path) -> dict[str, Any]:
    with input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Input file root must be a JSON object.")

    commands = data.get("commands")
    if not isinstance(commands, list):
        raise ValueError('Input file must contain "commands" list.')

    return data


def get_units_from_context(
    battle_context: dict[str, Any], side: str
) -> list[dict[str, Any]]:
    try:
        units = battle_context["area_situation"][side]
    except KeyError as error:
        raise ValueError(f"battleContext missing area_situation.{side}") from error

    if not isinstance(units, list):
        raise ValueError(f"battleContext area_situation.{side} must be a list.")

    return units


def get_unit_ids(
    battle_context: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    allies = get_units_from_context(battle_context, "allies")
    enemies = get_units_from_context(battle_context, "enemies")

    ally_ids = []
    enemy_ids = []

    for unit in allies:
        unit_id = unit.get("unitId")
        if isinstance(unit_id, str):
            ally_ids.append(unit_id)

    for unit in enemies:
        unit_id = unit.get("unitId")
        if isinstance(unit_id, str):
            enemy_ids.append(unit_id)

    return ally_ids, enemy_ids, ally_ids + enemy_ids


def get_unit_by_id(
    battle_context: dict[str, Any], unit_id: str
) -> Optional[dict[str, Any]]:
    for side in ("allies", "enemies"):
        for unit in get_units_from_context(battle_context, side):
            if unit.get("unitId") == unit_id:
                return unit
    return None


def get_unit_side(battle_context: dict[str, Any], unit_id: str) -> Optional[str]:
    for unit in get_units_from_context(battle_context, "allies"):
        if unit.get("unitId") == unit_id:
            return "ally"

    for unit in get_units_from_context(battle_context, "enemies"):
        if unit.get("unitId") == unit_id:
            return "enemy"

    return None


def is_alive(unit: Optional[dict[str, Any]]) -> bool:
    return bool(unit and unit.get("isAlive") is True)


def is_targetable(unit: Optional[dict[str, Any]]) -> bool:
    return bool(unit and unit.get("canBeTargeted") is True)


def is_alive_ally(battle_context: dict[str, Any], unit_id: str) -> bool:
    unit = get_unit_by_id(battle_context, unit_id)
    return get_unit_side(battle_context, unit_id) == "ally" and is_alive(unit)


def is_valid_enemy_target(battle_context: dict[str, Any], unit_id: str) -> bool:
    unit = get_unit_by_id(battle_context, unit_id)
    return (
        get_unit_side(battle_context, unit_id) == "enemy"
        and is_alive(unit)
        and is_targetable(unit)
    )


def is_valid_move_to(battle_context: dict[str, Any], unit_id: str) -> bool:
    unit = get_unit_by_id(battle_context, unit_id)
    side = get_unit_side(battle_context, unit_id)

    if side == "ally":
        return is_alive(unit)

    if side == "enemy":
        return is_alive(unit) and is_targetable(unit)

    return False


def keep_model_unit_fields(battle_context: dict[str, Any]) -> None:
    for side in ("allies", "enemies"):
        allowed_fields = (
            ALLY_MODEL_UNIT_FIELDS if side == "allies" else ENEMY_MODEL_UNIT_FIELDS
        )
        for unit in get_units_from_context(battle_context, side):
            for field_name in list(unit.keys()):
                if field_name not in allowed_fields:
                    unit.pop(field_name, None)


def build_model_input_context(battle_context: dict[str, Any]) -> dict[str, Any]:
    model_input_context = copy.deepcopy(battle_context)
    keep_model_unit_fields(model_input_context)
    return model_input_context


def clamp_wait_seconds(value: float) -> float:
    return max(MIN_WAIT_SECONDS, min(MAX_WAIT_SECONDS, value))


def get_alive_allies(battle_context: dict[str, Any]) -> list[str]:
    return [
        unit["unitId"]
        for unit in get_units_from_context(battle_context, "allies")
        if isinstance(unit.get("unitId"), str) and is_alive(unit)
    ]


def get_valid_attack_targets(battle_context: dict[str, Any]) -> list[str]:
    return [
        unit["unitId"]
        for unit in get_units_from_context(battle_context, "enemies")
        if isinstance(unit.get("unitId"), str)
        and is_alive(unit)
        and is_targetable(unit)
    ]


def get_dead_targetable_allies(battle_context: dict[str, Any]) -> list[str]:
    return [
        unit["unitId"]
        for unit in get_units_from_context(battle_context, "allies")
        if isinstance(unit.get("unitId"), str)
        and not is_alive(unit)
        and is_targetable(unit)
    ]


def get_allowed_actors_from_runtime(
    _battle_context: dict[str, Any], alive_allies: list[str]
) -> list[str]:
    return list(alive_allies)


def collect_invalid_runtime_units(
    battle_context: dict[str, Any],
) -> list[dict[str, str]]:
    invalid_units: list[dict[str, str]] = []

    for unit in get_units_from_context(battle_context, "allies"):
        unit_id = unit.get("unitId")
        if not isinstance(unit_id, str):
            continue

        if not is_alive(unit):
            invalid_units.append(
                {
                    "unitId": unit_id,
                    "side": "ally",
                    "reason": "dead",
                }
            )

    for unit in get_units_from_context(battle_context, "enemies"):
        unit_id = unit.get("unitId")
        if not isinstance(unit_id, str):
            continue

        reasons = []
        if not is_alive(unit):
            reasons.append("dead")
        if not is_targetable(unit):
            reasons.append("untargetable")

        if reasons:
            invalid_units.append(
                {
                    "unitId": unit_id,
                    "side": "enemy",
                    "reason": "+".join(reasons),
                }
            )

    return invalid_units


def analyze_command(battle_context: dict[str, Any]) -> dict[str, Any]:
    alive_allies = get_alive_allies(battle_context)
    dead_targetable_allies = get_dead_targetable_allies(battle_context)
    valid_attack_targets = get_valid_attack_targets(battle_context)
    allowed_actors = get_allowed_actors_from_runtime(battle_context, alive_allies)
    valid_move_to_units = alive_allies + valid_attack_targets
    invalid_units = collect_invalid_runtime_units(battle_context)

    return {
        "analysisMode": "runtime_constraint_summary",
        "description": "This object summarizes runtime-valid actors, targets, move destinations, and dead targetable allies. It does not parse or decide the user's intent.",
        "allowedActors": allowed_actors,
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


def build_user_message(battle_context: dict[str, Any]) -> str:
    model_input_context = build_model_input_context(battle_context)
    command_analysis = analyze_command(battle_context)

    output_schema_example = {
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

    user_message = {
        "input": model_input_context,
        "commandAnalysis": command_analysis,
        "output_schema_example": output_schema_example,
        "hard_constraints": [
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
        ],
    }

    return json.dumps(user_message, ensure_ascii=False, separators=(",", ":"))


def build_request_body(
    model_name: str,
    max_output_tokens: int,
    battle_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_message(battle_context),
            },
        ],
        "stream": True,
        "think": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.8,
            "num_predict": max_output_tokens,
            "num_ctx": 6000,
        },
        "keep_alive": "10m",
    }


def get_stream_piece(data: dict[str, Any]) -> str:
    message = data.get("message", {})

    content = message.get("content")
    if content:
        return content

    thinking = message.get("thinking")
    if thinking:
        return thinking

    response = data.get("response")
    if response:
        return response

    return ""


def extract_first_json_object(text: str) -> Optional[str]:
    start_index = text.find("{")
    if start_index < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start_index, len(text)):
        char = text[index]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start_index : index + 1]

    return None


def try_parse_json(raw_text: str) -> Optional[dict[str, Any]]:
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    candidates = [text]

    extracted = extract_first_json_object(text)
    if extracted is not None and extracted != text:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    return None


def actor_has_skill(battle_context: dict[str, Any], actor_id: str) -> bool:
    unit = get_unit_by_id(battle_context, actor_id)
    return bool(
        unit
        and isinstance(unit.get("skillDescription"), str)
        and unit.get("skillDescription")
    )


def get_actor_skill_description(
    battle_context: dict[str, Any], actor_id: str
) -> Optional[str]:
    unit = get_unit_by_id(battle_context, actor_id)
    if unit is None:
        return None

    value = unit.get("skillDescription")
    if isinstance(value, str) and value:
        return value

    return None


def can_actor_skill_target_dead(battle_context: dict[str, Any], actor_id: str) -> bool:
    unit = get_unit_by_id(battle_context, actor_id)
    return bool(unit and unit.get("canSkillTargetDead") is True)


def is_valid_skill_target(
    battle_context: dict[str, Any], actor_id: str, target_id: str
) -> bool:
    actor_unit = get_unit_by_id(battle_context, actor_id)
    target_unit = get_unit_by_id(battle_context, target_id)

    if actor_unit is None or target_unit is None:
        return False

    actor_skill_is_on_self = actor_unit.get("IsSkillOnSelf") is True
    actor_skill_is_on_other_ally = actor_unit.get("IsSkillOnOtherAlly") is True
    target_side = get_unit_side(battle_context, target_id)

    if actor_skill_is_on_self:
        if target_id != actor_id:
            return False
    else:
        if target_id == actor_id:
            return False

        if actor_skill_is_on_other_ally:
            if target_side != "ally":
                return False
        elif target_side != "enemy":
            return False

    if not is_targetable(target_unit):
        return False

    if is_alive(target_unit):
        return True

    return can_actor_skill_target_dead(battle_context, actor_id)


def get_allowed_attack_targets(command_analysis: dict[str, Any]) -> list[str]:
    allowed_attack_targets = command_analysis.get("allowedAttackTargets", [])
    if isinstance(allowed_attack_targets, list):
        return [target for target in allowed_attack_targets if isinstance(target, str)]

    return []


def get_valid_move_to_units(command_analysis: dict[str, Any]) -> list[str]:
    valid_move_to_units = command_analysis.get("validMoveToUnits", [])
    if isinstance(valid_move_to_units, list):
        return [unit_id for unit_id in valid_move_to_units if isinstance(unit_id, str)]

    return []


def sanitize_move(
    seq_item: dict[str, Any],
    actor_id: str,
    battle_context: dict[str, Any],
    command_analysis: dict[str, Any],
) -> Optional[dict[str, Any]]:
    subtype = seq_item.get("subtype")
    movement_type = seq_item.get("movementType")
    to_id = seq_item.get("to")

    if subtype not in ALLOWED_MOVE_SUBTYPES:
        return None

    if movement_type not in {"direct", "flank"}:
        movement_type = "direct"

    if not isinstance(to_id, str):
        return None

    valid_move_to_units = get_valid_move_to_units(command_analysis)

    if to_id == actor_id:
        return None

    if to_id not in valid_move_to_units:
        return None

    if not is_valid_move_to(battle_context, to_id):
        return None

    return {
        "type": "move",
        "subtype": subtype,
        "movementType": movement_type,
        "to": to_id,
    }


def sanitize_attack(
    seq_item: dict[str, Any],
    command_analysis: dict[str, Any],
) -> Optional[dict[str, Any]]:
    target = seq_item.get("target")
    allowed_attack_targets = get_allowed_attack_targets(command_analysis)

    if isinstance(target, str) and target in allowed_attack_targets:
        return {
            "type": "attack",
            "target": target,
        }

    return None


def sanitize_skill(
    seq_item: dict[str, Any],
    actor_id: str,
    battle_context: dict[str, Any],
    command_analysis: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not actor_has_skill(battle_context, actor_id):
        return None

    expected_description = get_actor_skill_description(battle_context, actor_id)
    if expected_description is None:
        return None

    description = seq_item.get("description")
    if description != expected_description:
        return None

    target = seq_item.get("target")
    if not isinstance(target, str):
        return None

    if not is_valid_skill_target(battle_context, actor_id, target):
        return None

    return {
        "type": "skill",
        "description": description,
        "target": target,
    }


def sanitize_wait(
    seq_item: dict[str, Any],
) -> Optional[dict[str, Any]]:
    duration = seq_item.get("durationSec")

    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        duration = DEFAULT_WAIT_SECONDS

    duration = clamp_wait_seconds(float(duration))

    if duration.is_integer():
        duration_value: int | float = int(duration)
    else:
        duration_value = duration

    return {
        "type": "wait",
        "durationSec": duration_value,
    }


def sanitize_skill_control(
    seq_item: dict[str, Any],
    actor_id: str,
    battle_context: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not actor_has_skill(battle_context, actor_id):
        return None

    mode = seq_item.get("mode")

    if mode == "forbid":
        return {
            "type": "skillControl",
            "mode": "forbid",
        }

    if mode != "defer":
        return None

    duration = seq_item.get("durationSec")

    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        duration = DEFAULT_SKILL_CONTROL_DEFER_SECONDS

    duration = clamp_wait_seconds(float(duration))

    if duration.is_integer():
        duration_value: int | float = int(duration)
    else:
        duration_value = duration

    return {
        "type": "skillControl",
        "mode": "defer",
        "durationSec": duration_value,
    }


def sanitize_sequence_item(
    seq_item: dict[str, Any],
    actor_id: str,
    battle_context: dict[str, Any],
    command_analysis: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not isinstance(seq_item, dict):
        return None

    action_type = seq_item.get("type")

    if action_type == "move":
        return sanitize_move(
            seq_item=seq_item,
            actor_id=actor_id,
            battle_context=battle_context,
            command_analysis=command_analysis,
        )

    if action_type == "attack":
        return sanitize_attack(
            seq_item=seq_item,
            command_analysis=command_analysis,
        )

    if action_type == "skill":
        return sanitize_skill(
            seq_item=seq_item,
            actor_id=actor_id,
            battle_context=battle_context,
            command_analysis=command_analysis,
        )

    if action_type == "wait":
        return sanitize_wait(seq_item=seq_item)

    if action_type == "skillControl":
        return sanitize_skill_control(
            seq_item=seq_item,
            actor_id=actor_id,
            battle_context=battle_context,
        )

    return None


def sanitize_parsed_output(
    parsed: dict[str, Any],
    battle_context: dict[str, Any],
    command_analysis: dict[str, Any],
) -> dict[str, Any]:
    thinking = parsed.get("thinking")
    if not isinstance(thinking, str):
        thinking = ""

    raw_action = parsed.get("action")
    if not isinstance(raw_action, list):
        raw_action = []

    allowed_actors = command_analysis.get("allowedActors", [])
    if not isinstance(allowed_actors, list):
        allowed_actors = []

    sanitized_action = []
    seen_actor_ids = set()

    for action_entry in raw_action:
        if not isinstance(action_entry, dict):
            continue

        actor_id = action_entry.get("unitId")
        sequence = action_entry.get("sequence")

        if not isinstance(actor_id, str):
            continue

        if actor_id in seen_actor_ids:
            continue

        if actor_id not in allowed_actors:
            continue

        if not is_alive_ally(battle_context, actor_id):
            continue

        if not isinstance(sequence, list):
            continue

        sanitized_sequence = []

        for seq_item in sequence:
            sanitized_item = sanitize_sequence_item(
                seq_item=seq_item,
                actor_id=actor_id,
                battle_context=battle_context,
                command_analysis=command_analysis,
            )

            if sanitized_item is None:
                continue

            if (
                sanitized_item["type"] == "wait"
                and sanitized_sequence
                and sanitized_sequence[-1]["type"] in {"attack", "skill"}
            ):
                continue

            sanitized_sequence.append(sanitized_item)

            if len(sanitized_sequence) >= MAX_ACTIONS_PER_ACTOR:
                break

        if not sanitized_sequence:
            continue

        sanitized_action.append(
            {
                "unitId": actor_id,
                "sequence": sanitized_sequence,
            }
        )
        seen_actor_ids.add(actor_id)

    sanitized_action_unit_ids = {entry["unitId"] for entry in sanitized_action}

    raw_dialog = parsed.get("dialog")
    if not isinstance(raw_dialog, list):
        raw_dialog = []

    sanitized_dialog = []

    for dialog_entry in raw_dialog:
        if not isinstance(dialog_entry, dict):
            continue

        unit_id = dialog_entry.get("unitId")
        text = dialog_entry.get("text")

        if unit_id not in sanitized_action_unit_ids:
            continue

        if not isinstance(text, str):
            text = ""

        sanitized_dialog.append(
            {
                "unitId": unit_id,
                "text": text,
            }
        )

    dialog_unit_ids = {entry["unitId"] for entry in sanitized_dialog}
    for action_entry in sanitized_action:
        unit_id = action_entry["unitId"]
        if unit_id in dialog_unit_ids:
            continue

        sanitized_dialog.append(
            {
                "unitId": unit_id,
                "text": "명령을 수행한다.",
            }
        )

    if not sanitized_action:
        sanitized_dialog = []

    return {
        "thinking": thinking,
        "dialog": sanitized_dialog,
        "action": sanitized_action,
    }


def validate_basic_contract(
    parsed: dict[str, Any],
    battle_context: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    ally_ids, enemy_ids, all_unit_ids = get_unit_ids(battle_context)

    expected_keys = {"thinking", "dialog", "action"}
    actual_keys = set(parsed.keys())
    if actual_keys != expected_keys:
        errors.append(f"top-level keys mismatch: {sorted(actual_keys)}")

    thinking = parsed.get("thinking")
    dialog = parsed.get("dialog")
    action = parsed.get("action")

    if not isinstance(thinking, str):
        errors.append("thinking is not a string")

    if not isinstance(dialog, list):
        errors.append("dialog is not a list")
        dialog = []

    if not isinstance(action, list):
        errors.append("action is not a list")
        action = []

    action_unit_ids: list[str] = []

    for action_entry in action:
        if not isinstance(action_entry, dict):
            errors.append("action item is not an object")
            continue

        if set(action_entry.keys()) != {"unitId", "sequence"}:
            errors.append(f"invalid action item keys: {sorted(action_entry.keys())}")

        unit_id = action_entry.get("unitId")
        sequence = action_entry.get("sequence")

        if unit_id not in ally_ids:
            errors.append(f"invalid action unitId: {unit_id}")

        if isinstance(unit_id, str):
            if unit_id in action_unit_ids:
                errors.append(f"duplicated action unitId: {unit_id}")
            action_unit_ids.append(unit_id)

        if not isinstance(sequence, list):
            errors.append(f"sequence is not a list: {unit_id}")
            continue

        if len(sequence) > MAX_ACTIONS_PER_ACTOR:
            errors.append(f"sequence too long: {unit_id}")

        for seq_item in sequence:
            if not isinstance(seq_item, dict):
                errors.append(f"sequence item is not an object: {unit_id}")
                continue

            action_type = seq_item.get("type")

            if action_type == "move":
                allowed_keys = {"type", "subtype", "movementType", "to"}
                if set(seq_item.keys()) != allowed_keys:
                    errors.append(f"invalid move keys: {sorted(seq_item.keys())}")

                subtype = seq_item.get("subtype")
                movement_type = seq_item.get("movementType")
                to_id = seq_item.get("to")

                if subtype not in ALLOWED_MOVE_SUBTYPES:
                    errors.append(f"invalid move subtype: {subtype}")

                if movement_type not in {"direct", "flank"}:
                    errors.append(f"invalid movementType: {movement_type}")

                if to_id not in all_unit_ids:
                    errors.append(f"invalid move to: {to_id}")
                elif not is_valid_move_to(battle_context, to_id):
                    errors.append(f"move to is not valid at runtime: {to_id}")
                elif to_id == unit_id:
                    errors.append(f"move to self is not allowed: {unit_id}")

            elif action_type == "attack":
                allowed_keys = {"type", "target"}
                if set(seq_item.keys()) != allowed_keys:
                    errors.append(f"invalid attack keys: {sorted(seq_item.keys())}")

                target = seq_item.get("target")
                if target not in enemy_ids:
                    errors.append(f"invalid attack target: {target}")
                elif not is_valid_enemy_target(battle_context, target):
                    errors.append(
                        f"attack target is not alive and targetable: {target}"
                    )

            elif action_type == "skill":
                allowed_keys = {"type", "description", "target"}
                if set(seq_item.keys()) != allowed_keys:
                    errors.append(f"invalid skill keys: {sorted(seq_item.keys())}")

                description = seq_item.get("description")
                if not isinstance(description, str):
                    errors.append("skill description is not a string")

                if isinstance(unit_id, str) and not actor_has_skill(
                    battle_context, unit_id
                ):
                    errors.append(f"skill actor has no skill: {unit_id}")
                elif isinstance(unit_id, str):
                    expected_description = get_actor_skill_description(
                        battle_context, unit_id
                    )
                    if description != expected_description:
                        errors.append(f"skill description mismatch: {unit_id}")

                target = seq_item.get("target")
                if target not in all_unit_ids:
                    errors.append(f"invalid skill target: {target}")
                elif isinstance(unit_id, str) and not is_valid_skill_target(
                    battle_context, unit_id, str(target)
                ):
                    errors.append(f"skill target is not valid at runtime: {target}")

            elif action_type == "wait":
                allowed_keys = {"type", "durationSec"}
                if set(seq_item.keys()) != allowed_keys:
                    errors.append(f"invalid wait keys: {sorted(seq_item.keys())}")

                duration = seq_item.get("durationSec")
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or duration < MIN_WAIT_SECONDS
                    or duration > MAX_WAIT_SECONDS
                ):
                    errors.append(f"invalid wait durationSec: {duration}")

            elif action_type == "skillControl":
                mode = seq_item.get("mode")

                if mode == "defer":
                    allowed_keys = {"type", "mode", "durationSec"}
                    if set(seq_item.keys()) != allowed_keys:
                        errors.append(
                            f"invalid skillControl defer keys: {sorted(seq_item.keys())}"
                        )

                    duration = seq_item.get("durationSec")
                    if (
                        isinstance(duration, bool)
                        or not isinstance(duration, (int, float))
                        or duration < MIN_WAIT_SECONDS
                        or duration > MAX_WAIT_SECONDS
                    ):
                        errors.append(f"invalid skillControl durationSec: {duration}")

                elif mode == "forbid":
                    allowed_keys = {"type", "mode"}
                    if set(seq_item.keys()) != allowed_keys:
                        errors.append(
                            f"invalid skillControl forbid keys: {sorted(seq_item.keys())}"
                        )

                else:
                    errors.append(f"invalid skillControl mode: {mode}")

                if isinstance(unit_id, str) and not actor_has_skill(
                    battle_context, unit_id
                ):
                    errors.append(f"skillControl actor has no skill: {unit_id}")

            else:
                errors.append(f"invalid sequence action type: {action_type}")

    for dialog_entry in dialog:
        if not isinstance(dialog_entry, dict):
            errors.append("dialog item is not an object")
            continue

        if set(dialog_entry.keys()) != {"unitId", "text"}:
            errors.append(f"invalid dialog item keys: {sorted(dialog_entry.keys())}")

        dialog_unit_id = dialog_entry.get("unitId")
        if dialog_unit_id not in action_unit_ids:
            errors.append(f"dialog unitId not found in action: {dialog_unit_id}")

        if not isinstance(dialog_entry.get("text"), str):
            errors.append(f"dialog text is not a string: {dialog_unit_id}")

    return errors


def validate_runtime_contract(
    parsed: dict[str, Any],
    battle_context: dict[str, Any],
    command_analysis: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    allowed_actors = set(command_analysis.get("allowedActors", []))
    allowed_attack_targets = set(command_analysis.get("allowedAttackTargets", []))
    valid_move_to_units = set(command_analysis.get("validMoveToUnits", []))

    action = parsed.get("action", [])
    if not isinstance(action, list):
        return ["action is not a list"]

    for action_entry in action:
        if not isinstance(action_entry, dict):
            continue

        actor_id = action_entry.get("unitId")
        if actor_id not in allowed_actors:
            errors.append(f"actor outside allowedActors: {actor_id}")

        sequence = action_entry.get("sequence", [])
        if not isinstance(sequence, list):
            continue

        for seq_item in sequence:
            if not isinstance(seq_item, dict):
                continue

            action_type = seq_item.get("type")

            if action_type == "attack":
                target = seq_item.get("target")
                if target not in allowed_attack_targets:
                    errors.append(
                        f"attack target outside allowedAttackTargets: {target}"
                    )

            elif action_type == "skill":
                description = seq_item.get("description")
                target = seq_item.get("target")

                if not isinstance(actor_id, str) or not actor_has_skill(
                    battle_context, actor_id
                ):
                    errors.append(f"skill actor has no skill: {actor_id}")
                else:
                    expected_description = get_actor_skill_description(
                        battle_context, actor_id
                    )
                    if description != expected_description:
                        errors.append(f"skill description mismatch: {actor_id}")

                    if not isinstance(target, str) or not is_valid_skill_target(
                        battle_context, actor_id, target
                    ):
                        errors.append(f"skill target is not valid at runtime: {target}")

            elif action_type == "move":
                subtype = seq_item.get("subtype")
                to_id = seq_item.get("to")

                if subtype not in ALLOWED_MOVE_SUBTYPES:
                    errors.append(
                        f"move subtype outside allowedMoveSubtypes: {subtype}"
                    )

                if to_id == actor_id:
                    errors.append(f"move to self: {actor_id}")

                if to_id not in valid_move_to_units:
                    errors.append(f"move to outside validMoveToUnits: {to_id}")

            elif action_type == "skillControl":
                if isinstance(actor_id, str) and not actor_has_skill(
                    battle_context, actor_id
                ):
                    errors.append(f"skillControl actor has no skill: {actor_id}")

    return errors


def ns_to_sec(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / 1_000_000_000


def format_optional_sec(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}s"


def is_json_equivalent(left: Any, right: Any) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(
        right, ensure_ascii=False, sort_keys=True
    )


def call_ollama(
    model_name: str,
    max_output_tokens: int,
    timeout: int,
    battle_context: dict[str, Any],
) -> dict[str, Any]:
    command_analysis = analyze_command(battle_context)

    request_body = build_request_body(
        model_name=model_name,
        max_output_tokens=max_output_tokens,
        battle_context=battle_context,
    )

    start_time = time.perf_counter()
    first_chunk_time: Optional[float] = None
    chunks: list[str] = []
    final_stats: dict[str, Any] = {}

    response = requests.post(
        OLLAMA_CHAT_URL,
        json=request_body,
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        data = json.loads(line)
        piece = get_stream_piece(data)

        if piece and first_chunk_time is None:
            first_chunk_time = time.perf_counter()

        if piece:
            chunks.append(piece)

        if data.get("done") is True:
            final_stats = data
            break

    end_time = time.perf_counter()
    raw_response = "".join(chunks)
    raw_parsed = try_parse_json(raw_response)

    if raw_parsed is None:
        sanitized_parsed = None
        contract_errors = ["JSON parse failed."]
        sanitizer_changed = False
    else:
        sanitized_parsed = sanitize_parsed_output(
            parsed=raw_parsed,
            battle_context=battle_context,
            command_analysis=command_analysis,
        )

        contract_errors = validate_basic_contract(sanitized_parsed, battle_context)
        contract_errors.extend(
            validate_runtime_contract(
                parsed=sanitized_parsed,
                battle_context=battle_context,
                command_analysis=command_analysis,
            )
        )
        sanitizer_changed = not is_json_equivalent(raw_parsed, sanitized_parsed)

    prompt_eval_duration_sec = ns_to_sec(final_stats.get("prompt_eval_duration"))
    eval_duration_sec = ns_to_sec(final_stats.get("eval_duration"))
    total_duration_sec = ns_to_sec(final_stats.get("total_duration"))
    load_duration_sec = ns_to_sec(final_stats.get("load_duration"))

    eval_count = final_stats.get("eval_count")
    decode_tokens_per_sec: Optional[float] = None
    if eval_count and eval_duration_sec and eval_duration_sec > 0:
        decode_tokens_per_sec = eval_count / eval_duration_sec

    return {
        "rawResponse": raw_response,
        "rawParsed": raw_parsed,
        "parsed": sanitized_parsed,
        "parseSuccess": raw_parsed is not None,
        "sanitizerChanged": sanitizer_changed,
        "contractErrors": contract_errors,
        "ttftSec": None if first_chunk_time is None else first_chunk_time - start_time,
        "totalResponseTimeSec": end_time - start_time,
        "responseChars": len(raw_response),
        "ollamaStats": {
            "promptEvalCount": final_stats.get("prompt_eval_count"),
            "evalCount": eval_count,
            "loadDurationSec": load_duration_sec,
            "promptEvalDurationSec": prompt_eval_duration_sec,
            "evalDurationSec": eval_duration_sec,
            "ollamaTotalDurationSec": total_duration_sec,
            "decodeTokensPerSec": decode_tokens_per_sec,
        },
    }


def unit_skill_label(unit: dict[str, Any]) -> str:
    if "skillDescription" not in unit:
        return "없음"

    description = str(unit["skillDescription"])
    if len(description) > 26:
        return description[:26] + "..."

    return description


def compact_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def collect_unit_ids_from_json_value(value: Any, known_unit_ids: set[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add_unit_id(unit_id: str) -> None:
        if unit_id in known_unit_ids and unit_id not in seen:
            found.append(unit_id)
            seen.add(unit_id)

    def walk(node: Any) -> None:
        if isinstance(node, str):
            if node in known_unit_ids:
                add_unit_id(node)
                return

            for match in UNIT_ID_PATTERN.findall(node):
                add_unit_id(match)
            return

        if isinstance(node, list):
            for item in node:
                walk(item)
            return

        if isinstance(node, dict):
            for item in node.values():
                walk(item)

    walk(value)
    return found


def raw_parsed_unit_ids_text(
    raw_parsed: Optional[dict[str, Any]], battle_context: dict[str, Any]
) -> str:
    if raw_parsed is None:
        return "none"

    _, _, all_unit_ids = get_unit_ids(battle_context)
    unit_ids = collect_unit_ids_from_json_value(raw_parsed, set(all_unit_ids))
    if not unit_ids:
        return "none"

    return ", ".join(unit_ids)


def append_raw_parsed_unit_ids(
    lines: list[str], result: dict[str, Any], battle_context: dict[str, Any]
) -> None:
    lines.append("### Raw Parsed Unit IDs")
    lines.append("")
    lines.append(raw_parsed_unit_ids_text(result["rawParsed"], battle_context))
    lines.append("")


def make_unit_table(units: list[dict[str, Any]], include_skill: bool) -> str:
    if include_skill:
        lines = [
            "| unitId | isAlive | canBeTargeted | isRanged | hpRatio | attackRatioToAvg | engagedByOpponentCount | teamFormationRole | closestTargetableOpponent | farthestTargetableOpponent | closestAliveAlly | farthestAliveAlly | skillDescription | IsSkillOnSelf | IsSkillOnOtherAlly | isSkillAoe | canSkillTargetDead |",
            "|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines = [
            "| unitId | isAlive | canBeTargeted | isRanged | hpRatio | attackRatioToAvg | engagedByOpponentCount | teamFormationRole |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]

    for unit in units:
        row = [
            str(unit["unitId"]),
            str(unit["isAlive"]).lower(),
            str(unit["canBeTargeted"]).lower(),
            str(unit.get("isRanged", False)).lower(),
            f"{float(unit['hpRatio']):.2f}",
            f"{float(unit['attackRatioToAvg']):.2f}",
            str(unit["engagedByOpponentCount"]),
            str(unit["teamFormationRole"]),
        ]

        if include_skill:
            row.append(compact_json_text(unit.get("closestTargetableOpponent")))
            row.append(compact_json_text(unit.get("farthestTargetableOpponent")))
            row.append(compact_json_text(unit.get("closestAliveAlly")))
            row.append(compact_json_text(unit.get("farthestAliveAlly")))
            row.append(unit_skill_label(unit))
            row.append(str(unit.get("IsSkillOnSelf", False)).lower())
            row.append(str(unit.get("IsSkillOnOtherAlly", False)).lower())
            row.append(str(unit.get("isSkillAoe", False)).lower())
            row.append(str(unit.get("canSkillTargetDead", False)).lower())

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def fenced_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def fenced_text(value: str) -> str:
    return "```\n" + value + "\n```"


def append_timing_block(lines: list[str], result: dict[str, Any]) -> None:
    stats = result["ollamaStats"]

    lines.append("### Timing")
    lines.append("")
    lines.append(f"- ttft_sec: {format_optional_sec(result['ttftSec'])}")
    lines.append(
        f"- total_response_time_sec: {format_optional_sec(result['totalResponseTimeSec'])}"
    )
    lines.append(f"- response_chars: {result['responseChars']}")
    lines.append(f"- prompt_eval_count: {stats['promptEvalCount']}")
    lines.append(f"- eval_count: {stats['evalCount']}")
    lines.append(
        f"- load_duration_sec: {format_optional_sec(stats['loadDurationSec'])}"
    )
    lines.append(
        f"- prompt_eval_duration_sec: {format_optional_sec(stats['promptEvalDurationSec'])}"
    )
    lines.append(
        f"- eval_duration_sec: {format_optional_sec(stats['evalDurationSec'])}"
    )
    lines.append(
        f"- ollama_total_duration_sec: {format_optional_sec(stats['ollamaTotalDurationSec'])}"
    )

    decode_tokens_per_sec = stats["decodeTokensPerSec"]
    if decode_tokens_per_sec is None:
        lines.append("- decode_tokens_per_sec: N/A")
    else:
        lines.append(f"- decode_tokens_per_sec: {decode_tokens_per_sec:.2f}")

    lines.append("")


def build_report(
    suite: dict[str, Any],
    results: list[dict[str, Any]],
    model_name: str,
    max_output_tokens: int,
    input_file: Path,
    started_at: datetime,
    finished_at: datetime,
) -> str:
    lines: list[str] = []

    command_order: list[str] = []
    scenario_counts: dict[str, int] = {}

    for item in results:
        command_id = item["commandId"]
        if command_id not in scenario_counts:
            command_order.append(command_id)
            scenario_counts[command_id] = 0
        scenario_counts[command_id] += 1

    command_number_map = {
        command_id: index + 1 for index, command_id in enumerate(command_order)
    }

    scenario_number_map: dict[tuple[str, str], int] = {}
    scenario_seen_counts: dict[str, int] = {}

    for item in results:
        command_id = item["commandId"]
        scenario_id = item["scenarioId"]

        scenario_seen_counts[command_id] = scenario_seen_counts.get(command_id, 0) + 1
        scenario_number_map[(command_id, scenario_id)] = scenario_seen_counts[
            command_id
        ]

        lines.append("# Gemma Ollama Battle Evaluation Report")
    lines.append("")

    lines.append("## Top-Level Scenario Review")
    lines.append("")

    for item in results:
        command_id = item["commandId"]
        command_text = item["commandText"]
        scenario_id = item["scenarioId"]
        scenario = item["scenario"]
        result = item["result"]

        command_no = command_number_map[command_id]
        scenario_no = scenario_number_map[(command_id, scenario_id)]
        scenario_total = scenario_counts[command_id]

        lines.append(
            f"### Original Command · Command {command_no}, Scenario {scenario_no}/{scenario_total} · {scenario_id}"
        )
        lines.append("")
        lines.append(command_text)
        lines.append("")

        lines.append("### Intended Situation")
        lines.append("")
        lines.append(str(scenario.get("intendedSituation", "")))
        lines.append("")

        lines.append("### Desirable Output")
        lines.append("")
        lines.append(str(scenario.get("desirableOutput", "")))
        lines.append("")

        lines.append("### Raw Parsed JSON")
        lines.append("")
        if result["rawParsed"] is None:
            lines.append("JSON parse failed.")
        else:
            lines.append(fenced_json(result["rawParsed"]))
        lines.append("")

        append_raw_parsed_unit_ids(lines, result, scenario["battleContext"])

    lines.append("---")
    lines.append("")

    lines.append("## Run Info")
    lines.append("")
    lines.append(f"- suite_name: {suite.get('suiteName', '')}")
    lines.append(f"- suite_description: {suite.get('description', '')}")
    lines.append(f"- model: {model_name}")
    lines.append(f"- endpoint: {OLLAMA_CHAT_URL}")
    lines.append(f"- max_output_tokens: {max_output_tokens}")
    lines.append("- format: not used")
    lines.append("- think: false")
    lines.append("- prompt_mode: full_json_field_names")
    lines.append("- command_analysis_mode: runtime_constraint_summary")
    lines.append("- sanitizer_mode: drop_invalid_runtime_actions_only")
    lines.append("- move_to_key: to")
    lines.append(f"- move_subtypes: {', '.join(sorted(ALLOWED_MOVE_SUBTYPES))}")
    lines.append("- movement_type: direct|flank")
    lines.append("- wait_action: enabled")
    lines.append("- skill_control_action: enabled")
    lines.append("- condition_policy: current_state_only")
    lines.append(f"- input_file: {input_file}")
    lines.append(f"- command_count: {len(command_order)}")
    lines.append(f"- scenario_count: {len(results)}")
    lines.append(f"- started_at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- finished_at: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## Report Structure")
    lines.append("")
    lines.append("이 리포트는 아래 순서로 읽으면 된다.")
    lines.append("")
    lines.append("1. **Run Info**: 실행 모델, 토큰 제한, 입력 파일, 실행 시각.")
    lines.append(
        "2. **Command / Scenario Index**: 어떤 명령에 어떤 전장 상황이 붙어 있는지 확인."
    )
    lines.append(
        "3. **Status Summary**: 전체 케이스의 JSON 파싱 성공 여부, sanitizer 변경 여부, 런타임 계약 검사 결과, 응답 시간 요약."
    )
    lines.append(
        "4. **Command Detail Sections**: 각 명령별 실제 입력 상황과 LLM 응답 확인."
    )
    lines.append("")
    lines.append("각 scenario 상세 섹션에는 다음 정보가 들어간다.")
    lines.append("")
    lines.append(
        "- **Allies / Enemies**: LLM에게 전달된 전장 정보에서 시스템 프롬프트를 제외하고 사람이 보기 좋게 정리한 정보."
    )
    lines.append(
        "- **Command Analysis**: 파이썬이 계산한 런타임 actor/target/action 허용 범위. 명령 의도를 파싱하지 않는다."
    )
    lines.append(
        "- **Intended Situation**: 이 테스트 케이스에서 의도한 전장 상황. LLM에게는 전달하지 않는다."
    )
    lines.append(
        "- **Desirable Output**: 사람이 기대하는 바람직한 출력. LLM에게는 전달하지 않는다."
    )
    lines.append("- **LLM Raw Response**: 모델이 실제로 반환한 원문.")
    lines.append("- **Raw Parsed JSON**: LLM 원문을 그대로 파싱한 JSON.")
    lines.append("- **Raw Parsed Unit IDs**: Raw Parsed JSON에 포함된 유닛 id 목록.")
    lines.append("- **Sanitized JSON**: 런타임 제약과 sanitizer를 거친 최종 JSON.")
    lines.append("- **Timing**: TTFT, 전체 응답 시간, Ollama 내부 토큰 처리 통계.")
    lines.append(
        "- **Sanitized Runtime Contract Check**: 최종 JSON의 구조와 런타임 유효성 검사 결과."
    )
    lines.append("")

    lines.append("## Command / Scenario Index")
    lines.append("")
    lines.append("| Command No. | Command ID | Scenario Count | Command Text |")
    lines.append("|---:|---|---:|---|")

    for command_id in command_order:
        first_item = next(item for item in results if item["commandId"] == command_id)
        lines.append(
            "| "
            f"{command_number_map[command_id]} | "
            f"{command_id} | "
            f"{scenario_counts[command_id]} | "
            f"{first_item['commandText']} |"
        )

    lines.append("")

    lines.append("## Status Summary")
    lines.append("")
    lines.append(
        "| Global Index | Command No. | Scenario No. | Command ID | Scenario ID | JSON Parse | Sanitizer Changed | Sanitized Runtime Contract Check | TTFT | Total Time | Chars |"
    )
    lines.append("|---:|---:|---:|---|---|---|---|---|---:|---:|---:|")

    for index, item in enumerate(results, start=1):
        result = item["result"]
        command_id = item["commandId"]
        scenario_id = item["scenarioId"]
        contract_status = "passed" if not result["contractErrors"] else "failed"
        parse_status = "success" if result["parseSuccess"] else "failed"
        sanitizer_status = "yes" if result.get("sanitizerChanged") else "no"

        lines.append(
            "| "
            f"{index} | "
            f"{command_number_map[command_id]} | "
            f"{scenario_number_map[(command_id, scenario_id)]}/{scenario_counts[command_id]} | "
            f"{command_id} | "
            f"{scenario_id} | "
            f"{parse_status} | "
            f"{sanitizer_status} | "
            f"{contract_status} | "
            f"{format_optional_sec(result['ttftSec'])} | "
            f"{format_optional_sec(result['totalResponseTimeSec'])} | "
            f"{result['responseChars']} |"
        )

    lines.append("")

    current_command_id: Optional[str] = None

    for item in results:
        command_id = item["commandId"]
        command_text = item["commandText"]
        scenario_id = item["scenarioId"]
        scenario = item["scenario"]
        result = item["result"]
        battle_context = scenario["battleContext"]
        command_no = command_number_map[command_id]
        scenario_no = scenario_number_map[(command_id, scenario_id)]
        scenario_total = scenario_counts[command_id]

        if command_id != current_command_id:
            current_command_id = command_id

            lines.append("---")
            lines.append("")
            lines.append(f"# Command {command_no} · {command_id}")
            lines.append("")
            lines.append("## Command Text")
            lines.append("")
            lines.append(command_text)
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(f"## Scenario {scenario_no}/{scenario_total} · {scenario_id}")
        lines.append("")
        lines.append(f"- title: {scenario.get('title', '')}")
        lines.append("")

        allies = get_units_from_context(battle_context, "allies")
        enemies = get_units_from_context(battle_context, "enemies")
        
        lines.append("### Allies")
        lines.append("")
        lines.append(make_unit_table(allies, include_skill=True))
        lines.append("")

        lines.append("### Enemies")
        lines.append("")
        lines.append(make_unit_table(enemies, include_skill=False))
        lines.append("")

        lines.append("### Model Input")
        lines.append("")
        lines.append(fenced_json(build_model_input_context(battle_context)))
        lines.append("")

        lines.append("### Command Analysis")
        lines.append("")
        lines.append(fenced_json(analyze_command(battle_context)))
        lines.append("")

        lines.append("### Original Command")
        lines.append("")
        lines.append(command_text)
        lines.append("")

        lines.append("### Intended Situation")
        lines.append("")
        lines.append(str(scenario.get("intendedSituation", "")))
        lines.append("")

        lines.append("### Desirable Output")
        lines.append("")
        lines.append(str(scenario.get("desirableOutput", "")))
        lines.append("")

        lines.append("### LLM Raw Response")
        lines.append("")
        lines.append(fenced_text(result["rawResponse"]))
        lines.append("")

        lines.append("### Raw Parsed JSON")
        lines.append("")
        if result["rawParsed"] is None:
            lines.append("JSON parse failed.")
        else:
            lines.append(fenced_json(result["rawParsed"]))
        lines.append("")

        append_raw_parsed_unit_ids(lines, result, battle_context)

        lines.append("### Sanitized JSON")
        lines.append("")
        if result["parsed"] is None:
            lines.append("JSON parse failed.")
        else:
            lines.append(fenced_json(result["parsed"]))
        lines.append("")

        append_timing_block(lines, result)

        lines.append("### Sanitized Runtime Contract Check")
        lines.append("")
        if not result["contractErrors"]:
            lines.append("passed")
        else:
            lines.append("failed")
            lines.append("")
            for error in result["contractErrors"]:
                lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines)


def collect_jobs(suite: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    commands = suite["commands"]

    for command in commands:
        command_id = command.get("commandId")
        command_text = command.get("commandText")
        scenarios = command.get("scenarios")

        if not isinstance(command_id, str):
            raise ValueError("Each command must have string commandId.")

        if not isinstance(command_text, str):
            raise ValueError(f"Command {command_id} must have string commandText.")

        if not isinstance(scenarios, list):
            raise ValueError(f"Command {command_id} must have scenarios list.")

        for scenario in scenarios:
            scenario_id = scenario.get("scenarioId")
            battle_context = scenario.get("battleContext")

            if not isinstance(scenario_id, str):
                raise ValueError(
                    f"Command {command_id} has scenario without string scenarioId."
                )

            if not isinstance(battle_context, dict):
                raise ValueError(
                    f"Scenario {scenario_id} must have battleContext object."
                )

            if battle_context.get("command") != command_text:
                battle_context["command"] = command_text

            jobs.append(
                {
                    "commandId": command_id,
                    "commandText": command_text,
                    "scenarioId": scenario_id,
                    "scenario": scenario,
                    "battleContext": battle_context,
                }
            )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gemma4:e2b",
        choices=["gemma4:e2b", "gemma4:e4b"],
        help="Ollama model name.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=500,
        help="Maximum generated tokens. Ollama option: num_predict.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout seconds.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help="Input JSON file path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory path.",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)

    suite = load_eval_suite(input_file)
    jobs = collect_jobs(suite)

    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now()
    results: list[dict[str, Any]] = []

    print("=== BATCH REQUEST ===")
    print(f"model: {args.model}")
    print(f"endpoint: {OLLAMA_CHAT_URL}")
    print(f"max_output_tokens: {args.max_output_tokens}")
    print("format: not used")
    print("think: false")
    print("prompt_mode: full_json_field_names")
    print("command_analysis_mode: runtime_constraint_summary")
    print("sanitizer_mode: drop_invalid_runtime_actions_only")
    print("move_to_key: to")
    print(f"move_subtypes: {', '.join(sorted(ALLOWED_MOVE_SUBTYPES))}")
    print("movement_type: direct|flank")
    print("wait_action: enabled")
    print("skill_control_action: enabled")
    print("condition_policy: current_state_only")
    print(f"input_file: {input_file}")
    print(f"job_count: {len(jobs)}")
    print()

    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] "
            f"command={job['commandId']} scenario={job['scenarioId']}"
        )

        scenario_started = time.perf_counter()

        try:
            result = call_ollama(
                model_name=args.model,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                battle_context=job["battleContext"],
            )
        except Exception as error:
            elapsed = time.perf_counter() - scenario_started
            result = {
                "rawResponse": "",
                "rawParsed": None,
                "parsed": None,
                "parseSuccess": False,
                "sanitizerChanged": False,
                "contractErrors": [f"Request failed: {repr(error)}"],
                "ttftSec": None,
                "totalResponseTimeSec": elapsed,
                "responseChars": 0,
                "ollamaStats": {
                    "promptEvalCount": None,
                    "evalCount": None,
                    "loadDurationSec": None,
                    "promptEvalDurationSec": None,
                    "evalDurationSec": None,
                    "ollamaTotalDurationSec": None,
                    "decodeTokensPerSec": None,
                },
            }

        parse_status = "success" if result["parseSuccess"] else "failed"
        contract_status = "passed" if not result["contractErrors"] else "failed"
        sanitizer_status = "changed" if result.get("sanitizerChanged") else "unchanged"

        print(
            "  "
            f"parse={parse_status}, "
            f"sanitizer={sanitizer_status}, "
            f"contract={contract_status}, "
            f"ttft={format_optional_sec(result['ttftSec'])}, "
            f"total={format_optional_sec(result['totalResponseTimeSec'])}, "
            f"chars={result['responseChars']}"
        )

        results.append(
            {
                "commandId": job["commandId"],
                "commandText": job["commandText"],
                "scenarioId": job["scenarioId"],
                "scenario": job["scenario"],
                "result": result,
            }
        )

    finished_at = datetime.now()

    report = build_report(
        suite=suite,
        results=results,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
        input_file=input_file,
        started_at=started_at,
        finished_at=finished_at,
    )

    timestamp = finished_at.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{slugify_model_name(args.model)}_{timestamp}.md"

    output_file.write_text(report, encoding="utf-8")

    print()
    print("=== DONE ===")
    print(f"output_file: {output_file}")


if __name__ == "__main__":
    main()