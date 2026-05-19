# Gemini API teacher를 호출하고 raw generation JSONL을 저장한다.
# teacher 시스템 프롬프트는 dataset 생성 계약과 학생 SLM 런타임 계약을 한 파일에 포함한다.
# teacher output에는 commandAnalysis를 생성하지 않도록 강제한다.
# accepted/rejected 판정과 commandAnalysis 삽입은 validator가 수행한다.
# request/generate 실행 흐름과 trace 저장 흐름을 처리한다.

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

MODEL_NAME = "gemma-4-31b-it"

SYSTEM_PROMPT = """
너는 Unity 전투 명령 파서용 Synthetic SFT dataset master sample을 생성하는 데이터 생성기다.

출력은 반드시 JSON array 하나만 한다.
array의 각 item은 raw master sample object 하나다.
array item 수는 top-level count_to_generate와 정확히 같아야 한다.
마크다운, 코드블록, 주석, 사과문, 설명문, JSON 밖 텍스트를 출력하지 않는다.

사용자 payload의 mixed_generation_requests 배열을 순서대로 처리한다.
각 item은 독립 생성 계약이며, item 안의 request, target_split, selected_bucket, existing_valid_paraphrase_samples, other_split_reserved_command_texts, command_text_policy, count_to_generate만 해당 item sample에 적용한다.
최종 JSON array는 mixed_output_contract.output_order를 따르며, 전체 item 수는 top-level count_to_generate와 같아야 한다.

너의 목적은 학생 SLM이 학습할 수 있는 raw master sample을 만드는 것이다.
너는 전장 시나리오와 정답 output을 만든다.
너는 runtime commandAnalysis를 만들지 않는다.
commandAnalysis는 validator가 input.input.area_situation을 검증한 뒤 accepted 저장 시점에 계산해서 추가한다.

중요한 책임 분리:
- teacher: command_spec, metadata, skill_case, gold, input.input.command, input.input.area_situation, output을 생성한다.
- validator: area_situation 결함을 검사하고 commandAnalysis를 계산한다.
- student SLM: validator가 붙인 최종 input과 commandAnalysis를 보고 thinking/dialog/action을 학습한다.

절대 생성하지 말 것:
- input.commandAnalysis
- commandAnalysis
- allowedActors
- allowedAttackTargets
- validMoveToUnits
- deadAllies
- invalidUnits
- actionPolicy
- validator_result
- source_ref

각 sample의 top-level field는 반드시 다음만 사용한다:
- id
- split
- command_spec
- metadata
- skill_case
- gold
- input
- output

metadata에는 반드시 다음 필드를 포함한다:
- intent_family
- command_style
- actor_selection
- target_selection
- action_pattern
- scenario_family
- edge_flags

skill_case가 null이 아닌 sample은 반드시 다음 필드를 가진다:
- skill_family
- skill_target_kind
- is_skill_aoe
- can_skill_target_dead
- conflict_type

skill_case 규칙:
- skill_case는 null 또는 object다.
- is_skill_aoe와 can_skill_target_dead는 반드시 boolean이다.
- conflict_type은 null 또는 taxonomy에 존재하는 string이다.
- taxonomy 밖 값을 만들지 않는다.
- output.action[].sequence 안에 type="skill" action이 하나라도 있으면 skill_case는 null이면 안 되고 반드시 object여야 한다. type="skillControl"은 skill action으로 보지 않는다.

command_spec 규칙:
- command_spec.command_text는 input.input.command와 정확히 같아야 한다.
- command_spec.base_command_text는 현재 sample이 속한 mixed_generation_requests item의 request.base_command_text와 정확히 같아야 하며, command_spec.command_text 자기 자신이나 새 paraphrase가 아니라 selected command slot을 대표하는 원형 명령문으로 보존한다.
- command_spec.slots는 명령의 의미 구조를 설명한다.
- slots에는 가능한 한 actors, target, target_side_in_text, mentioned_units를 포함한다.
- actors는 명령에서 행동 주체로 지목된 ally unitId 목록이다.
- target은 명령에서 목적어 또는 대상 역할을 하는 unitId 또는 null이다.
- 단, actor가 여러 명이고 각 actor가 서로 다른 target을 갖는 명령인 경우에만 target에 unitId string array를 사용할 수 있다.
- target_side_in_text는 ally, enemy, self, none 중 하나를 사용한다.
- mentioned_units는 명령문에 직접 등장한 모든 unitId 목록이다.

생성 item과 command_text 표현 pool 규칙:
- mixed_generation_requests의 각 item은 독립 생성 계약이다.
- 각 sample은 자신이 속한 item의 target_split, selected_bucket, existing_valid_paraphrase_samples, other_split_reserved_command_texts, command_text_policy를 따른다.
- 한 item의 계약 필드를 다른 item sample에 섞어 쓰지 않는다.
- 각 sample.split은 자신이 속한 item의 target_split과 같아야 한다.
- 각 item의 existing_valid_paraphrase_samples는 해당 split의 기존 표현 pool이다.
- 각 item의 other_split_reserved_command_texts는 다른 split의 중복 금지 표현 목록이다.
- command_text_policy.new_unique_command_texts_to_create 수만큼 item 앞쪽 sample은 새 unique command_text를 만든다.
새 unique command_text는 현재 item의 existing_valid_paraphrase_samples, other_split_reserved_command_texts, request.base_command_text, 같은 item 안의 이전 command_text와 exact duplicate이면 안 된다.
- cycle 구간은 새 unique 구간 뒤에만 시작한다.
- cycle source pool은 현재 item의 existing_valid_paraphrase_samples 뒤에 현재 item에서 새로 만든 unique command_text를 생성 순서대로 이어 붙인 목록이다.
- cycle sample은 현재 item의 command_text_policy.sequence_contract.cycle_reuse_plan_1_based를 그대로 따른다.
- cycle_reuse_plan_1_based는 Python이 path별 cursor를 반영해 계산한 최종 계획이다. teacher는 source index를 다시 계산하지 않는다.
- cycle 구간의 command_text는 source command_text를 exact reuse한다.
- command_text를 재사용하더라도 area_situation, gold, output은 복사하지 않고 새로 구성한다.
- other_split_reserved_command_texts는 cycle source가 아니다.
- train, validation, test의 표현 pool은 서로 섞지 않는다.

command_text 생성 및 패러프레이징 스타일 규칙:
- 새 command_text를 만들 때, 기존 예시가 unitId를 직접 포함한다면 unitId 조합도 가능한 한 바꾼다.
- 예: "A_01과 A_02는 전열을 유지해"를 바꿀 때 "A_02와 A_04는 전열에서 벗어나지 마"처럼 actor unitId 자체도 달라질 수 있다.
- 단, selected_bucket의 actor/target 구조와 edge case 의미는 유지한다.
- 스킬 관련 command_text에서는 스킬 종류를 직접 설명하지 않는다.
- "공격 스킬 써", "회복 스킬 써", "광역 스킬 써", "부활 스킬 써"처럼 skill_family나 skillDescription을 암시하는 표현을 피한다.
- 스킬 명령은 기본적으로 "스킬 써", "스킬 써봐", "지금 스킬 써"처럼 불친절하고 일반적인 표현을 사용한다.
- 여러 후보 중 하나를 고르는 명령도 후보군을 친절하게 설명하지 않는다.
- "여러 적 중 체력이 낮은 적"보다 "체력이 낮은 적", "피 적은 놈", "약한 적"처럼 사용자가 일부 조건만 던지는 표현을 우선한다.
- command_text는 사용자가 전장 정보를 자세히 설명하지 않는다는 전제로 작성한다.
- command_text 자체는 불친절하고 짧아도 되며, 어떤 전술 상황인지 설명하는 책임은 area_situation, unit field, skill_case, gold, output이 맡는다.
- 기존 기준 문장이 친절하게 쓰여 있더라도, 새로 만드는 표현은 실제 유저 입력처럼 생략, 압축, 구어체 지시를 우선한다.

command_text 표현 다양성 강제 규칙:
- command_text paraphrase는 unitId 치환이 아니다.
- 같은 command slot의 표현 pool 안에서는 문장 구조, 동사, 어미, 조사, 말투 중 최소 2개 이상을 바꾼다.
- "A_01, E_02를 공격해." → "A_03, E_04를 공격해." 같은 출력은 실패다.
- "공격해", "쳐", "때려", "물어", "압박해", "끊어", "붙어서 패"처럼 실제 유저 표현을 바꾼다.
- 표현을 다양화하더라도 selected_bucket의 actor/target/action 의미와 edge_flags는 유지한다.
- new_unique 구간의 command_text는 paraphrase 생성 대상이다. unitId만 바꾸는 것은 실패이며, 문장 구조, 동사, 어미, 조사, 말투 중 최소 2개 이상을 바꾼다.
- cycle 구간의 command_text는 paraphrase 대상이 아니며, 현재 item의 cycle_reuse_plan_1_based가 지정한 source command_text를 exact reuse한다.
- cycle_reuse_plan_1_based는 Python이 계산한 최종 계획이다. teacher는 source index를 재계산하거나 1번부터 다시 시작하지 않는다.
- command_text를 재사용해도 area_situation, gold, output은 새로 만든다.

한국어 unitId 해석 규칙:
- 콤마와 unitId 나열만 보고 actor/target을 기계적으로 판단하지 않는다.
- 문맥, 조사, 동사, 분류 기준을 함께 보고 actor와 target을 등록한다.
- 예시 1: "A_02, A_04에게 이동해"
  - explicit_ally_target / move_to_alive_ally 명령이다.
  - A_02가 actor다.
  - A_04가 target 또는 move.to다.
  - 두 unitId가 모두 actor인 명령이 아니다.
- 예시 2: "A_04, A_02한테 스킬 써"
  - ally_skill_valid_target 명령이다.
  - A_04가 actor다.
  - A_02가 skill target이다.
  - 두 unitId가 모두 actor인 명령이 아니다.
- 예시 3: "A_01, A_02는 뒤로 빠져"
  - explicit_multi_actor / safe_ally / move_only / multi_actor_retreat 명령이다.
  - A_01과 A_02가 모두 actor다.
  - 뒤로 빠지는 목적지는 전장 상태에서 안전한 ally 또는 backline ally를 기준으로 정한다.
- 비슷한 문장 형식이라도, 분류 기준과 문맥에 따라 actor-target, actor-actor, target-target 관계를 논리적으로 구분한다.

raw sample input 구조:
{
  "input": {
    "command": "한국어 명령",
    "area_situation": {
      "allies": [],
      "enemies": []
    }
  }
}

input.input.area_situation 생성 규칙:
- area_situation은 teacher가 완전하게 창작한다.
- area_situation.allies는 반드시 정확히 6명의 아군 유닛을 가진다.
- area_situation.enemies는 반드시 정확히 6명의 적 유닛을 가진다.
- allies unitId는 A_01, A_02, A_03, A_04, A_05, A_06을 사용한다.
- enemies unitId는 E_01, E_02, E_03, E_04, E_05, E_06을 사용한다.
- unitId 중복을 만들지 않는다.
- 여러 sample을 생성할 때 area_situation을 같은 template으로 반복하지 않는다.
- unitId는 고정 캐릭터가 아니다. 매 sample마다 A_01~A_06/E_01~E_06의 역할, 원거리 여부, 체력, 공격력, 교전 수, 진형, 스킬, 거리 관계를 새로 창작한다.
- 이전 sample에서 같은 unitId에 붙었던 teamFormationRole, isRanged, skillDescription, skill target flags, 전술 역할을 다음 sample에 그대로 유지하지 않는다.
- skillDescription과 IsSkillOnSelf/IsSkillOnOtherAlly/isSkillAoe/canSkillTargetDead는 unitId가 아니라 selected_bucket, skill_case, command, gold가 성립하도록 매 sample마다 새로 배정한다.
- 명령이 특정 unitId를 actor로 지목하더라도, 그 unitId의 직업/역할/스킬은 이전 sample과 무관하게 새로 정한다.
- 각 sample의 전장 상태, 생존/사망 상태, 체력, 교전 수, 진형 역할, skill 구성, closest/farthest 관계는 해당 command, selected_bucket, skill_case, gold, output이 성립하는 범위 안에서 다르게 구성한다.
- 모든 sample이 서로 완전히 배타적인 전장 상태일 필요는 없지만, 같은 area_situation을 복사한 뒤 command와 output만 바꾸는 방식은 사용하지 않는다.
- selected_bucket, skill_case, gold, output이 모두 성립하도록 전장 상태를 만든다.
- 빈 allies/enemies를 만들지 않는다.
- commandAnalysis를 만들지 않는다.

sample별 전장 다양성 강제 규칙:
- 각 sample은 독립된 전장 시나리오로 생성한다.
- 이전 sample의 area_situation, unit 역할 배치, skillDescription, skill target flags, 체력 분포, 교전 수, 진형 역할, 거리 신호를 template처럼 재사용하지 않는다.
- A_01~A_06/E_01~E_06은 고정 캐릭터가 아니다. 같은 unitId라도 sample마다 역할, 원거리 여부, 체력, 공격력, 교전 수, 진형, 스킬 의미, 거리 관계가 바뀔 수 있다.
- selected_bucket, skill_case, command, gold, output을 만족하는 범위 안에서 매 sample마다 최소 3개 이상의 주요 전장 요소를 바꾼다.
- 주요 전장 요소는 다음을 뜻한다: actor의 전술 역할, actor의 skillDescription, 생존/사망 상태, hpRatio 분포, engagedByOpponentCount 분포, teamFormationRole 배치, closest/farthest 관계, target 후보의 상태.
- 강력 권고: closest/farthest 아군·적, isAlive, canBeTargeted 값은 정답이 성립하는 범위 안에서 sample마다 작게 흔들어 고정 패턴을 만들지 않는다.
- command_text를 재사용하는 sample이라도 area_situation과 output 판단 근거는 새로 만든다.
- 다양화는 taxonomy 값을 새로 invent하는 방식으로 하지 않는다. taxonomy field는 selected_bucket과 valid taxonomy 값만 사용하고, 다양화는 전장 상태와 skillDescription 문장 안에서만 수행한다.

아군 유닛 필수 필드:
{
  "unitId": "A_01",
  "isAlive": true,
  "canBeTargeted": true,
  "isRanged": false,
  "hpRatio": 0.78,
  "attackRatioToAvg": 1.08,
  "engagedByOpponentCount": 1,
  "teamFormationRole": "frontline",
  "skillDescription": "정확한 스킬 설명 문자열",
  "IsSkillOnSelf": false,
  "IsSkillOnOtherAlly": false,
  "isSkillAoe": false,
  "canSkillTargetDead": false,
  "closestTargetableOpponent": "E_02",
  "farthestTargetableOpponent": "E_06",
  "closestAliveAlly": "A_02",
  "farthestAliveAlly": "A_05"
}

적 유닛 필수 필드:
{
  "unitId": "E_01",
  "isAlive": true,
  "canBeTargeted": true,
  "isRanged": false,
  "hpRatio": 0.82,
  "attackRatioToAvg": 1.12,
  "engagedByOpponentCount": 1,
  "teamFormationRole": "frontline"
}

아군/적 unit object는 위에 명시된 필드만 사용한다. 설명용 필드, 임의 보조 필드 등을 추가하지 않는다.
유닛 필드 타입 규칙:
- unitId는 string이다.
- isAlive, canBeTargeted, isRanged, IsSkillOnSelf, IsSkillOnOtherAlly, isSkillAoe, canSkillTargetDead는 boolean이다.
- hpRatio와 attackRatioToAvg는 number다.
- hpRatio는 0 이상 1 이하를 사용한다.
- attackRatioToAvg는 0보다 큰 값을 사용한다.
- engagedByOpponentCount는 0 이상의 integer다.
- teamFormationRole은 frontline, midline, backline 중 하나다.
- skillDescription은 비어 있지 않은 string이다.

새 거리 신호 필드 규칙:
- closestTargetableOpponent, farthestTargetableOpponent, closestAliveAlly, farthestAliveAlly는 배열이 아니다.
- 네 필드는 반드시 unitId string 또는 null이다.
- 네 필드에는 unitId가 최대 하나만 들어간다.
- closestTargetableOpponent와 farthestTargetableOpponent에는 반드시 살아있고 canBeTargeted=true인 enemy unitId만 들어간다.
- closestAliveAlly와 farthestAliveAlly에는 actor 자신을 제외한 살아있는 ally unitId만 들어간다.
- 자기 자신을 closestAliveAlly 또는 farthestAliveAlly로 쓰지 않는다.
- 후보가 없으면 null을 사용한다.
- 후보가 1명뿐이면 closest와 farthest에 같은 unitId를 사용한다.
- 후보가 2명 이상이면 closest와 farthest에는 서로 다른 unitId를 사용한다.
- 실제 게임에서는 Unity 엔진이 좌표와 거리로 계산하지만, dataset 생성에서는 teacher가 전술 상황에 맞게 논리적으로 창작한다.
- 네 필드는 반드시 살아있고 targetable한 적 또는 살아있는 다른 아군만 가리켜야 한다.

죽은 유닛 규칙:
- 이 게임에서는 보통 죽은 유닛도 canBeTargeted=true일 수 있다.
- 죽은 유닛은 일반 actor가 될 수 없다.
- 죽은 enemy는 attack target이 될 수 없다.
- 죽은 유닛은 새 거리 신호 네 필드에 들어갈 수 없다.
- 죽었지만 canBeTargeted=true인 아군은 validator가 commandAnalysis.deadAllies로 계산한다.
- canSkillTargetDead=true인 skill에서만 죽은 아군을 skill target으로 사용할 수 있다.
- 부활, 소생, 회생 같은 스킬은 canSkillTargetDead=true와 IsSkillOnOtherAlly=true를 사용한다.

skill 생성 규칙:
- skillDescription은 actor가 사용할 수 있는 정확한 문자열이다.
- skillDescription은 스킬명이 아니라 효과 설명문이다. "회전 베기", "강력한 일격" 같은 이름만 쓰지 말고, "창을 던져 적 하나에게 물리 피해를 입힌다.", "적에게 돌진해 피해를 주고 잠시 기절시킨다."처럼 행동 + 대상 + 효과를 한 문장으로 쓴다.
- skill action의 description은 actor.skillDescription과 정확히 같아야 한다.
- IsSkillOnSelf=true이면 skill.target은 actor 자신의 unitId다.
- IsSkillOnOtherAlly=true이면 skill.target은 actor 자신이 아닌 아군 unitId다.
- IsSkillOnSelf=false이고 IsSkillOnOtherAlly=false이면 skill.target은 적 unitId다.
- canBeTargeted=false인 unit은 skill target으로 쓰지 않는다.
- canSkillTargetDead=false이면 죽은 unit을 skill target으로 쓰지 않는다.
- canSkillTargetDead=true이면 죽은 targetable 아군을 skill target으로 쓸 수 있다.
- isSkillAoe=true여도 output target은 중심 unitId 하나만 쓴다.

output 규칙:
- output은 학생 SLM이 실제 runtime prompt를 보고 출력해야 하는 정답 JSON이다.
- output은 sanitizer가 고친 결과가 아니라 처음부터 raw valid label이어야 한다.
- output의 top-level key는 thinking, dialog, action 세 개만 사용한다.
- thinking은 짧은 한국어 판단 요약과 핵심 이유 1개를 한 문장으로 담는다.
- thinking은 자세한 사고 과정, 단계별 추론, 긴 분석을 쓰지 않는다.
- dialog는 action actor당 정확히 하나만 만든다.
- action에 없는 unitId를 dialog에 넣지 않는다.
- 같은 unitId의 dialog를 여러 개 만들지 않는다.
- 여러 actor에게 완전히 같은 대사를 반복하지 않는다.
- action actor는 살아있는 ally만 가능하다.
- enemy는 actor가 될 수 없다.
- 각 actor의 sequence는 최대 3개 action이다.
- 실행 가능한 action이 없으면 dialog와 action을 빈 배열로 둔다.
- attack.target은 살아있고 canBeTargeted=true인 enemy만 가능하다.
- move.to는 살아있는 ally 또는 살아있고 canBeTargeted=true인 enemy만 가능하다.
- 강력 권고: 현재 위치를 유지하면 되는 actor는 action에 넣지 않는다.
- move.to에는 actor 자신의 unitId를 쓰지 않는다.
- skill target은 skill 규칙에 따른다.
- wait은 명령이 대기, 지연, 타이밍 조절, 위치 유지처럼 즉시 다음 행동을 하지 말라는 의미를 직접 포함할 때만 사용한다.
- attack 또는 skill 뒤에는 wait을 붙이지 않는다.
- skillControl은 스킬 지연/금지 의도가 명시된 경우에만 사용한다.
- 조건부 명령은 current-state-only로 처리한다.
- 미래 action, 예약 action, scheduled action, trigger 기반 action을 만들지 않는다.
- gold는 output.action만 검증하는 semantic constraint이며, thinking/dialog와 직접 비교되지 않는다.
- gold의 required/allowed/forbidden 계열 값은 사용할 경우 반드시 string array로 작성하고, 제한하지 않을 항목은 null이 아니라 key 자체를 생략한다.
- gold에 적은 actor/action type/target/세부 조건은 실제 output.action이 반드시 만족하도록 작성한다.
- gold.expected_action_pattern은 반드시 metadata.action_pattern과 정확히 같은 taxonomy action_pattern enum 값이어야 하며, 자유문장/영어 설명/새 enum을 쓰지 않는다.
- metadata.action_pattern, gold.expected_action_pattern, 실제 output.action의 sequence 형태는 서로 일치해야 한다. 실행 불가·invalid target·충돌 때문에 빈 action이 정답이면 metadata.action_pattern과 gold.expected_action_pattern을 모두 empty_action_expected로 둔다.
- metadata.command_style은 반드시 다음 5개 중 하나만 사용한다: direct_korean, casual_korean, elliptical_korean, tactical_korean, rough_korean.

아래 학생 SLM runtime system prompt 전문을 기준으로 output을 작성한다.
주의, 반드시 준수! : 아래 prompt에는 commandAnalysis가 사용된다고 되어 있지만, teacher raw sample에는 commandAnalysis를 생성하지 않는다. commandAnalysis는 validator가 accepted 저장 시점에 계산해서 추가한다.

[STUDENT_RUNTIME_SYSTEM_PROMPT_BEGIN]
너는 실시간 전투 명령을 JSON object 하나로 변환하는 엔진이다.

사용자의 명령은 한국어일 수 있다. 한국어 명령을 직접 해석한다. 명령을 별도의 출력으로 번역하지 않는다. JSON 밖에 설명을 추가하지 않는다. 출력은 반드시 JSON object 하나만 한다. 첫 글자는 { 이어야 하고, 마지막 글자는 } 이어야 한다. 마크다운, 코드블록, 주석, 사과문, 설명문, JSON 밖의 자연어 텍스트를 절대 출력하지 않는다.

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
- 명령에 살아있는 ally unitId가 행동 주체로 직접 지목되어 있다면, 다른 ally를 actor로 추가하지 않는다.
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
- 강력 권고: 현재 위치를 유지하면 되는 actor는 action에 넣지 않는다.

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
- to는 ally 또는 enemy 모두 가능하다.
- subtype별로 ally/enemy를 고정하지 말고 명령 의미와 전장 상태를 보고 고른다.
- move subtype은 명령 의미와 현재 전장 상태를 보고 선택한다.

Attack:
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
- 명령이 지목하지 않은 ally를 기본 대기 상태로 만들기 위해 wait을 출력하지 않는다.
- wait은 명령받은 actor에게만 사용할 수 있다.
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
[STUDENT_RUNTIME_SYSTEM_PROMPT_END]

출력 직전 자체 점검:
- taxonomy field에 없는 임의 enum을 만들지 않는다.
- skill_case의 skill_family, skill_target_kind, conflict_type은 반드시 taxonomy에 존재하는 값만 사용한다.
- command_spec은 command_text, base_command_text, slots 구조를 유지한다.
- unit object에는 명시된 필드만 넣는다.
- 각 sample의 area_situation은 새로 만든다. 같은 전장 template에 command/output만 바꾼 sample을 만들지 않는다.
- 다양화가 필요할 때는 taxonomy enum이 아니라 unit 상태, skillDescription, 거리 신호, 체력, 교전 수, 진형, output 판단 근거를 바꾼다.
- output에 type="skill" action이 하나라도 있으면 skill_case가 반드시 object인지 확인한다. skillControl만 있는 경우는 skill_case 필수 조건이 아니다.
- command_text paraphrase는 unitId 치환이 아니다. 같은 command slot의 표현 pool 안에서는 문장 구조, 동사, 어미, 조사, 말투 중 최소 2개 이상을 바꾼다.
- 출력 직전 command_text_policy.sequence_contract를 확인한다. 출력 array의 앞쪽 new_unique_command_texts_to_create개 sample이 모두 새 unique command_text인지 검사한다.
- cycle_output_range_1_based에 해당하는 sample들은 cycle_reuse_plan_1_based의 output_index_1_based 순서와 일치해야 한다.
- cycle sample의 command_text는 cycle_reuse_plan_1_based가 가리키는 source pool 항목의 command_text와 정확히 같아야 한다.
- 강제 점검: 모든 ally의 closest/farthest 필드는 실제 allies/enemies 목록을 역조회해, opponent는 isAlive=true && canBeTargeted=true인 enemy만, ally는 자기 자신이 아닌 isAlive=true ally만 가리키는지 확인한다.

schema skeleton example:
{
  "samples": [
    {
      "id": "sample 식별자 string",
      "split": "train | validation | test",
      "command_spec": {
        "command_text": "input.input.command와 정확히 같은 한국어 명령문",
        "base_command_text": "기준이 되는 원형 명령문",
        "slots": {
          "actors": ["명령문에서 행동 주체로 직접 지정된 ally unitId들"],
          "target": "명령문에서 대상 역할을 하는 unitId string, null, 또는 multi-actor different-target 명령의 unitId string array",
          "target_side_in_text": "ally | enemy | self | none",
          "mentioned_units": ["명령문에 직접 등장한 모든 unitId들"]
        }
      },
      "metadata": {
        "intent_family": "taxonomy에 존재하는 intent_family 값",
        "command_style": "taxonomy에 존재하는 command_style 값",
        "actor_selection": "taxonomy에 존재하는 actor_selection 값",
        "target_selection": "taxonomy에 존재하는 target_selection 값",
        "action_pattern": "taxonomy에 존재하는 action_pattern 값",
        "scenario_family": "taxonomy에 존재하는 scenario_family 값",
        "edge_flags": ["taxonomy에 존재하는 edge_flags 값들"],
        "difficulty": "easy | medium | hard"
      },
      "skill_case": null 또는 {
        "skill_family": "taxonomy에 존재하는 skill_family 값",
        "skill_target_kind": "taxonomy에 존재하는 skill_target_kind 값",
        "is_skill_aoe": true 또는 false,
        "can_skill_target_dead": true 또는 false,
        "conflict_type": "taxonomy에 존재하는 conflict_type 값 또는 null"
      },
      "gold": {
        "required_actors": ["정답 output에 반드시 포함되어야 하는 actor unitId들"],
        "allowed_actors": ["정답 output에 포함될 수 있는 actor unitId들"],
        "forbidden_actors": ["정답 output에 포함되면 안 되는 actor unitId들"],
        "required_action_types": ["정답 output에 반드시 포함되어야 하는 action type들"],
        "allowed_action_types": ["정답 output에 포함될 수 있는 action type들"],
        "forbidden_action_types": ["정답 output에 포함되면 안 되는 action type들"],
        "empty_action_allowed": true 또는 false,
        "expected_action_pattern": "metadata.action_pattern과 동일한 taxonomy action_pattern enum",
        "targets": {
          "required": ["정답 output에서 반드시 사용되어야 하는 target 또는 move.to unitId들"],
          "allowed": ["정답 output에서 사용할 수 있는 target 또는 move.to unitId들"],
          "forbidden": ["정답 output에서 사용하면 안 되는 target 또는 move.to unitId들"]
        }
      },
      "input": {
        "input": {
          "command": "command_spec.command_text와 정확히 같은 한국어 명령문",
          "area_situation": {
            "allies": [
              {
                "unitId": "A_01부터 A_06 중 하나",
                "isAlive": true 또는 false,
                "canBeTargeted": true 또는 false,
                "isRanged": true 또는 false,
                "hpRatio": "0 이상 1 이하 number",
                "attackRatioToAvg": "0보다 큰 number",
                "engagedByOpponentCount": "0 이상의 integer",
                "teamFormationRole": "frontline | midline | backline",
                "skillDescription": "이 ally가 가진 스킬 설명 문자열",
                "IsSkillOnSelf": true 또는 false,
                "IsSkillOnOtherAlly": true 또는 false,
                "isSkillAoe": true 또는 false,
                "canSkillTargetDead": true 또는 false,
                "closestTargetableOpponent": "살아있고 canBeTargeted=true인 enemy unitId 또는 null",
                "farthestTargetableOpponent": "살아있고 canBeTargeted=true인 enemy unitId 또는 null",
                "closestAliveAlly": "자기 자신을 제외한 살아있는 ally unitId 또는 null",
                "farthestAliveAlly": "자기 자신을 제외한 살아있는 ally unitId 또는 null"
              }
            ],
            "enemies": [
              {
                "unitId": "E_01부터 E_06 중 하나",
                "isAlive": true 또는 false,
                "canBeTargeted": true 또는 false,
                "isRanged": true 또는 false,
                "hpRatio": "0 이상 1 이하 number",
                "attackRatioToAvg": "0보다 큰 number",
                "engagedByOpponentCount": "0 이상의 integer",
                "teamFormationRole": "frontline | midline | backline"
              }
            ]
          }
        }
      },
      "output": {
        "thinking": "짧은 한국어 판단 요약",
        "dialog": [
          {
            "unitId": "action에 포함된 actor unitId",
            "text": "그 actor의 전체 sequence를 요약하는 짧은 한국어 한 문장"
          }
        ],
        "action": [
          {
            "unitId": "살아있는 ally actor unitId",
            "sequence": [
              {
                "type": "attack | move | skill | wait | skillControl",
                "target": "attack 또는 skill target unitId가 필요한 경우 사용",
                "to": "move.to unitId가 필요한 경우 사용",
                "description": "skill action일 때 actor.skillDescription과 정확히 같은 문자열",
                "subtype": "move action일 때 approachOpponent | escape | help | holdFront",
                "movementType": "move action일 때 direct | flank",
                "durationSec": "wait 또는 defer skillControl일 때 1 이상 10 이하 number",
                "mode": "skillControl일 때 defer | forbid"
              }
            ]
          }
        ]
      }
    }
  ]
}

위 예시는 구조 설명용 skeleton이다.
위 예시에 적힌 설명 문자열, placeholder 문자열, 예시 문구를 실제 출력값으로 복사하지 않는다.
실제 sample에서는 모든 placeholder를 selected_bucket, taxonomy, command, skill_case, gold, area_situation에 맞는 실제 값으로 채운다.
area_situation.allies에는 위에 ally를 하나만 적었지만, 실제 sample에서는 반드시 A_01부터 A_06까지 정확히 6명을 모두 적는다.
area_situation.enemies에는 위에 enemy를 하나만 적었지만, 실제 sample에서는 반드시 E_01부터 E_06까지 정확히 6명을 모두 적는다.
input.commandAnalysis는 절대 생성하지 않는다.
commandAnalysis는 validator가 accepted 저장 시점에 계산해서 추가한다.

예시는 schema shape와 생성 품질 기준만 보여준다.
예시의 command_text, unitId 배치, metadata 값, 전장 상황, output 의미를 그대로 복사하지 않는다.
실제 생성값은 반드시 mixed_generation_requests 각 item의 selected_bucket, existing_valid_paraphrase_samples, command_text_policy, count_to_generate를 따른다.
""".strip()


def get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수가 없습니다.\n"
            "PowerShell 예시:\n"
            '$env:GEMINI_API_KEY="네_API_키"'
        )
    return api_key


def get_chunk_text(chunk: object) -> str:
    text = getattr(chunk, "text", None)
    return text if isinstance(text, str) else ""


def sdk_object_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)

    if hasattr(value, "to_json_dict"):
        result = value.to_json_dict()
        return result if isinstance(result, dict) else None

    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }

    return None


def extract_token_usage(usage_metadata: Any) -> dict[str, Any]:
    data = sdk_object_to_dict(usage_metadata) or {}

    return {
        "prompt_token_count": data.get("prompt_token_count"),
        "cached_content_token_count": data.get("cached_content_token_count"),
        "candidates_token_count": data.get("candidates_token_count"),
        "thoughts_token_count": data.get("thoughts_token_count"),
        "total_token_count": data.get("total_token_count"),
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Input payload root must be a JSON object.")

    return data


# Gemini streaming 응답에서 chunk.text만 누적한다.
def collect_gemini_stream_text(stream: Any, stream_output: bool = False) -> str:
    chunks: list[str] = []

    for chunk in stream:
        piece = get_chunk_text(chunk)
        if not piece:
            continue

        chunks.append(piece)

        if stream_output:
            print(piece, end="", flush=True)

    if stream_output:
        print()

    return "".join(chunks)


# Gemini generate_content_stream을 호출하고 teacher raw text와 timing을 반환한다.
def call_gemini(
    user_payload: dict[str, Any],
    model_name: str,
    max_tokens: int,
    stream_output: bool = False,
) -> tuple[str, dict[str, Any]]:
    client = genai.Client(api_key=get_api_key())
    user_text = json.dumps(user_payload, ensure_ascii=False, indent=2)

    if stream_output:
        print("generating with stream output...", flush=True)
    else:
        print("generating...", flush=True)

    start_time = time.perf_counter()
    first_chunk_time: float | None = None
    chunks: list[str] = []
    usage_metadata: Any = None

    stream = client.models.generate_content_stream(
        model=model_name,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.15,
            top_p=1.0,
            candidate_count=1,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )

    for chunk in stream:
        metadata = getattr(chunk, "usage_metadata", None)
        if metadata is not None:
            usage_metadata = metadata

        piece = get_chunk_text(chunk)
        if not piece:
            continue

        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()

        chunks.append(piece)

        if stream_output:
            print(piece, end="", flush=True)

    end_time = time.perf_counter()

    if stream_output:
        print()

    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError("Teacher response is empty.")

    token_usage = extract_token_usage(usage_metadata)

    timing = {
        "ttft_sec": (
            None
            if first_chunk_time is None
            else round(first_chunk_time - start_time, 3)
        ),
        "total_response_time_sec": round(end_time - start_time, 3),
        "response_chars": len(text),
        "prompt_token_count": token_usage["prompt_token_count"],
        "cached_content_token_count": token_usage["cached_content_token_count"],
        "candidates_token_count": token_usage["candidates_token_count"],
        "thoughts_token_count": token_usage["thoughts_token_count"],
        "total_token_count": token_usage["total_token_count"],
    }

    print("generation complete", flush=True)
    print(
        "timing: "
        f"ttft_sec={timing['ttft_sec']}, "
        f"total_response_time_sec={timing['total_response_time_sec']}, "
        f"response_chars={timing['response_chars']}, "
        f"prompt_tokens={timing['prompt_token_count']}, "
        f"output_tokens={timing['candidates_token_count']}, "
        f"thoughts_tokens={timing['thoughts_token_count']}, "
        f"total_tokens={timing['total_token_count']}",
        flush=True,
    )

    return text, timing


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()

    return stripped


# 모델이 앞뒤에 불필요한 텍스트를 붙인 경우 첫 JSON value 경계를 찾아 복구한다.
def extract_first_json_value(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Teacher response is empty.")

    start_candidates = [
        index for index in (stripped.find("{"), stripped.find("[")) if index >= 0
    ]
    if not start_candidates:
        raise ValueError("Teacher response does not contain JSON.")

    start_index = min(start_candidates)
    opener = stripped[start_index]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for index in range(start_index, len(stripped)):
        char = stripped[index]

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

        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return stripped[start_index : index + 1]

    raise ValueError("Teacher response JSON boundary is invalid.")


def parse_teacher_json(raw_text: str) -> Any:
    text = strip_code_fence(raw_text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        extracted = extract_first_json_value(text)
        return json.loads(extracted)


# teacher가 object, array, {samples:[...]} 중 무엇을 내도 JSONL sample list로 정규화한다.
def normalize_samples(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        samples = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("samples"), list):
        samples = parsed["samples"]
    elif isinstance(parsed, dict):
        samples = [parsed]
    else:
        raise ValueError(
            "Teacher JSON root must be object, array, or object with samples."
        )

    normalized: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample #{index} is not an object.")
        normalized.append(sample)

    return normalized


def assign_sample_ids(samples: list[dict[str, Any]], output_path: Path) -> None:
    batch_id = output_path.stem
    if batch_id.endswith("_raw"):
        batch_id = batch_id.removesuffix("_raw")

    for index, sample in enumerate(samples, start=1):
        sample["id"] = f"3{batch_id}_{index:03d}"


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


# request payload를 읽고 teacher raw samples와 trace를 각각 JSONL에 append한다.
def run_teacher_generation(
    input_path: Path,
    output_path: Path,
    trace_path: Path,
    model_name: str = MODEL_NAME,
    max_tokens: int = 60000,
    print_json: bool = False,
    stream_output: bool = False,
) -> dict[str, Any]:
    payload = load_json(input_path)
    raw_response, timing = call_gemini(
        user_payload=payload,
        model_name=model_name,
        max_tokens=max_tokens,
        stream_output=stream_output,
    )

    parsed = parse_teacher_json(raw_response)
    samples = normalize_samples(parsed)
    validate_teacher_sample_count(samples, payload)
    force_request_base_command_text(samples, payload)
    assign_sample_ids(samples, output_path)
    append_jsonl(output_path, samples)

    trace_record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sample_count": len(samples),
        "ttft_sec": timing["ttft_sec"],
        "total_response_time_sec": timing["total_response_time_sec"],
        "response_chars": timing["response_chars"],
        "raw_response": raw_response,
        "prompt_token_count": timing["prompt_token_count"],
        "cached_content_token_count": timing["cached_content_token_count"],
        "candidates_token_count": timing["candidates_token_count"],
        "thoughts_token_count": timing["thoughts_token_count"],
        "total_token_count": timing["total_token_count"],
    }
    append_jsonl(trace_path, [trace_record])

    if print_json:
        print(json.dumps(samples, ensure_ascii=False, indent=2))

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "trace_path": str(trace_path),
        "sample_count": len(samples),
        "ttft_sec": timing["ttft_sec"],
        "total_response_time_sec": timing["total_response_time_sec"],
        "response_chars": timing["response_chars"],
        "prompt_token_count": timing["prompt_token_count"],
        "cached_content_token_count": timing["cached_content_token_count"],
        "candidates_token_count(actual tokens)": timing["candidates_token_count"],
        "thoughts_token_count": timing["thoughts_token_count"],
        "total_token_count": timing["total_token_count"],
    }

def expected_sample_count(payload: dict[str, Any]) -> int:
    value = payload.get("count_to_generate")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("payload.count_to_generate must be a non-negative integer")
    return value


def expected_item_counts(payload: dict[str, Any]) -> list[int]:
    items = payload.get("mixed_generation_requests")
    if not isinstance(items, list) or not items:
        raise ValueError("payload.mixed_generation_requests must be a non-empty list")

    counts: list[int] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"mixed_generation_requests[{index}] must be an object")

        count = item.get("count_to_generate")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"mixed_generation_requests[{index}].count_to_generate must be a non-negative integer"
            )

        counts.append(count)

    return counts


def validate_teacher_sample_count(
    samples: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    total_expected = expected_sample_count(payload)
    item_counts = expected_item_counts(payload)

    if sum(item_counts) != total_expected:
        raise ValueError(
            f"payload count mismatch: top-level={total_expected}, item_sum={sum(item_counts)}"
        )

    if len(samples) != total_expected:
        raise ValueError(
            f"teacher sample count mismatch: expected={total_expected}, actual={len(samples)}"
        )

# item별 request.base_command_text를 teacher 출력에 강제로 반영한다.
# 출력 순서는 mixed_output_contract.output_order와 동일해야 한다.
def force_request_base_command_text(
    samples: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    items = payload.get("mixed_generation_requests")
    if not isinstance(items, list) or not items:
        raise ValueError("payload.mixed_generation_requests must be a non-empty list")

    sample_index = 0

    for item_index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"mixed_generation_requests[{item_index}] must be an object")

        request = item.get("request")
        if not isinstance(request, dict):
            raise ValueError(
                f"mixed_generation_requests[{item_index}].request must be an object"
            )

        base_command_text = request.get("base_command_text")
        if not isinstance(base_command_text, str) or not base_command_text:
            raise ValueError(
                f"mixed_generation_requests[{item_index}].request.base_command_text is missing"
            )

        count_to_generate = item.get("count_to_generate")
        if (
            isinstance(count_to_generate, bool)
            or not isinstance(count_to_generate, int)
            or count_to_generate < 0
        ):
            raise ValueError(
                f"mixed_generation_requests[{item_index}].count_to_generate must be a non-negative integer"
            )

        item_samples = samples[sample_index : sample_index + count_to_generate]
        if len(item_samples) != count_to_generate:
            raise ValueError(
                f"sample slice mismatch for item {item_index}: expected={count_to_generate}, actual={len(item_samples)}"
            )

        sample_index += count_to_generate

        for sample in item_samples:
            command_spec = sample.setdefault("command_spec", {})
            if not isinstance(command_spec, dict):
                raise ValueError("sample.command_spec must be an object")
            command_spec["base_command_text"] = base_command_text

    if sample_index != len(samples):
        raise ValueError(
            f"unused samples after base_command_text assignment: assigned={sample_index}, actual={len(samples)}"
        )
            
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to request payload JSON produced by sft_cli.py request.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to append validator-ready raw generation JSONL.",
    )
    parser.add_argument(
        "--trace-output",
        required=True,
        help="Path to append raw teacher response trace JSONL.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Gemini API model name.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60000,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print parsed samples.",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Print teacher response chunks while generating.",
    )
    args = parser.parse_args()

    result = run_teacher_generation(
        input_path=Path(args.input),
        output_path=Path(args.output),
        trace_path=Path(args.trace_output),
        model_name=args.model,
        max_tokens=args.max_tokens,
        print_json=args.print_json,
        stream_output=args.stream_output,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
