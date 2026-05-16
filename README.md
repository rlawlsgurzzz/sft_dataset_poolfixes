# Synthetic SFT Dataset 설계 요약

## 1. 목적

이 레포는 전장 상태와 한국어 자연어 명령을 입력으로 받아, 학생 SLM이 전투 명령 결과를 안정적인 JSON으로 출력하도록 학습시키는 Synthetic SFT dataset을 관리한다.

```text
전장 상태 + 한국어 명령 + commandAnalysis
→ thinking / dialog / action JSON
```

학습 대상은 문장 치환이 아니라 전투 명령 파서 동작 전체다. 모델은 명령 의미, actor/target/action 판단, skill target side, 현재 전장 상태, 런타임 유효성, dialog/action actor 일치를 함께 학습한다.

품질 우선순위는 다음이다.

```text
raw output semantic correctness
> schema correctness
> runtime validity
> policy validity
> dialog 품질
> 표현 다양성
```

SFT label은 처음부터 유효한 assistant output이어야 하며, 보정된 결과를 label로 사용하지 않는다.

---

## 2. 데이터 생성과 검증 역할

### 2.1 teacher LLM

teacher LLM은 master sample 후보를 생성한다.

teacher가 생성하는 항목은 다음이다.

```text
id
split
command_spec
metadata
skill_case
gold
input.input.command
input.input.area_situation
output
```

teacher는 전장 상태를 완전하게 창작하고, 그 전장 상태와 명령에 맞는 정답 `thinking / dialog / action`을 만든다.

teacher는 다음 항목을 생성하지 않는다.

```text
input.commandAnalysis
commandAnalysis
allowedActors
allowedAttackTargets
validMoveToUnits
deadAllies
invalidUnits
actionPolicy
validator_result
source_ref
```

teacher는 generation payload에 포함된 다음 정보를 기준으로 샘플을 만든다.

```text
selected_bucket
target_split
existing_valid_paraphrase_samples
other_split_reserved_command_texts
command_text_policy
runtime_generation_contract
area_situation_contract
assistant_output_contract
generation_constraints
```

### 2.2 validator

validator는 teacher가 생성한 raw sample을 검증한다.

검증 범위는 다음이다.

```text
JSON parse
master sample schema
split validity
command_spec consistency
metadata taxonomy
skill_case taxonomy
area_situation schema
unit runtime validity
output schema
action runtime validity
skill target validity
dialog/action actor consistency
gold semantic consistency
```

검증을 통과한 sample은 `accepted`에 저장된다. 저장 시 validator가 `area_situation`을 읽어 `input.commandAnalysis`를 계산해 추가한다.

검증을 통과하지 못한 sample은 `rejected`에 저장된다.

### 2.3 student SLM

student SLM은 `accepted` sample에서 만들어진 최종 입력을 학습한다.

학생 모델의 입력은 다음 구조를 가진다.

```json
{
  "input": {
    "command": "한국어 명령",
    "area_situation": {
      "allies": [],
      "enemies": []
    }
  },
  "commandAnalysis": {
    "analysisMode": "runtime_constraint_summary",
    "allowedActors": [],
    "allowedAttackTargets": [],
    "validMoveToUnits": [],
    "deadAllies": [],
    "invalidUnits": [],
    "actionPolicy": {}
  }
}
```

학생 모델의 출력은 다음 세 top-level key로 고정된다.

```json
{
  "thinking": "짧은 판단 요약",
  "dialog": [],
  "action": []
}
```

학생 SLM은 `commandAnalysis`를 참고하지만, 명령의 actor/target/action 의미 해석은 직접 수행한다. `commandAnalysis`는 명령 의도 파서가 아니라 런타임 제약 요약이다.

---

## 3. raw sample 구조

teacher가 생성하는 raw sample은 다음 구조를 따른다.

```json
{
  "id": "sample id",
  "split": "train",
  "command_spec": {},
  "metadata": {},
  "skill_case": null,
  "gold": {},
  "input": {
    "input": {
      "command": "한국어 명령",
      "area_situation": {
        "allies": [],
        "enemies": []
      }
    }
  },
  "output": {
    "thinking": "짧은 판단 요약",
    "dialog": [],
    "action": []
  }
}
```

raw sample은 teacher LLM의 원본 생성물이며, 아직 학습 데이터가 아니다.

---

## 4. accepted sample 구조

accepted sample은 raw sample에 validator 산출물이 추가된 형태다.

```json
{
  "input": {
    "input": {
      "command": "한국어 명령",
      "area_situation": {
        "allies": [],
        "enemies": []
      }
    },
    "commandAnalysis": {
      "analysisMode": "runtime_constraint_summary",
      "allowedActors": [],
      "allowedAttackTargets": [],
      "validMoveToUnits": [],
      "deadAllies": [],
      "invalidUnits": [],
      "actionPolicy": {}
    }
  },
  "validator_result": {
    "passed": true,
    "failure_reasons": []
  }
}
```

`accepted` 폴더가 학습 데이터와 coverage 집계의 source of truth다. coverage report는 accepted 샘플을 기준으로 재생성된다.

---

## 5. 전장 상태 스키마

`area_situation`은 항상 두 배열을 가진다.

```json
{
  "allies": [],
  "enemies": []
}
```

### 5.1 allies

`allies`에는 정확히 6명의 아군이 들어간다.

```text
A_01
A_02
A_03
A_04
A_05
A_06
```

아군 유닛 필수 필드는 다음이다.

```json
{
  "unitId": "A_01",
  "isAlive": true,
  "canBeTargeted": true,
  "isRanged": false,
  "hpRatio": 0.78,
  "attackRatioToAvg": 1.08,
  "engagedByOpponentCount": 1,
  "teamFormationRole": "frontline",
  "skillDescription": "스킬 설명 문자열",
  "IsSkillOnSelf": false,
  "IsSkillOnOtherAlly": false,
  "isSkillAoe": false,
  "canSkillTargetDead": false,
  "closestTargetableOpponent": "E_02",
  "farthestTargetableOpponent": "E_06",
  "closestAliveAlly": "A_02",
  "farthestAliveAlly": "A_05"
}
```

### 5.2 enemies

`enemies`에는 정확히 6명의 적이 들어간다.

```text
E_01
E_02
E_03
E_04
E_05
E_06
```

적 유닛 필수 필드는 다음이다.

```json
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
```

### 5.3 단일 거리 신호 필드

아군 유닛에는 네 개의 단일 거리 신호 필드가 있다.

```text
closestTargetableOpponent
farthestTargetableOpponent
closestAliveAlly
farthestAliveAlly
```

규칙은 다음이다.

```text
- 네 필드는 배열이 아니다.
- 값은 unitId string 또는 null이다.
- unitId는 최대 하나만 들어간다.
- closestTargetableOpponent와 farthestTargetableOpponent에는 살아있고 canBeTargeted=true인 enemy만 들어간다.
- closestAliveAlly와 farthestAliveAlly에는 자기 자신을 제외한 살아있는 ally만 들어간다.
- 후보가 없으면 null이다.
- 후보가 1명뿐이면 closest와 farthest가 같은 unitId다.
- 후보가 2명 이상이면 closest와 farthest는 서로 다른 unitId다.
```

실제 거리 계산은 런타임 엔진에서 수행한다. Synthetic dataset에서는 teacher가 전술 상황에 맞는 논리적 거리 관계를 만든다.

---

## 6. 죽은 유닛 규칙

이 게임에서는 죽은 유닛도 보통 `canBeTargeted=true`일 수 있다.

죽은 유닛 규칙은 다음이다.

```text
- 죽은 ally는 actor가 될 수 없다.
- 죽은 enemy는 attack target이 될 수 없다.
- 죽은 unit은 move.to가 될 수 없다.
- 죽은 unit은 단일 거리 신호 필드에 들어갈 수 없다.
- 죽었지만 canBeTargeted=true인 ally는 validator가 commandAnalysis.deadAllies에 포함한다.
- canSkillTargetDead=true인 skill만 죽은 targetable ally를 skill target으로 사용할 수 있다.
```

`canSkillTargetDead=true`는 skill target에서만 적용되는 특수 예외다. 일반 attack, move, actor 선정에는 적용되지 않는다.

---

## 7. commandAnalysis

`commandAnalysis`는 명령 의도 파서가 아니다. 역할은 현재 전장 상태에서 런타임상 가능한 범위를 요약하는 것이다.

```json
{
  "analysisMode": "runtime_constraint_summary",
  "allowedActors": [],
  "allowedAttackTargets": [],
  "validMoveToUnits": [],
  "deadAllies": [],
  "invalidUnits": [],
  "actionPolicy": {}
}
```

계산 기준은 다음이다.

```text
allowedActors:
  살아있는 ally

allowedAttackTargets:
  살아있고 canBeTargeted=true인 enemy

validMoveToUnits:
  살아있는 ally + 살아있고 canBeTargeted=true인 enemy

deadAllies:
  죽었지만 canBeTargeted=true인 ally

invalidUnits:
  죽은 ally, 죽은 enemy, untargetable enemy 등

actionPolicy:
  action 관련 고정 정책
```

예를 들어 다음 해석은 SLM의 책임이다.

```text
"손이 비는 아군" → 현재 압박이 낮고 행동 가능한 아군
"체력이 낮은 아군" → hpRatio가 낮은 아군
"제일 위험한 적" → attackRatioToAvg나 현재 위협 상황이 높은 적
"아군 뒤쪽으로 빠져" → safe ally 또는 후열 방향으로 escape
```

---

## 8. 출력 action 스키마

허용 action type은 다음이다.

```text
move
attack
skill
wait
skillControl
```

### 8.1 move

```json
{
  "type": "move",
  "subtype": "approachOpponent",
  "movementType": "direct",
  "to": "E_01"
}
```

규칙은 다음이다.

```text
subtype:
  approachOpponent | escape | help | holdFront

movementType:
  direct | flank

to:
  validMoveToUnits 안의 unitId

to에는 actor 자신의 unitId를 쓰지 않는다.
```

move는 좌표를 출력하지 않는다. 엔진이 `subtype`, `movementType`, `to`를 바탕으로 실제 이동 위치를 계산한다.

### 8.2 attack

```json
{
  "type": "attack",
  "target": "E_01"
}
```

`target`은 `allowedAttackTargets` 안의 enemy unitId다. 죽은 enemy나 untargetable enemy는 attack target이 될 수 없다.

### 8.3 skill

```json
{
  "type": "skill",
  "description": "actor의 정확한 skillDescription 문자열",
  "target": "unitId"
}
```

규칙은 다음이다.

```text
description은 actor.skillDescription과 정확히 같아야 한다.
IsSkillOnSelf=true이면 target은 actor 자신이다.
IsSkillOnOtherAlly=true이면 target은 actor 자신이 아닌 ally다.
IsSkillOnSelf=false이고 IsSkillOnOtherAlly=false이면 target은 enemy다.
canSkillTargetDead=false이면 죽은 unit을 target으로 쓰지 않는다.
canSkillTargetDead=true이면 죽은 targetable ally를 target으로 쓸 수 있다.
isSkillAoe=true여도 output target은 중심 unitId 하나만 쓴다.
```

skill target은 명령문에 적힌 대상만으로 결정하지 않는다. actor의 skill field와 현재 전장 상태를 함께 기준으로 결정한다.

skill action이 하나라도 있으면 `skill_case`는 반드시 object여야 하며 `null`이면 validator가 실패 처리한다.

### 8.4 wait

```json
{
  "type": "wait",
  "durationSec": 2
}
```

규칙은 다음이다.

```text
wait은 명령에 대기, 지연, 타이밍 조절, 위치 유지 의미가 직접 있을 때만 사용한다.
durationSec는 1 이상 10 이하 number다.
attack 또는 skill 뒤에는 wait을 붙이지 않는다.
```

### 8.5 skillControl

```json
{
  "type": "skillControl",
  "mode": "defer",
  "durationSec": 5
}
```

또는:

```json
{
  "type": "skillControl",
  "mode": "forbid"
}
```

규칙은 다음이다.

```text
skillControl은 스킬 지연 또는 금지 의도가 명령에 명시될 때만 사용한다.
mode=defer이면 durationSec는 1 이상 10 이하 number다.
mode=forbid이면 durationSec를 쓰지 않는다.
```

---

## 9. dialog 규칙

```text
- dialog는 action actor당 정확히 하나만 만든다.
- action에 없는 unitId는 dialog에 넣지 않는다.
- 같은 unitId의 dialog를 여러 개 만들지 않는다.
- 여러 actor에게 완전히 같은 text를 반복하지 않는다.
- text는 actor의 전체 sequence를 짧은 한국어 한 문장으로 요약한다.
```

schema 안정성과 actor 일치가 표현 다양성보다 우선이다.

---

## 10. command_spec

`command_spec`은 명령문과 명령문 내부 slot을 설명한다.

```json
{
  "command_text": "A_02, A_04에게 이동해",
  "base_command_text": "A_01, A_02에게 이동해",
  "slots": {
    "actors": ["A_02"],
    "target": "A_04",
    "target_side_in_text": "ally",
    "mentioned_units": ["A_02", "A_04"]
  }
}
```

규칙은 다음이다.

```text
command_spec.command_text는 input.input.command와 정확히 같아야 한다.
base_command_text는 command slot의 기준 문장이다.
actors는 명령문에서 행동 주체로 지정된 ally unitId 목록이다.
target은 명령문에서 대상 역할을 하는 unitId 또는 null이다.
target_side_in_text는 ally | enemy | self | none 중 하나다.
mentioned_units는 명령문에 직접 등장한 모든 unitId 목록이다.
```

한국어 명령은 조사, 동사, 문맥으로 actor와 target을 구분한다.

```text
"A_02, A_04에게 이동해"
→ A_02 actor, A_04 target

"A_04, A_02한테 스킬 써"
→ A_04 actor, A_02 target

"A_01, A_02는 뒤로 빠져"
→ A_01과 A_02 모두 actor
```

---

## 11. metadata, taxonomy, skill_case

`config/taxonomy_sot.json`이 분류 기준의 source of truth다.

`metadata`는 다음 필드를 가진다.

```json
{
  "intent_family": "skill",
  "command_style": "direct_korean",
  "actor_selection": "explicit_actor",
  "target_selection": "explicit_enemy_target",
  "action_pattern": "skill_only",
  "scenario_family": "skill_self_target_conflict",
  "edge_flags": []
}
```

모든 값은 taxonomy 안에 존재해야 한다.

일반 명령 coverage는 다음 축을 기준으로 관리한다.

```text
intent_family
actor_selection
target_selection
action_pattern
scenario_family
edge_flags
```

`skill_case`는 skill 관련 sample에서만 object이며, skill과 무관한 sample에서는 `null`이다.

```json
{
  "skill_family": "self_buff",
  "skill_target_kind": "self",
  "is_skill_aoe": false,
  "can_skill_target_dead": false,
  "conflict_type": "text_enemy_target_but_self_skill"
}
```

skill 관련 주요 기준은 다음과 같다.

```text
skill_family
skill_target_kind
conflict_type
is_skill_aoe
can_skill_target_dead
```

skill intent request는 명시적인 skill path override가 있거나, 선택된 command slot의 accepted sample에서 단일 `skill_case`를 추론할 수 있어야 한다.

---

## 12. gold

`gold`는 validator가 output의 의미 일치를 검사하기 위한 정답 기준이다.

주요 필드는 다음이다.

```json
{
  "required_actors": [],
  "allowed_actors": [],
  "forbidden_actors": [],
  "required_action_types": [],
  "allowed_action_types": [],
  "forbidden_action_types": [],
  "empty_action_allowed": false,
  "expected_action_pattern": "attack_only",
  "targets": {
    "required": [],
    "allowed": [],
    "forbidden": []
  }
}
```

`gold.expected_action_pattern`은 반드시 taxonomy의 `action_pattern` enum 값이어야 하며, `metadata.action_pattern`과 정확히 같아야 한다. validator는 값이 문자열이 아니거나 taxonomy에 없거나 metadata와 다르면 실패 처리한다.

필요한 경우 action별 세부 기준도 포함할 수 있다.

```json
{
  "attack": {
    "actor": "A_01",
    "required_target": "E_02"
  },
  "skill": {
    "actor": "A_04",
    "required_target": "A_02",
    "description_exact": "죽은 아군 하나를 전투 가능한 상태로 되살린다."
  },
  "move": {
    "actor": "A_02",
    "required_subtype": "help",
    "required_to": "A_04"
  },
  "skillControl": {
    "actor": "A_03",
    "required_mode": "defer"
  }
}
```

학생 모델의 최종 출력에 `gold`는 포함되지 않는다.

---

## 13. split과 command_text 표현 pool / cycle

데이터셋 split은 다음 세 가지다.

```text
train
validation
test
```

각 split은 서로 다른 command_text 표현 pool을 가진다.

```text
train: 8개
validation: 4개
test: 4개
```

같은 command slot에서 teacher LLM은 다음 규칙을 따른다.

```text
1. 요청 split의 accepted 표현은 existing_valid_paraphrase_samples로 전달된다.
2. 다른 split의 accepted 표현은 other_split_reserved_command_texts로 전달된다.
3. 요청 split의 표현 pool이 가득 차기 전에는 새 command_text를 만든다.
4. 새 command_text는 같은 split의 기존 표현과 exact duplicate이면 안 된다.
5. 새 command_text는 다른 split의 reserved 표현과 exact duplicate이면 안 된다.
6. 출력 array의 앞쪽 new_unique_command_texts_to_create개 sample은 반드시 새 unique command_text로 만든다.
7. 새 unique command_text 생성이 모두 끝난 뒤에만 cycle sample을 만든다.
8. cycle source pool은 existing_valid_paraphrase_samples 뒤에 이번 응답에서 새로 만든 unique command_text를 output 순서대로 이어 붙인 목록이다.
9. cycle sample은 command_text_policy.sequence_contract.cycle_reuse_plan_1_based를 따라 source command_text를 재사용한다.
10. cycle 구간에서 첫 번째 source만 반복하거나 같은 command_text를 연속 반복하지 않는다.
11. 다른 split의 reserved command_text는 cycle source로 사용할 수 없다.
12. command_text를 재사용하더라도 area_situation, gold, output은 새로 구성한다.
```

`command_text_policy.sequence_contract`는 output 순서를 `new_unique_first_then_cycle`로 고정하고, cycle 구간의 source index를 1-based round-robin plan으로 명시한다.

요청 split에 기존 표현이 없으면 teacher는 해당 split의 pool limit까지 새 command_text를 만든다. 요청 개수가 pool limit을 초과하면 같은 요청에서 새로 만든 command_text까지 포함하여 같은 split pool 안에서 cycle한다.

---

## 14. generation automation

자동화 plan은 `generation_automation/auto_generation_plan_0001.txt` 같은 파일에 작성한다.

plan line 형식은 다음이다.

```text
<split> <generation_request>
```

규칙은 다음이다.

```text
- split은 train, validation, test 중 하나다.
- generation_request는 sft_cli.py request/generate에서 쓰는 numeric request와 stable request를 모두 허용한다.
- skill path override도 허용한다.
- count가 max-per-request보다 크면 같은 request prefix 안에서 여러 batch task로 자동 분할된다.
- 기본 max-per-request는 10이다.
```

예시는 다음이다.

```text
train c1-2-1-3-1-10/2-3-1.7
validation c1-2-1-3-1-10/2-3-1.3
test c1-2-1-3-1-10.2
train skill.explicit_actor.explicit_ally_target.skill_only.ally_skill_valid_target#1/ally_heal.ally_alive.null.6
```

자동 생성 흐름은 다음이다.

```text
1. plan 파일을 읽는다.
2. automation plan validator로 preflight 검증을 실행한다.
3. plan line을 max-per-request 단위의 batch task로 확장한다.
4. 각 task마다 request payload, raw generation, trace 파일을 순번 파일명으로 만든다.
5. teacher raw output 생성 직후 validator를 실행한다.
6. accepted/rejected 결과와 파일 경로를 run document에 누적 기록한다.
7. 필요하면 coverage report를 갱신한다.
```

자동화 결과 문서는 `generation_automation/auto_run_<timestamp>_plan_<NNNN>.md`에 저장된다. 오류 상세는 같은 timestamp의 `_errors.jsonl`에 저장된다.

---

## 15. 주요 파일

### 15.1 `config/taxonomy_sot.json`

```text
분류 기준
목표 비율
general valid matrix
skill valid matrix
edge_flags enum
taxonomy 설명
숫자 path 변환 기준
```

### 15.2 `scripts/sft_taxonomy.py`

```text
taxonomy_sot.json 로드
숫자 path 파싱
stable path 변환
general path 검증
skill override path 검증
selected_bucket 설명 추출
metadata와 skill_case taxonomy 검증
```

### 15.3 `scripts/sft_coverage_report.py`

```text
accepted/*.jsonl을 읽어 coverage를 집계하고 reports/*.md를 재생성한다.
```

생성되는 report는 다음이다.

```text
reports/taxonomy_sot.md
reports/general_coverage.md
reports/skill_coverage.md
reports/coverage_summary.md
```

### 15.4 `scripts/sft_generation_request.py`

```text
숫자/stable path 요청을 파싱한다.
accepted sample에서 요청 command slot을 찾는다.
요청 split의 cycle 가능 표현 pool을 구성한다.
다른 split의 duplicate 금지 표현 목록을 구성한다.
skill intent에서 skill path를 명시 또는 추론한다.
teacher LLM 요청 payload를 만든다.
```

### 15.5 `scripts/sft_teacher_client.py`

```text
teacher LLM을 호출한다.
raw generation JSONL을 저장한다.
trace JSONL을 저장한다.
stream output을 터미널에 표시할 수 있다.
```

### 15.6 `scripts/sft_validator.py`

```text
teacher raw sample을 검증한다.
accepted 또는 rejected로 분류한다.
accepted 저장 시 commandAnalysis를 계산해 추가한다.
gold.expected_action_pattern과 metadata.action_pattern의 일치를 검증한다.
skill action이 있으면 skill_case object가 존재하는지 검증한다.
```

### 15.7 `scripts/sft_cli.py`

```text
report 생성
generation request payload 생성
teacher generate 실행
validator 실행
refresh-report 실행
```

### 15.8 `scripts/sft_auto_generate.py`

```text
automation plan을 batch task로 확장한다.
plan preflight 검증 후 순차 생성/검증을 실행한다.
각 task 결과를 run document에 누적 기록한다.
--dry-run으로 payload build 가능 여부만 점검할 수 있다.
```

### 15.9 `scripts/sft_validate_automation_plans.py`

```text
generation_automation/auto_generation_plan_*.txt를 검증한다.
--plan으로 특정 plan number 또는 파일명만 검증할 수 있다.
invalid 항목은 numeric/stable request 형식과 실패 이유를 함께 출력한다.
```

---

## 16. 주요 명령어

### 16.1 자동화 plan dry-run

```powershell
py -3.11 scripts/sft_auto_generate.py --plan generation_automation/auto_generation_plan_0001.txt --dry-run
```

### 16.2 자동화 plan 실행

```powershell
py -3.11 scripts/sft_auto_generate.py --plan generation_automation/auto_generation_plan_0001.txt --yes --stream-output --refresh-report
```

### 16.3 전체 자동화 plan 검증

```powershell
py -3.11 scripts/sft_validate_automation_plans.py
```

### 16.4 특정 자동화 plan 검증

```powershell
py -3.11 scripts/sft_validate_automation_plans.py --plan 0001
```

### 16.5 coverage report 재생성

```powershell
py -3.11 scripts/sft_cli.py report
```

### 16.6 생성 요청 payload 만들기

```powershell
py -3.11 scripts/sft_cli.py request c1-1-1-1-1-1.10 --split train --print-json
```

```powershell
py -3.11 scripts/sft_cli.py request c1-1-1-1-1-1.4 --split validation --print-json
```

```powershell
py -3.11 scripts/sft_cli.py request c1-1-1-1-1-1.4 --split test --print-json
```

### 16.7 teacher generate 실행

```powershell
py -3.11 scripts/sft_cli.py generate c1-1-1-1-1-1.10 --split train --stream-output
```

```powershell
py -3.11 scripts/sft_cli.py generate c1-1-1-1-1-1.4 --split validation --stream-output
```

```powershell
py -3.11 scripts/sft_cli.py generate c1-1-1-1-1-1.4 --split test --stream-output
```

### 16.8 skill override 포함 generate

```powershell
py -3.11 scripts/sft_cli.py generate c3-1-1-4-4-1/2-3-2.4 --split validation --stream-output --print-json
```

### 16.9 teacher raw output 검증

```powershell
py -3.11 scripts/sft_cli.py validate --input raw_generations/batch_0001_raw.jsonl --refresh-report
```

### 16.10 teacher raw output dry-run 검증

```powershell
py -3.11 scripts/sft_cli.py validate --input raw_generations/batch_0001_raw.jsonl --dry-run
```

### 16.11 jsonl pretty

```powershell
python scripts/jsonl_pretty.py accepted_20260512_005252
python scripts/jsonl_pretty.py accepted/accepted_20260512_005252.jsonl
python scripts/jsonl_pretty.py seed_master_0001
```

---

## 17. 전체 루프

```text
1. taxonomy_sot.json으로 분류 기준과 valid matrix를 정의한다.
2. accepted 폴더를 기준으로 coverage report를 생성한다.
3. 부족한 command slot을 숫자 또는 stable path로 선택한다.
4. 단일 요청은 sft_cli.py request/generate로 처리한다.
5. 묶음 요청은 generation_automation의 plan 파일로 처리한다.
6. sft_generation_request.py가 teacher payload를 만든다.
7. payload에는 selected bucket, split pool, reserved command_text, sequence contract, generation contract가 포함된다.
8. sft_teacher_client.py가 teacher LLM을 호출해 raw sample 후보와 trace를 저장한다.
9. sft_validator.py가 raw sample을 검증한다.
10. 통과 sample은 commandAnalysis가 추가되어 accepted에 저장된다.
11. 실패 sample은 rejected에 저장된다.
12. coverage report를 다시 생성한다.
13. accepted sample을 SFT 학습 데이터로 변환한다.
```

---

## 18. 설계 원칙

```text
- accepted 폴더가 학습 데이터와 coverage의 source of truth다.
- reports/*.md는 accepted 기준 derived report다.
- taxonomy_sot.json이 분류 기준의 source of truth다.
- 숫자 path와 stable path는 generation request 입력 형식이다.
- 내부 처리는 stable path 기준이다.
- teacher raw output에는 commandAnalysis가 없다.
- commandAnalysis는 validator 산출물이다.
- selected_bucket이 의미/상황/edge/skill_case 조건을 가진다.
- existing_valid_paraphrase_samples는 요청 split의 cycle 가능 표현 pool이다.
- other_split_reserved_command_texts는 다른 split의 duplicate 금지 표현 목록이다.
- 같은 split 표현 pool 안에서만 command_text cycle을 허용한다.
- cycle은 sequence_contract.cycle_reuse_plan_1_based를 따른다.
- 다른 split 표현은 cycle source로 쓰지 않는다.
- command_text를 cycle로 재사용하더라도 area_situation, gold, output은 새로 만든다.
- source_ref는 teacher 입력과 output에 넣지 않는다.
- validator 통과 실패 sample은 학습 데이터에 넣지 않는다.
- skill action이 있으면 skill_case는 반드시 object다.
- gold.expected_action_pattern은 metadata.action_pattern과 정확히 같아야 한다.
```