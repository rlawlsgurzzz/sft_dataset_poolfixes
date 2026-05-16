# Taxonomy SOT

Coverage taxonomy SOT for battle-command Synthetic SFT generation. Runtime input/output shape follows the current battle command parser: input + commandAnalysis -> thinking/dialog/action.

## Global Targets

### intent_family

| key | target | description |
|---|---:|---|
| attack | 24% | 기본 공격 또는 공격 중심 명령이다. |
| move | 22% | 위치 이동, 후퇴, 접근, 지원, 전열 유지 명령이다. |
| skill | 26% | skill action을 사용하는 명령이다. |
| skillControl | 8% | 스킬 사용을 지연하거나 금지하는 명령이다. |
| wait | 8% | 대기, 지연, 타이밍 조절 명령이다. |
| empty | 12% | 현재 전장 상태에서 실행 가능한 action이 없어 empty output이 기대되는 명령이다. |

### actor_selection

| key | target | description |
|---|---:|---|
| explicit_actor | 40% | 명령에 단일 아군 actor가 직접 지목된다. |
| explicit_multi_actor | 12% | 명령에 복수 아군 actor가 직접 지목된다. |
| global_condition | 16% | 체력, 압박, 여유 여부 같은 조건으로 actor를 고른다. |
| global_role_based | 12% | 전열, 후열, 원거리, 근접 같은 역할로 actor를 고른다. |
| global_state_based | 15% | 현재 전장 상태, 교전 수, 안전성, 지원 가능성을 근거로 actor를 고른다. |
| no_valid_actor | 5% | 명령 의미에 맞는 유효 actor가 없거나 지목 actor가 행동 불가능하다. |

### target_selection

| key | target | description |
|---|---:|---|
| explicit_enemy_target | 21% | 명령에 적 unitId가 직접 지목된다. |
| explicit_ally_target | 13% | 명령에 아군 unitId가 직접 지목된다. |
| nearest_enemy | 8% | 가장 가까운 적을 target으로 고른다. |
| lowest_hp_enemy | 8% | 체력이 가장 낮은 적을 target으로 고른다. |
| highest_threat_enemy | 7% | 공격력이 높거나 현재 위협이 큰 적을 target으로 고른다. |
| role_based_enemy | 8% | 원거리 적, 후열 적, 근접 적 등 역할 기반으로 적을 고른다. |
| pressure_source_enemy | 6% | 특정 아군을 압박하거나 공격 중인 적을 고른다. |
| safe_ally | 6% | 후방 또는 안전한 아군을 이동 기준으로 고른다. |
| low_hp_ally | 6% | 체력이 낮은 아군을 target 또는 지원 대상으로 고른다. |
| backline_ally | 4% | teamFormationRole이 backline인 아군을 고른다. |
| invalid_explicit_target | 7% | 명시 target이 죽었거나 untargetable이거나 action/skill target 규칙과 충돌한다. |
| none | 6% | target이 필요 없거나 명시되지 않은 wait, skillControl, holdFront류 명령이다. |

### action_pattern

| key | target | description |
|---|---:|---|
| attack_only | 13% | 정답 output이 attack action만 가진다. |
| move_only | 13% | 정답 output이 move action만 가진다. |
| move_then_attack | 11% | 정답 output이 move 후 attack sequence를 가진다. |
| skill_only | 19% | 정답 output이 skill action만 가진다. |
| move_then_skill | 4% | 정답 output이 move 후 skill sequence를 가진다. |
| wait_only | 6% | 정답 output이 wait action만 가진다. |
| wait_then_attack | 3% | 정답 output이 wait 후 attack sequence를 가진다. |
| wait_then_skill | 2% | 정답 output이 wait 후 skill sequence를 가진다. |
| skillControl_defer | 4% | 정답 output이 skillControl defer action을 가진다. |
| skillControl_forbid | 4% | 정답 output이 skillControl forbid action을 가진다. |
| multi_actor_same_target | 6% | 여러 actor가 같은 target 또는 같은 전술 목적을 공유한다. |
| multi_actor_different_targets | 3% | 여러 actor가 서로 다른 target 또는 역할을 수행한다. |
| empty_action_expected | 12% | 정답 output의 dialog/action이 비어야 한다. |

### command_style

| key | target | description |
|---|---:|---|
| direct_korean | 35% | 명확하고 직설적인 표준 한국어 명령이다. |
| casual_korean | 20% | 자연스러운 구어체 한국어 명령이다. |
| elliptical_korean | 20% | 조사, 주어, 목적어가 일부 생략된 한국어 명령이다. |
| tactical_korean | 15% | 전술적 목적과 역할이 비교적 명확히 표현된 한국어 명령이다. |
| rough_korean | 10% | 거친 반말이나 게임식 표현이 포함된 한국어 명령이다. |

## General Valid Matrix

### attack

- allowed_actor_selection
  - explicit_actor: 35%
  - explicit_multi_actor: 15%
  - global_condition: 20%
  - global_role_based: 15%
  - global_state_based: 15%
- allowed_target_selection
  - explicit_enemy_target: 30%
  - nearest_enemy: 15%
  - lowest_hp_enemy: 15%
  - highest_threat_enemy: 12%
  - role_based_enemy: 12%
  - pressure_source_enemy: 10%
  - invalid_explicit_target: 6%
- allowed_action_pattern
  - attack_only: 45%
  - move_then_attack: 25%
  - multi_actor_same_target: 18%
  - multi_actor_different_targets: 7%
  - empty_action_expected: 5%
- allowed_scenario_family
  - simple_clear_target: 18%
  - multiple_valid_targets: 10%
  - nearest_target_clear: 8%
  - lowest_hp_target_clear: 8%
  - highest_threat_target_clear: 7%
  - role_based_target_clear: 8%
  - pressure_source_target_clear: 8%
  - focus_fire_clear: 8%
  - flank_attack_requested: 6%
  - dead_named_target: 6%
  - untargetable_named_target: 5%
  - selected_actor_dead: 4%
  - no_valid_target: 4%

### move

- allowed_actor_selection
  - explicit_actor: 45%
  - explicit_multi_actor: 10%
  - global_condition: 15%
  - global_role_based: 15%
  - global_state_based: 10%
  - no_valid_actor: 5%
- allowed_target_selection
  - explicit_ally_target: 22%
  - explicit_enemy_target: 18%
  - safe_ally: 16%
  - low_hp_ally: 12%
  - backline_ally: 10%
  - role_based_enemy: 5%
  - invalid_explicit_target: 7%
  - none: 10%
- allowed_action_pattern
  - move_only: 55%
  - move_then_attack: 20%
  - move_then_skill: 8%
  - multi_actor_same_target: 5%
  - multi_actor_different_targets: 2%
  - empty_action_expected: 10%
- allowed_scenario_family
  - move_to_alive_ally: 10%
  - move_to_dead_ally: 5%
  - approach_enemy_only: 10%
  - approach_enemy_then_attack: 10%
  - flank_enemy_then_attack: 8%
  - retreat_to_backline_ally: 12%
  - low_hp_actor_escape: 10%
  - help_ally: 10%
  - support_low_hp_ally: 8%
  - hold_front: 7%
  - move_to_self_attempt: 5%
  - no_matching_actor: 5%

### skill

- allowed_actor_selection
  - explicit_actor: 55%
  - explicit_multi_actor: 10%
  - global_condition: 10%
  - global_role_based: 10%
  - global_state_based: 10%
  - no_valid_actor: 5%
- allowed_target_selection
  - explicit_enemy_target: 26%
  - explicit_ally_target: 22%
  - lowest_hp_enemy: 8%
  - nearest_enemy: 5%
  - role_based_enemy: 8%
  - low_hp_ally: 10%
  - invalid_explicit_target: 13%
  - none: 8%
- allowed_action_pattern
  - skill_only: 70%
  - move_then_skill: 10%
  - multi_actor_same_target: 7%
  - multi_actor_different_targets: 3%
  - empty_action_expected: 10%
- allowed_scenario_family
  - enemy_skill_valid_target: 10%
  - ally_skill_valid_target: 10%
  - self_skill_no_target: 8%
  - self_skill_enemy_target_conflict: 8%
  - ally_skill_enemy_target_conflict: 8%
  - enemy_skill_ally_target_conflict: 8%
  - resurrection_dead_ally_valid: 8%
  - resurrection_living_ally_conflict: 6%
  - dead_target_forbidden: 6%
  - aoe_skill_center_selection: 8%
  - actor_has_no_skill: 8%
  - approach_then_skill: 4%
  - no_valid_skill_actor: 4%
  - no_valid_skill_target: 4%

### skillControl

- allowed_actor_selection
  - explicit_actor: 70%
  - explicit_multi_actor: 20%
  - no_valid_actor: 10%
- allowed_target_selection
  - none: 100%
- allowed_action_pattern
  - skillControl_defer: 55%
  - skillControl_forbid: 35%
  - empty_action_expected: 10%
- allowed_scenario_family
  - explicit_defer_skill: 25%
  - defer_without_duration: 15%
  - explicit_forbid_skill: 25%
  - forbid_without_duration: 10%
  - multi_actor_defer_skill: 10%
  - multi_actor_forbid_skill: 5%
  - actor_has_no_skill: 5%
  - selected_actor_dead: 5%

### wait

- allowed_actor_selection
  - explicit_actor: 60%
  - explicit_multi_actor: 15%
  - global_condition: 15%
  - no_valid_actor: 10%
- allowed_target_selection
  - explicit_enemy_target: 10%
  - none: 90%
- allowed_action_pattern
  - wait_only: 65%
  - wait_then_attack: 20%
  - wait_then_skill: 10%
  - empty_action_expected: 5%
- allowed_scenario_family
  - explicit_wait: 25%
  - explicit_wait_duration: 20%
  - wait_then_attack_valid: 15%
  - wait_then_skill_valid: 10%
  - hold_position_wait: 10%
  - multi_actor_wait: 8%
  - no_matching_wait_actor: 6%
  - selected_actor_dead: 6%

### empty

- allowed_actor_selection
  - explicit_actor: 30%
  - explicit_multi_actor: 10%
  - global_condition: 25%
  - global_role_based: 15%
  - global_state_based: 10%
  - no_valid_actor: 10%
- allowed_target_selection
  - explicit_enemy_target: 20%
  - explicit_ally_target: 15%
  - invalid_explicit_target: 30%
  - low_hp_ally: 5%
  - role_based_enemy: 5%
  - lowest_hp_enemy: 5%
  - none: 20%
- allowed_action_pattern
  - empty_action_expected: 100%
- allowed_scenario_family
  - named_actor_dead: 12%
  - all_named_actors_dead: 6%
  - named_target_dead: 10%
  - named_target_untargetable: 10%
  - actor_outside_allowedActors: 8%
  - attack_target_outside_allowedTargets: 8%
  - move_to_self_attempt: 6%
  - skill_target_dead_not_allowed: 8%
  - skill_actor_has_no_skill: 8%
  - no_matching_actor: 10%
  - no_matching_role_actor: 6%
  - no_valid_target: 8%

## Skill Valid Matrix

### self_buff

- allowed_skill_target_kind
  - self: 100%
- allowed_conflict_type
  - null: 45%
  - text_enemy_target_but_self_skill: 35%
  - text_ally_target_but_self_skill: 20%

### ally_shield

- allowed_skill_target_kind
  - ally_alive: 85%
  - ally_dead: 15%
- allowed_conflict_type
  - null: 50%
  - text_enemy_target_but_ally_skill: 30%
  - text_dead_target_but_skill_cannot_target_dead: 20%

### ally_heal

- allowed_skill_target_kind
  - ally_alive: 85%
  - ally_dead: 15%
- allowed_conflict_type
  - null: 55%
  - text_enemy_target_but_ally_skill: 25%
  - text_dead_target_but_skill_cannot_target_dead: 20%

### ally_resurrection

- allowed_skill_target_kind
  - ally_dead: 70%
  - ally_alive: 30%
- allowed_conflict_type
  - null: 65%
  - text_living_target_but_resurrection_skill: 35%

### enemy_single_target_attack

- allowed_skill_target_kind
  - enemy_alive: 80%
  - enemy_dead: 20%
- allowed_conflict_type
  - null: 55%
  - text_ally_target_but_enemy_skill: 25%
  - text_dead_target_but_skill_cannot_target_dead: 20%

### enemy_debuff

- allowed_skill_target_kind
  - enemy_alive: 85%
  - enemy_dead: 15%
- allowed_conflict_type
  - null: 60%
  - text_ally_target_but_enemy_skill: 25%
  - text_dead_target_but_skill_cannot_target_dead: 15%

### enemy_aoe_attack

- allowed_skill_target_kind
  - enemy_alive: 100%
- allowed_conflict_type
  - null: 75%
  - text_ally_target_but_enemy_skill: 25%
- required_edge_flags
  - aoe_skill_requires_single_center_target

### mobility_skill

- allowed_skill_target_kind
  - self: 50%
  - enemy_alive: 50%
- allowed_conflict_type
  - null: 100%

### no_skill

- allowed_skill_target_kind
  - none: 100%
- allowed_conflict_type
  - skill_actor_has_no_skill: 100%

## Edge Flags

| edge_flag | description |
|---|---|
| named_actor_dead | 명시 actor가 죽어 있다. |
| all_named_actors_dead | 명시된 모든 actor가 죽어 있다. |
| actor_outside_allowedActors | actor가 allowedActors 밖이다. |
| no_matching_actor | 조건에 맞는 actor가 없다. |
| no_matching_role_actor | 역할 조건에 맞는 actor가 없다. |
| no_valid_actor | 유효 actor가 없다. |
| free_actor_selection | 손이 비거나 압박받지 않는 actor를 선택해야 한다. |
| low_hp_actor_selection | 체력이 낮은 actor를 선택해야 한다. |
| healthy_actor_selection | 체력이 여유 있는 actor를 선택해야 한다. |
| frontline_actor_selection | 전열 actor를 선택해야 한다. |
| actor_role_ranged | 원거리 actor를 선택해야 한다. |
| actor_role_melee | 근접 actor를 선택해야 한다. |
| named_target_dead | 명시 target이 죽어 있다. |
| named_target_untargetable | 명시 target이 untargetable이다. |
| attack_target_outside_allowedTargets | attack target이 allowedAttackTargets 밖이다. |
| no_valid_target | 유효 target이 없다. |
| low_hp_ally_target | 체력이 낮은 아군 target을 선택해야 한다. |
| target_role_ranged_enemy | 원거리 적 target을 선택해야 한다. |
| target_role_backline_enemy | 후열 적 target을 선택해야 한다. |
| pressure_source_target_clear | 아군을 압박하는 적을 target으로 선택해야 한다. |
| flank_requested | 우회, 측면, 후방 접근이 요구된다. |
| retreat_to_safe_ally | 안전한 아군 쪽으로 후퇴해야 한다. |
| hold_front_requested | 전열 유지가 요구된다. |
| move_to_self_attempt | move.to가 actor 본인이 될 위험이 있다. |
| help_ally_then_attack | 아군 지원 후 주변 적 공격이 필요하다. |
| explicit_enemy_target_conflicts_with_self_skill | 명시 적 target과 self skill 규칙이 충돌한다. |
| explicit_ally_target_conflicts_with_self_skill | 명시 아군 target과 self skill 규칙이 충돌한다. |
| explicit_enemy_target_conflicts_with_ally_skill | 명시 적 target과 ally skill 규칙이 충돌한다. |
| explicit_ally_target_conflicts_with_enemy_skill | 명시 아군 target과 enemy skill 규칙이 충돌한다. |
| skill_target_dead_not_allowed | 죽은 skill target이 허용되지 않는다. |
| dead_ally_skill_target_allowed | 죽은 아군 skill target이 허용된다. |
| text_living_target_but_resurrection_skill | 살아있는 target과 부활 skill이 충돌한다. |
| actor_has_no_skill | actor에게 skillDescription이 없다. |
| self_skill_without_explicit_target | 명시 target 없이 self skill을 사용해야 한다. |
| aoe_skill_requires_single_center_target | AOE skill이지만 target 하나만 출력해야 한다. |
| mobility_skill_self_escape | self mobility skill로 이탈해야 한다. |
| mobility_skill_enemy_approach | mobility skill로 적에게 접근해야 한다. |
| explicit_wait_duration | 대기 시간이 명시되어 있다. |
| wait_then_attack | 대기 후 공격해야 한다. |
| wait_then_skill | 대기 후 스킬을 써야 한다. |
| skillControl_duration_unspecified | 스킬 지연 시간이 명시되지 않았다. |
| multi_actor_skillControl | 여러 actor에게 skillControl을 적용해야 한다. |
| empty_action_expected | 빈 action이 기대된다. |
| no_valid_skill_actor | 유효 skill actor가 없다. |
| no_valid_skill_target | 유효 skill target이 없다. |
