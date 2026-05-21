# Coverage Summary

## intent_family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack | 29.8% | 24% | 1019 |
| move | 21.1% | 22% | 721 |
| skill | 23.8% | 26% | 816 |
| skillControl | 7.6% | 8% | 260 |
| wait | 7.6% | 8% | 262 |
| empty | 10.1% | 12% | 347 |

## Actor Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_actor | 56.4% | 40% | 1932 |
| explicit_multi_actor | 15.9% | 12% | 544 |
| global_condition | 9.4% | 16% | 322 |
| global_role_based | 6.8% | 12% | 232 |
| global_state_based | 6.1% | 15% | 210 |
| no_valid_actor | 5.4% | 5% | 185 |

## Target Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_enemy_target | 27.1% | 21% | 927 |
| explicit_ally_target | 11.0% | 13% | 377 |
| nearest_enemy | 4.8% | 8% | 164 |
| lowest_hp_enemy | 5.1% | 8% | 173 |
| highest_threat_enemy | 4.7% | 7% | 160 |
| role_based_enemy | 7.2% | 8% | 248 |
| pressure_source_enemy | 2.5% | 6% | 84 |
| safe_ally | 4.0% | 6% | 137 |
| low_hp_ally | 4.1% | 6% | 139 |
| backline_ally | 1.8% | 4% | 60 |
| invalid_explicit_target | 8.2% | 7% | 280 |
| none | 19.7% | 6% | 676 |

## Action Pattern

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack_only | 10.6% | 13% | 363 |
| move_only | 11.7% | 13% | 402 |
| move_then_attack | 11.8% | 11% | 403 |
| skill_only | 16.8% | 19% | 575 |
| move_then_skill | 1.0% | 4% | 35 |
| wait_only | 4.7% | 6% | 160 |
| wait_then_attack | 1.6% | 3% | 54 |
| wait_then_skill | 0.0% | 2% | 1 |
| skillControl_defer | 3.4% | 4% | 117 |
| skillControl_forbid | 2.8% | 4% | 97 |
| multi_actor_same_target | 8.4% | 6% | 289 |
| multi_actor_different_targets | 3.9% | 3% | 133 |
| empty_action_expected | 23.2% | 12% | 796 |

## Command Style

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| direct_korean | 31.1% | 35% | 1065 |
| casual_korean | 31.0% | 20% | 1063 |
| elliptical_korean | 8.2% | 20% | 280 |
| tactical_korean | 12.3% | 15% | 420 |
| rough_korean | 17.4% | 10% | 597 |

## Skill Family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_single_target_attack | 29.6% | 18% | 243 |
| self_buff | 8.7% | 14% | 71 |
| ally_shield | 7.8% | 13% | 64 |
| ally_heal | 14.9% | 11% | 122 |
| ally_resurrection | 13.2% | 13% | 108 |
| enemy_aoe_attack | 10.4% | 12% | 85 |
| enemy_debuff | 3.3% | 8% | 27 |
| mobility_skill | 5.1% | 5% | 42 |
| no_skill | 7.1% | 6% | 58 |

## Skill Target Kind

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_alive | 33.7% | 36% | 276 |
| ally_alive | 28.2% | 22% | 231 |
| self | 13.4% | 16% | 110 |
| ally_dead | 7.7% | 12% | 63 |
| enemy_dead | 10.0% | 5% | 82 |
| none | 7.1% | 9% | 58 |

## Conflict Type

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| null | 47.4% | 48% | 389 |
| text_enemy_target_but_self_skill | 4.3% | 8% | 35 |
| text_enemy_target_but_ally_skill | 10.2% | 9% | 84 |
| text_ally_target_but_enemy_skill | 10.5% | 9% | 86 |
| text_dead_target_but_skill_cannot_target_dead | 10.4% | 8% | 85 |
| text_living_target_but_resurrection_skill | 5.9% | 6% | 48 |
| skill_actor_has_no_skill | 7.1% | 7% | 58 |

## Taxonomy Errors

No taxonomy errors found.
