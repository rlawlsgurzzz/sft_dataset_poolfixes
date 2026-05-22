# Coverage Summary

## intent_family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack | 24.3% | 24% | 1444 |
| move | 21.8% | 22% | 1299 |
| skill | 26.3% | 26% | 1564 |
| skillControl | 8.0% | 8% | 476 |
| wait | 8.0% | 8% | 476 |
| empty | 11.6% | 12% | 689 |

## Actor Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_actor | 50.2% | 40% | 2984 |
| explicit_multi_actor | 13.5% | 12% | 801 |
| global_condition | 12.7% | 16% | 756 |
| global_role_based | 9.3% | 12% | 553 |
| global_state_based | 9.6% | 15% | 571 |
| no_valid_actor | 4.8% | 5% | 283 |

## Target Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_enemy_target | 23.4% | 21% | 1390 |
| explicit_ally_target | 10.5% | 13% | 627 |
| nearest_enemy | 5.6% | 8% | 332 |
| lowest_hp_enemy | 7.4% | 8% | 440 |
| highest_threat_enemy | 3.2% | 7% | 190 |
| role_based_enemy | 7.9% | 8% | 472 |
| pressure_source_enemy | 2.5% | 6% | 146 |
| safe_ally | 6.1% | 6% | 360 |
| low_hp_ally | 7.4% | 6% | 442 |
| backline_ally | 1.4% | 4% | 84 |
| invalid_explicit_target | 6.2% | 7% | 368 |
| none | 18.4% | 6% | 1097 |

## Action Pattern

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack_only | 9.4% | 13% | 559 |
| move_only | 13.9% | 13% | 828 |
| move_then_attack | 10.5% | 11% | 625 |
| skill_only | 18.4% | 19% | 1096 |
| move_then_skill | 1.8% | 4% | 106 |
| wait_only | 5.3% | 6% | 314 |
| wait_then_attack | 1.8% | 3% | 109 |
| wait_then_skill | 0.0% | 2% | 0 |
| skillControl_defer | 3.8% | 4% | 226 |
| skillControl_forbid | 3.2% | 4% | 193 |
| multi_actor_same_target | 6.9% | 6% | 410 |
| multi_actor_different_targets | 3.2% | 3% | 191 |
| empty_action_expected | 21.7% | 12% | 1291 |

## Command Style

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| direct_korean | 26.7% | 35% | 1591 |
| casual_korean | 32.1% | 20% | 1910 |
| elliptical_korean | 7.2% | 20% | 431 |
| tactical_korean | 16.2% | 15% | 961 |
| rough_korean | 17.7% | 10% | 1055 |

## Skill Family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_single_target_attack | 38.4% | 18% | 601 |
| self_buff | 5.8% | 14% | 90 |
| ally_shield | 5.1% | 13% | 80 |
| ally_heal | 19.6% | 11% | 307 |
| ally_resurrection | 8.8% | 13% | 138 |
| enemy_aoe_attack | 9.4% | 12% | 147 |
| enemy_debuff | 4.8% | 8% | 75 |
| mobility_skill | 3.3% | 5% | 51 |
| no_skill | 4.8% | 6% | 75 |

## Skill Target Kind

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_alive | 46.0% | 36% | 719 |
| ally_alive | 27.0% | 22% | 423 |
| self | 8.2% | 16% | 129 |
| ally_dead | 6.5% | 12% | 102 |
| enemy_dead | 7.4% | 5% | 116 |
| none | 4.8% | 9% | 75 |

## Conflict Type

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| null | 64.1% | 48% | 1003 |
| text_enemy_target_but_self_skill | 2.9% | 8% | 45 |
| text_enemy_target_but_ally_skill | 6.0% | 9% | 94 |
| text_ally_target_but_enemy_skill | 6.8% | 9% | 106 |
| text_dead_target_but_skill_cannot_target_dead | 8.8% | 8% | 138 |
| text_living_target_but_resurrection_skill | 3.7% | 6% | 58 |
| skill_actor_has_no_skill | 4.8% | 7% | 75 |

## Taxonomy Errors

No taxonomy errors found.
