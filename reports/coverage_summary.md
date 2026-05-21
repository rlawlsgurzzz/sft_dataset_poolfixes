# Coverage Summary

## intent_family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack | 29.8% | 24% | 1221 |
| move | 21.0% | 22% | 861 |
| skill | 24.1% | 26% | 988 |
| skillControl | 7.8% | 8% | 321 |
| wait | 7.4% | 8% | 304 |
| empty | 9.8% | 12% | 400 |

## Actor Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_actor | 57.6% | 40% | 2357 |
| explicit_multi_actor | 14.7% | 12% | 601 |
| global_condition | 10.1% | 16% | 413 |
| global_role_based | 6.4% | 12% | 263 |
| global_state_based | 6.1% | 15% | 248 |
| no_valid_actor | 5.2% | 5% | 213 |

## Target Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_enemy_target | 27.9% | 21% | 1141 |
| explicit_ally_target | 11.7% | 13% | 479 |
| nearest_enemy | 4.8% | 8% | 196 |
| lowest_hp_enemy | 5.3% | 8% | 215 |
| highest_threat_enemy | 3.9% | 7% | 160 |
| role_based_enemy | 6.3% | 8% | 259 |
| pressure_source_enemy | 2.5% | 6% | 104 |
| safe_ally | 3.6% | 6% | 147 |
| low_hp_ally | 4.2% | 6% | 170 |
| backline_ally | 1.7% | 4% | 70 |
| invalid_explicit_target | 7.8% | 7% | 321 |
| none | 20.3% | 6% | 833 |

## Action Pattern

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack_only | 10.5% | 13% | 430 |
| move_only | 12.3% | 13% | 504 |
| move_then_attack | 12.2% | 11% | 499 |
| skill_only | 16.8% | 19% | 689 |
| move_then_skill | 1.1% | 4% | 47 |
| wait_only | 4.9% | 6% | 202 |
| wait_then_attack | 1.3% | 3% | 54 |
| wait_then_skill | 0.0% | 2% | 1 |
| skillControl_defer | 3.6% | 4% | 147 |
| skillControl_forbid | 3.1% | 4% | 128 |
| multi_actor_same_target | 8.9% | 6% | 363 |
| multi_actor_different_targets | 3.5% | 3% | 145 |
| empty_action_expected | 21.6% | 12% | 886 |

## Command Style

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| direct_korean | 31.7% | 35% | 1299 |
| casual_korean | 30.8% | 20% | 1262 |
| elliptical_korean | 8.2% | 20% | 335 |
| tactical_korean | 12.0% | 15% | 491 |
| rough_korean | 17.3% | 10% | 708 |

## Skill Family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_single_target_attack | 29.2% | 18% | 290 |
| self_buff | 8.2% | 14% | 81 |
| ally_shield | 6.7% | 13% | 66 |
| ally_heal | 14.4% | 11% | 143 |
| ally_resurrection | 12.9% | 13% | 128 |
| enemy_aoe_attack | 12.0% | 12% | 119 |
| enemy_debuff | 2.7% | 8% | 27 |
| mobility_skill | 8.1% | 5% | 80 |
| no_skill | 5.8% | 6% | 58 |

## Skill Target Kind

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_alive | 35.0% | 36% | 347 |
| ally_alive | 26.6% | 22% | 264 |
| self | 15.9% | 16% | 158 |
| ally_dead | 7.4% | 12% | 73 |
| enemy_dead | 9.3% | 5% | 92 |
| none | 5.8% | 9% | 58 |

## Conflict Type

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| null | 53.5% | 48% | 531 |
| text_enemy_target_but_self_skill | 4.5% | 8% | 45 |
| text_enemy_target_but_ally_skill | 8.5% | 9% | 84 |
| text_ally_target_but_enemy_skill | 8.7% | 9% | 86 |
| text_dead_target_but_skill_cannot_target_dead | 9.6% | 8% | 95 |
| text_living_target_but_resurrection_skill | 5.8% | 6% | 58 |
| skill_actor_has_no_skill | 5.8% | 7% | 58 |

## Taxonomy Errors

No taxonomy errors found.
