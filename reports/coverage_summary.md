# Coverage Summary

## intent_family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack | 22.7% | 24% | 843 |
| move | 23.2% | 22% | 861 |
| skill | 26.4% | 26% | 981 |
| skillControl | 8.7% | 8% | 321 |
| wait | 8.2% | 8% | 304 |
| empty | 10.8% | 12% | 400 |

## Actor Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_actor | 57.2% | 40% | 2122 |
| explicit_multi_actor | 14.5% | 12% | 537 |
| global_condition | 9.4% | 16% | 348 |
| global_role_based | 6.5% | 12% | 242 |
| global_state_based | 6.7% | 15% | 248 |
| no_valid_actor | 5.7% | 5% | 213 |

## Target Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_enemy_target | 26.1% | 21% | 969 |
| explicit_ally_target | 12.9% | 13% | 479 |
| nearest_enemy | 4.5% | 8% | 167 |
| lowest_hp_enemy | 5.2% | 8% | 194 |
| highest_threat_enemy | 3.1% | 7% | 114 |
| role_based_enemy | 6.0% | 8% | 222 |
| pressure_source_enemy | 1.9% | 6% | 69 |
| safe_ally | 4.0% | 6% | 147 |
| low_hp_ally | 4.6% | 6% | 170 |
| backline_ally | 1.9% | 4% | 70 |
| invalid_explicit_target | 7.6% | 7% | 283 |
| none | 22.3% | 6% | 826 |

## Action Pattern

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack_only | 8.2% | 13% | 303 |
| move_only | 13.6% | 13% | 504 |
| move_then_attack | 11.0% | 11% | 409 |
| skill_only | 18.6% | 19% | 689 |
| move_then_skill | 1.3% | 4% | 47 |
| wait_only | 5.4% | 6% | 202 |
| wait_then_attack | 1.5% | 3% | 54 |
| wait_then_skill | 0.0% | 2% | 1 |
| skillControl_defer | 4.0% | 4% | 147 |
| skillControl_forbid | 3.5% | 4% | 128 |
| multi_actor_same_target | 7.3% | 6% | 271 |
| multi_actor_different_targets | 3.1% | 3% | 114 |
| empty_action_expected | 22.7% | 12% | 841 |

## Command Style

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| direct_korean | 30.4% | 35% | 1129 |
| casual_korean | 31.6% | 20% | 1172 |
| elliptical_korean | 8.5% | 20% | 317 |
| tactical_korean | 12.3% | 15% | 458 |
| rough_korean | 17.1% | 10% | 634 |

## Skill Family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_single_target_attack | 29.4% | 18% | 290 |
| self_buff | 8.2% | 14% | 81 |
| ally_shield | 6.7% | 13% | 66 |
| ally_heal | 14.5% | 11% | 143 |
| ally_resurrection | 13.0% | 13% | 128 |
| enemy_aoe_attack | 12.1% | 12% | 119 |
| enemy_debuff | 2.7% | 8% | 27 |
| mobility_skill | 8.1% | 5% | 80 |
| no_skill | 5.2% | 6% | 51 |

## Skill Target Kind

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_alive | 35.2% | 36% | 347 |
| ally_alive | 26.8% | 22% | 264 |
| self | 16.0% | 16% | 158 |
| ally_dead | 7.4% | 12% | 73 |
| enemy_dead | 9.3% | 5% | 92 |
| none | 5.2% | 9% | 51 |

## Conflict Type

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| null | 53.9% | 48% | 531 |
| text_enemy_target_but_self_skill | 4.6% | 8% | 45 |
| text_enemy_target_but_ally_skill | 8.5% | 9% | 84 |
| text_ally_target_but_enemy_skill | 8.7% | 9% | 86 |
| text_dead_target_but_skill_cannot_target_dead | 9.6% | 8% | 95 |
| text_living_target_but_resurrection_skill | 5.9% | 6% | 58 |
| skill_actor_has_no_skill | 5.2% | 7% | 51 |

## Taxonomy Errors

No taxonomy errors found.
