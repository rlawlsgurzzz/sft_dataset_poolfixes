# Coverage Summary

## intent_family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack | 27.2% | 24% | 1315 |
| move | 20.6% | 22% | 994 |
| skill | 27.7% | 26% | 1336 |
| skillControl | 7.2% | 8% | 350 |
| wait | 7.3% | 8% | 352 |
| empty | 10.0% | 12% | 483 |

## Actor Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_actor | 55.1% | 40% | 2663 |
| explicit_multi_actor | 15.5% | 12% | 751 |
| global_condition | 10.1% | 16% | 486 |
| global_role_based | 7.4% | 12% | 356 |
| global_state_based | 6.9% | 15% | 335 |
| no_valid_actor | 4.9% | 5% | 239 |

## Target Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_enemy_target | 25.9% | 21% | 1251 |
| explicit_ally_target | 11.3% | 13% | 544 |
| nearest_enemy | 5.5% | 8% | 264 |
| lowest_hp_enemy | 6.0% | 8% | 291 |
| highest_threat_enemy | 3.9% | 7% | 190 |
| role_based_enemy | 7.8% | 8% | 378 |
| pressure_source_enemy | 2.6% | 6% | 126 |
| safe_ally | 4.6% | 6% | 220 |
| low_hp_ally | 4.8% | 6% | 231 |
| backline_ally | 1.4% | 4% | 70 |
| invalid_explicit_target | 7.0% | 7% | 338 |
| none | 19.2% | 6% | 927 |

## Action Pattern

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack_only | 10.2% | 13% | 494 |
| move_only | 11.8% | 13% | 572 |
| move_then_attack | 11.1% | 11% | 537 |
| skill_only | 18.6% | 19% | 897 |
| move_then_skill | 1.7% | 4% | 82 |
| wait_only | 4.7% | 6% | 225 |
| wait_then_attack | 1.5% | 3% | 74 |
| wait_then_skill | 0.0% | 2% | 0 |
| skillControl_defer | 3.3% | 4% | 157 |
| skillControl_forbid | 2.8% | 4% | 136 |
| multi_actor_same_target | 8.1% | 6% | 390 |
| multi_actor_different_targets | 4.0% | 3% | 191 |
| empty_action_expected | 22.3% | 12% | 1075 |

## Command Style

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| direct_korean | 27.3% | 35% | 1321 |
| casual_korean | 32.7% | 20% | 1578 |
| elliptical_korean | 6.2% | 20% | 298 |
| tactical_korean | 16.0% | 15% | 773 |
| rough_korean | 17.8% | 10% | 860 |

## Skill Family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_single_target_attack | 34.5% | 18% | 461 |
| self_buff | 6.7% | 14% | 90 |
| ally_shield | 6.0% | 13% | 80 |
| ally_heal | 16.4% | 11% | 219 |
| ally_resurrection | 10.3% | 13% | 138 |
| enemy_aoe_attack | 11.0% | 12% | 147 |
| enemy_debuff | 5.6% | 8% | 75 |
| mobility_skill | 3.8% | 5% | 51 |
| no_skill | 5.6% | 6% | 75 |

## Skill Target Kind

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_alive | 43.3% | 36% | 579 |
| ally_alive | 25.1% | 22% | 335 |
| self | 9.7% | 16% | 129 |
| ally_dead | 7.6% | 12% | 102 |
| enemy_dead | 8.7% | 5% | 116 |
| none | 5.6% | 9% | 75 |

## Conflict Type

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| null | 58.0% | 48% | 775 |
| text_enemy_target_but_self_skill | 3.4% | 8% | 45 |
| text_enemy_target_but_ally_skill | 7.0% | 9% | 94 |
| text_ally_target_but_enemy_skill | 7.9% | 9% | 106 |
| text_dead_target_but_skill_cannot_target_dead | 10.3% | 8% | 138 |
| text_living_target_but_resurrection_skill | 4.3% | 6% | 58 |
| skill_actor_has_no_skill | 5.6% | 7% | 75 |

## Taxonomy Errors

No taxonomy errors found.
