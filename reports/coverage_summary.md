# Coverage Summary

## intent_family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack | 28.0% | 24% | 1073 |
| move | 20.2% | 22% | 774 |
| skill | 28.1% | 26% | 1077 |
| skillControl | 7.1% | 8% | 271 |
| wait | 7.1% | 8% | 272 |
| empty | 9.5% | 12% | 363 |

## Actor Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_actor | 58.0% | 40% | 2222 |
| explicit_multi_actor | 16.5% | 12% | 631 |
| global_condition | 8.5% | 16% | 326 |
| global_role_based | 6.2% | 12% | 236 |
| global_state_based | 5.6% | 15% | 216 |
| no_valid_actor | 5.2% | 5% | 199 |

## Target Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_enemy_target | 27.1% | 21% | 1039 |
| explicit_ally_target | 11.3% | 13% | 434 |
| nearest_enemy | 4.8% | 8% | 184 |
| lowest_hp_enemy | 5.5% | 8% | 211 |
| highest_threat_enemy | 4.2% | 7% | 160 |
| role_based_enemy | 7.8% | 8% | 298 |
| pressure_source_enemy | 2.5% | 6% | 96 |
| safe_ally | 4.2% | 6% | 160 |
| low_hp_ally | 4.5% | 6% | 172 |
| backline_ally | 1.6% | 4% | 60 |
| invalid_explicit_target | 7.3% | 7% | 278 |
| none | 19.3% | 6% | 738 |

## Action Pattern

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack_only | 10.0% | 13% | 384 |
| move_only | 11.5% | 13% | 442 |
| move_then_attack | 11.1% | 11% | 425 |
| skill_only | 18.7% | 19% | 718 |
| move_then_skill | 1.4% | 4% | 52 |
| wait_only | 4.3% | 6% | 165 |
| wait_then_attack | 1.4% | 3% | 54 |
| wait_then_skill | 0.0% | 2% | 0 |
| skillControl_defer | 3.1% | 4% | 117 |
| skillControl_forbid | 2.5% | 4% | 97 |
| multi_actor_same_target | 8.6% | 6% | 330 |
| multi_actor_different_targets | 4.2% | 3% | 161 |
| empty_action_expected | 23.1% | 12% | 885 |

## Command Style

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| direct_korean | 29.3% | 35% | 1123 |
| casual_korean | 32.0% | 20% | 1224 |
| elliptical_korean | 7.7% | 20% | 296 |
| tactical_korean | 13.3% | 15% | 510 |
| rough_korean | 17.7% | 10% | 677 |

## Skill Family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_single_target_attack | 29.8% | 18% | 321 |
| self_buff | 6.5% | 14% | 70 |
| ally_shield | 7.4% | 13% | 80 |
| ally_heal | 16.7% | 11% | 180 |
| ally_resurrection | 10.0% | 13% | 108 |
| enemy_aoe_attack | 11.8% | 12% | 127 |
| enemy_debuff | 7.0% | 8% | 75 |
| mobility_skill | 4.7% | 5% | 51 |
| no_skill | 6.0% | 6% | 65 |

## Skill Target Kind

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_alive | 40.8% | 36% | 439 |
| ally_alive | 26.6% | 22% | 286 |
| self | 10.1% | 16% | 109 |
| ally_dead | 7.6% | 12% | 82 |
| enemy_dead | 8.9% | 5% | 96 |
| none | 6.0% | 9% | 65 |

## Conflict Type

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| null | 56.3% | 48% | 606 |
| text_enemy_target_but_self_skill | 3.2% | 8% | 35 |
| text_enemy_target_but_ally_skill | 7.8% | 9% | 84 |
| text_ally_target_but_enemy_skill | 8.0% | 9% | 86 |
| text_dead_target_but_skill_cannot_target_dead | 11.0% | 8% | 118 |
| text_living_target_but_resurrection_skill | 4.5% | 6% | 48 |
| skill_actor_has_no_skill | 6.0% | 7% | 65 |

## Taxonomy Errors

No taxonomy errors found.
