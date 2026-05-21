# Coverage Summary

## intent_family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack | 28.0% | 24% | 1068 |
| move | 20.2% | 22% | 769 |
| skill | 28.1% | 26% | 1072 |
| skillControl | 7.1% | 8% | 271 |
| wait | 7.1% | 8% | 272 |
| empty | 9.5% | 12% | 363 |

## Actor Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_actor | 57.9% | 40% | 2207 |
| explicit_multi_actor | 16.5% | 12% | 631 |
| global_condition | 8.5% | 16% | 326 |
| global_role_based | 6.2% | 12% | 236 |
| global_state_based | 5.7% | 15% | 216 |
| no_valid_actor | 5.2% | 5% | 199 |

## Target Selection

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| explicit_enemy_target | 27.0% | 21% | 1029 |
| explicit_ally_target | 11.4% | 13% | 434 |
| nearest_enemy | 4.8% | 8% | 184 |
| lowest_hp_enemy | 5.5% | 8% | 211 |
| highest_threat_enemy | 4.2% | 7% | 160 |
| role_based_enemy | 7.8% | 8% | 298 |
| pressure_source_enemy | 2.5% | 6% | 96 |
| safe_ally | 4.2% | 6% | 160 |
| low_hp_ally | 4.5% | 6% | 172 |
| backline_ally | 1.6% | 4% | 60 |
| invalid_explicit_target | 7.3% | 7% | 278 |
| none | 19.2% | 6% | 733 |

## Action Pattern

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| attack_only | 9.9% | 13% | 379 |
| move_only | 11.5% | 13% | 437 |
| move_then_attack | 11.1% | 11% | 425 |
| skill_only | 18.8% | 19% | 718 |
| move_then_skill | 1.4% | 4% | 52 |
| wait_only | 4.3% | 6% | 165 |
| wait_then_attack | 1.4% | 3% | 54 |
| wait_then_skill | 0.0% | 2% | 0 |
| skillControl_defer | 3.1% | 4% | 117 |
| skillControl_forbid | 2.5% | 4% | 97 |
| multi_actor_same_target | 8.7% | 6% | 330 |
| multi_actor_different_targets | 4.2% | 3% | 161 |
| empty_action_expected | 23.1% | 12% | 880 |

## Command Style

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| direct_korean | 29.4% | 35% | 1120 |
| casual_korean | 32.0% | 20% | 1219 |
| elliptical_korean | 7.7% | 20% | 295 |
| tactical_korean | 13.3% | 15% | 506 |
| rough_korean | 17.7% | 10% | 675 |

## Skill Family

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_single_target_attack | 29.9% | 18% | 321 |
| self_buff | 6.5% | 14% | 70 |
| ally_shield | 7.5% | 13% | 80 |
| ally_heal | 16.8% | 11% | 180 |
| ally_resurrection | 10.1% | 13% | 108 |
| enemy_aoe_attack | 11.8% | 12% | 127 |
| enemy_debuff | 6.5% | 8% | 70 |
| mobility_skill | 4.8% | 5% | 51 |
| no_skill | 6.1% | 6% | 65 |

## Skill Target Kind

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| enemy_alive | 41.0% | 36% | 439 |
| ally_alive | 26.7% | 22% | 286 |
| self | 10.2% | 16% | 109 |
| ally_dead | 7.6% | 12% | 82 |
| enemy_dead | 8.5% | 5% | 91 |
| none | 6.1% | 9% | 65 |

## Conflict Type

| key | current_ratio | target_ratio | count |
|---|---:|---:|---:|
| null | 56.5% | 48% | 606 |
| text_enemy_target_but_self_skill | 3.3% | 8% | 35 |
| text_enemy_target_but_ally_skill | 7.8% | 9% | 84 |
| text_ally_target_but_enemy_skill | 8.0% | 9% | 86 |
| text_dead_target_but_skill_cannot_target_dead | 10.5% | 8% | 113 |
| text_living_target_but_resurrection_skill | 4.5% | 6% | 48 |
| skill_actor_has_no_skill | 6.1% | 7% | 65 |

## Taxonomy Errors

No taxonomy errors found.
