# Deferred consolidation and reconstruction

CoreV2 uses one entry per selected episode; explicit Segmentation duplicates are removed and Rigid is reduced from 27 to six (four singles, one double, one triple; nine target obligations). The initial 52-entry selection changed membership only. The user-authorized Mapping refresh expands it to 54 entries and applies the existing joint-world evaluator to two five-question Connectivity episodes.

## Implemented Mapping refresh

| Subclass | Selected building | Episode structure | Remaining evidence |
| --- | --- | --- | --- |
| Connectivity | school_geography | Five unique adjacency questions, four distinct rooms; one answer sheet and shared unordered visits | Native joint-world relation/door verification and budget calibration |
| Connectivity | house_double_floor_upper | Five unique adjacency questions, five distinct rooms; one answer sheet and shared unordered visits | Same gates; source questions include three Yes and two No answers |
| Long-Term Navigation | office_cubicles_left | Four ordered arrivals through four rooms; approximately 43.8 m source planner path | Native traversal, doorway clearance and budget calibration; initialization passed |
| Long-Term Navigation | Beechwood_0_int | Six ordered arrivals through six distinct rooms | Same route execution gates |

The two hall_arch_wood entries are removed from CoreV2 but preserved in original Core and on disk. The two Connectivity parents have no separate child cards in CoreV2. Navigation remains an ordered route task: unordered arrival sharing must never erase required returns. The shared-world native measurement provider remains a release gate; a complete screenshot bundle does not supply that certificate. Record definitions and pending checks in [corev2_mapping_refresh.json](/local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/lists/corev2_mapping_refresh.json).

| Proposed family | Existing subclasses | Required distinction |
| --- | --- | --- |
| Stacking | Action Order Inference; Stacking & Stability | Supplied versus inferred order; eventually select 2–3 complementary tasks |
| Search and counting | Counting w Occlusion; Merged Observation; Spatial Segmentation | Search region, observation accumulation, segmentation; category ambiguity and illumination may become tags |
| Building exploration | Connectivity; Regional Boundary | Shared world and answer sheet, preserving both topology and boundary scoring |
| Geometric arrangement | Geometric Configuration; Linear Alignment | Shape constraints and alignment tolerances |
| Active visual discrimination | Partial Occlusion; View Hallucination | Viewpoint-dependent evidence versus misleading projections |
| Reflection | Three reflection subclasses | Actual optical geometry and source head-pose verification must pass first |
| Containment | Material Transparency; Rigid Containment | Retain transparency only when it changes required reasoning |

Keep temporal observation-history and unobserved world-change distinct. Keep physical uncovering of deformable objects distinct from viewpoint-only occlusion. Preserve route execution versus route selection tags.

## Shared-scene episodes to implement later

The earlier Connectivity/Regional inventory had 28 source tasks across ten buildings. If worlds and initial states are compatible, its 20 default entries could become ten building episodes. The new Connectivity parents do not implement any cross-subclass merge; the following table remains a historical reconstruction backlog.

| Building | Current entries → proposed episode | Blocker / priority |
| --- | --- | --- |
| restaurant_cafeteria | 2 → 1 | Both setups accepted; first compatibility review |
| hall_arch_wood | 2 → 1 | Boundary run cancelled |
| school_biology | 3 → 1 | Historical reconstruction option only; the CoreV2 proposal was rejected during native initialization |
| school_gym | 2 → 1 | Connectivity failed |
| hall_glass_ceiling | 2 → 1 | Setup failure |
| Merom_0_garden | 2 → 1 | Setup failure |
| hall_train_station | 2 → 1 | Setup failure |
| office_vendor_machine | 3 → 1 | Failed/interrupted evidence |
| Rs_garden | 1 → 1 | Existing five-source parent |
| Rs_int | 1 → 1 | Single-source building |

Search parents also need separate attention: house_single_floor combines three mechanisms but its latest parent failed; an accepted older counting source cannot silently replace its coverage. office_cubicles_right combines illumination and segmentation but remains an additional, unqualified prototype. The earlier audit found 23 same-room cross-category groups; room equality alone is insufficient evidence of compatible object states, physics, robot pose or observation protocol.

Before merging, compare exact world patches, initial robot/object states, public goals, observable evidence, scoring and budgets. Keep one parent entry only after all child goals are scored independently and the shared setup passes. Do not sum old task budgets blindly or transfer old setup certificates to a changed world. Failed or unknown classes stay excluded instead of being filled from semantically different historical backups.
