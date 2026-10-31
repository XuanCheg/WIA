# WIA v0.2 dataset card

## Scope and composition

World Interactive Agency evaluates embodied perception, physical action and evidence-dependent decisions in native simulated scenes. The main selection contains 60 definitions: 46 retained original tasks, one additional already-contained control (T52), nine InteractiveWorld-inspired mechanisms and four everyday household geometry scenes. Each catalogue alias has one selected scene variant. The task/scene pair is one instance, not two independent examples. There are 59 legacy task IDs: the R-A1/R-A2 direct/reflected controls intentionally share one. Select and report those two by alias and definition hash; never merge them by task ID.

| Family | Meaning | Instances |
| --- | --- | ---: |
| AP | Action Planning | 11 |
| NT | Navigation and Topology | 8 |
| GF | Geometry and Fit | 8 |
| SR | Spatial Relations | 9 |
| GG | Grounding and Goal-directed Handling | 16 |
| TM | Temporal Memory | 8 |

The precise disjoint primary taxonomy and subclass coverage are in `catalogue.json`. Categories describe intended constructs; taxonomy validity and empirical difficulty are not established by selection. No train/test split or successful demonstration corpus is claimed. Ten balance extensions, retired D tasks and repetitive candidate variants remain outside this release.

## Task semantics

AP-1 evaluates the final specified support structure rather than inference of a hidden action order. AP-4 includes closed-loop recovery mechanisms. T02 permits the cutting board as a base and completion at any location. Relevant final-state tasks do not require a prior grasp or a five-centimeter lift. T22–T25 allow stable per-item verification followed by removal to free shared containers. Movable background furniture may be moved when the task permits it; explicit protected apparatus rules still apply.

H1 uses a rigid package with a back-side acceptance mark. M1 retains cleaning-tool use context and visible usage events; eligibility is an event state, not a simulated contamination claim. M4 measures processing and transport on the same simulation clock and pauses the world during model inference; processing requires correct station occupancy. GF2-1/GF2-2 use an orientation-constrained juice-carton cubby and cutting-board drying slot. GF3-1/GF3-2 assign cucumbers by length and spice jars by height.

## Evaluation protocol

Use task defaults for seeds, step budgets, world timing, action mode and observation condition. Record model identity, backend, generation settings, history policy, overrides, runtime, latency and cost alongside task outcomes. Do not combine runs that change these conditions without reporting the difference. The public task payload and authorized robot sensors are policy inputs; private geometry, answer keys, evaluator state and world coordinates used for privileged debugging are not ordinary benchmark observations.

Report per-task success and outcomes, then per-family results. Subclass-then-family macro averaging is a reporting recommendation; the runtime does not silently reweight scores. Physical failures, invalid model outputs, timeouts and initialization failures must remain distinct. Report runtime, decision count and model cost where available. Nine single mechanism representatives do not alone establish counterfactual sensitivity; separate controlled variants are needed for that claim.

## Evidence and limitations

The earlier pre-release scan loaded all 59 original selected definitions without errors and found a rendered capture for each. This covers initialization and inspection, not successful manipulation. Existing household evidence covers four native scene initializations, seven pickup targets and seven goal/staging targets with bounded geometric witnesses. It accounts for torso motion, joint limits, sampled stances, collision proxies and sampled arm paths. The four scenes received no policy rollout in that validation. Static witnesses do not establish grasp closure, contact force, transport or insertion success.

T22/T46 chair-access concerns were closed by human review, and additional GF verification was waived after visual inspection. The follow-up directly tested 27 ramp releases without robot actions or model calls: the ball was admissible in every tested scene, and T44's bowl admission varied at the tie boundary. T52 adds an already-contained example while leaving T15–T17 unchanged; its exact definition passed native Inside/contact/stability verification with zero robot actions and no model calls, and 21 native review views were captured. All definitions preserve their pending qualification status. See `KNOWN_ISSUES.md` and `VALIDATION.md` for the exact evidence and remaining limitations. Private review notes and screenshot caches are not part of this data package.

## Provenance and licensing

The source manifest distinguishes authored inputs from externally obtained ESI records. Native assets remain external under their own terms. WIA's Apache-2.0 license applies to its contributions and does not relicense upstream content. Export provenance records normalization separately from the original captured files. See `THIRD_PARTY.md` for component boundaries.
