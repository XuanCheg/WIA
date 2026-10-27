# World-Interactive Agency — V1 Taxonomy-first Implementation Spec

**Status:** proposal for classification freeze, no GitHub writes or new native simulations in this review. 2026-10-22.
**Code snapshot:** `33bada4bb278471b7a925cd8ddfa3d46a19f52fb`.

## Research hypothesis and iteration plan

As general agents, VLAs and WAMs become able to act physically, benchmark what makes perception, geometry, memory and planning **convert into correct physical consequences**. This is not an exhaustive benchmark; v1 is a compact diagnostic task suite, and v2 expands independent mechanisms and environments.

1. Freeze scientific taxonomy, stable subclass IDs, and physical success contracts.
2. Reuse existing CoreV2 Tasks/Scenes/Evaluators as first-class assets and annotate repairs.
3. Fill genuine cognitive-mechanism gaps by scene selection with robot-native feasibility checks.
4. Implement task variants only after scene-screening; qualify actual physical trajectory and scoring.

## Frozen v1 taxonomy (categories are stable; number of tasks remains extensible)

| ID | Major family | Subcategory | Retained candidates | Embodied success evidence |
|---|---|---|---:|---|
| AP-1 | Action Planning & Physical Experimentation | Prescribed-order Physical Execution | 3 | 构造中间态及最终承托关系且顺序正确 |
| AP-2 | Action Planning & Physical Experimentation | Stability-Constrained Construction | 4 | 真实释放后满足相互支持与稳定时长 |
| AP-3 | Action Planning & Physical Experimentation | Experiment-to-Action Dynamics | 3 | 完成受控试验、基于不同条件选对物体、实际交付 |
| AP-4 | Action Planning & Physical Experimentation | Closed-loop Recovery & Adaptation | 0 | 真实可逆扰动后，在受控预算内恢复目标终态 |
| NT-1 | Situated Navigation & Topology | Connectivity & Topology Verification | 3 | 真实跨区域检查或充分的本体观测及正确关系判定 |
| NT-2 | Situated Navigation & Topology | Ordered Multi-region Navigation | 2 | 物理到达/稳定且遵守先后关系 |
| NT-3 | Situated Navigation & Topology | Semantic Region Grounding | 3 | 正确观察锚点、到达/报告目标区域（QA作为辅助指标） |
| GF-1 | Geometric Feasibility & Affordance | Containment & Affordance Selection | 4 | 真实拾取、插入/释放、合法撤手和稳定状态 |
| GF-2 | Geometric Feasibility & Affordance | Dimension-constrained Insertion | 3 | 物体和夹爪完整经过限口并满足合法终态 |
| GF-3 | Geometric Feasibility & Affordance | Metric Sorting & Assignment | 2 | 所有目标按规定属性正确放入不同位置 |
| SR-1 | Spatial Reasoning-to-Action | Geometric Arrangement & Alignment | 5 | 实际摆放后满足指定几何关系，排除初态已成的捷径 |
| SR-2 | Spatial Reasoning-to-Action | Contact & Separation Regulation | 4 | 物体被调整至指定接触或分离状态且稳定 |
| SR-3 | Spatial Reasoning-to-Action | Reflection-grounded Reconfiguration | 3 | 镜像判断影响真物体操作后，空间距离/次序正确反转 |
| GG-1 | Goal Grounding & Active Perception | Active Search & Physical Collection | 3 | 正确目标全部真实收集，干扰物未进入终态 |
| GG-2 | Goal Grounding & Active Perception | Occlusion Removal & Retrieval | 3 | 经真实布料接触揭露目标并完成抓取/取回 |
| GG-3 | Goal Grounding & Active Perception | Mirror-grounded Target Selection | 5 | 依据镜像对应/直视分支执行正确交付；正反样本配对 |
| GG-4 | Goal Grounding & Active Perception | Appearance-conditioned Object Handling | 3 | 观察真实内容状态后执行插入或合法不操作，保证开口可达 |
| TM-1 | Temporal Memory & World-State Revision | Episodic Observation-to-Action | 2 | 根据自身真实观察历史执行正确目标操作 |
| TM-2 | Temporal Memory & World-State Revision | Unobserved Change & Contingent Action | 6 | 两阶段比较后按增/删/移/换/不变执行正确操作 |

**Inventory:** 61 candidates = 48 CoreV2 retained + 13 new designs, from 51 original − 3 superseded + 13 designs. Only AP-4 is entirely empty. The new designs include 8 mirror and 5 dimensional-size tasks.

## Reuse decision policy

| Level | Count | Meaning |
|---|---:|---|
| R0 | 18 | 复用完整Task Definition + Evaluator；只补真实性准入验证 |
| R1 | 20 | 复用Task/Evaluator，针对场景、可达性、参数或观测做局部修复 |
| R2 | 10 | 复用Converter/Source/Scene，但任务目标、对照条件或评分需实质修订 |
| N1 | 8 | 已有ESI镜面源与Converter，需创建真实Action Task和场景适配 |
| N2 | 5 | 已有ESI尺寸资产/Converter可参考，限口或排序的物理评价需新实现 |

Reuse level is **not** a launch-readiness certificate. The 48 CoreV2 candidates may have YAML and Evaluators while their native physical success remains unverified. For R0, preserve contract and run qualification. For R2, preserve source identity, converters, and tested layout whenever possible, but fix the scientific mechanism. For mirror N1, reuse the existing optical converters, not their observation-only success criteria. For size N2, the existing original comparison adapter does not establish constrained insertion.

## Known P0 gaps and scene picks

| Slot | Gap / reason | P0 action | Candidate source scene |
|---|---|---|---|
| AP-4 | 真正空缺 | 新增1个可逆世界扰动后的恢复任务；建议源T26场景 | Beechwood_0_garden / utility_room_0 |
| AP-3 | 任务已有，但无法区分实验选择策略 | 让同装置至少两种参数产生不同唯一赢家 | house_single_floor / living_room_0 |
| NT-3 | 词面泄露/QA物理终态弱 | T11移除标签答案泄露；T09/T10强化可验证的锚点到访 | school_gym / full_scene & restaurant_cafeteria |
| GF-2 | 任务设计存在但无物理装置与评分 | D1-1、D1-2先实现，D1-3留压力样本 | Beechwood_0_garden / utility_room_0; Merom_0_garden / utility_room_0 |
| SR-3 | 原镜面Converter有，物理重构任务未构建 | R-S1优先，R-S3/2备选 | Merom_0_garden / living_room_0 |
| GG-3 | 原镜面源/Converter有，物理交付未构建 | R-C1+C3+R-A2先形成正负对照 | Ihlen_0_int / garage_0; restaurant_hotel / dining_room_0 |
| GG-4 | 已有Task但原容器可能物理不可穿过 | 替换/新增宽口版本，保留源任务作为对照/档案 | Beechwood_0_garden / dining_room_0 |
| TM-2 | 同一scene和固定初态导致答案捷径 | 初态置换、同当前观测历史对照、完整两阶段操作 | house_single_floor / dining_room_0 |

## Scene-selection protocol

Classify each candidate as `PASS` / `FAIL` / `INCONCLUSIVE` at each gate. A failure in a required gate excludes the scenario from publication, without declaring all scenarios in its source subclass impossible.

| Gate | Measurement | Required evidence |
|---|---|---|
| G0 Science/Shortcut | 任务存在单一可验证认知挑战；初态不能直接泄露答案 | 初态视图、无历史/无交互/固定策略基线 |
| G1 Native Loadability | 真实source/scene assets和原物体装载；无穿模、悬浮或先天任务已完成 | 原生OmniGibson scene initialization+settle |
| G2 Actual Geometry | 机器人可通行面积 A_nav、物体/工作区范围 A_task、真实最短可行路长 | floor map+R1 chassis膨胀、实际路径搜索 |
| G3 Relevant Entity Inventory | 目标/选项/干扰/容器/支持台/可动物品分别计数;不以prim数冒充 | Task binding registry + native object registry |
| G4 Reachability & Grasp | 至少一个从机器人真实初始姿态出发的操作姿态，夹爪/臂、碰撞、抬升可行 | R1动作+IK/碰撞+抓取与释放 |
| G5 Whole Action Feasibility | 必须检查pickup→transport→insert/stack→release→withdraw，含特殊关门/布料动作 | 一条合法完整物理参考轨迹 |
| G6 Agent Sensor Evidence | 所有必要证据能从正式8路相机或允许动作获得；human-only不计 | 实际agent frame/camera matrix/光学ray检查 |
| G7 Evaluator Fidelity | 原生世界状态验证，目标谓词不依赖Agent自报完成 | native scorer与物理结果审计 |
| G8 Task Independent Mechanism | 同scene不同任务可有控制条件但不要宣传成独立scene | 多变量机制/对照与数据集family ID |
| G9 Cost and Scale | 启动时间、wall time、sim time、Agent decisions、路径与显存 | episode receipts，记录median/P90 |
| G10 Versioned Provenance | Source, modified Scene, Task Definition, seeds, scorer, observation protocol版本分离 | manifest/provenance |

Do **not** substitute scene-label footprint for navigable area, Euclidean endpoints for a feasible path, or USD prim count for relevant movable objects. Do not treat scenario capture or initialization as a completed robot trajectory. The current audit has native-reported label areas for only four navigation tasks; all local usable areas, full background counts, and complete grasp/transport outcomes remain pending.

## Execution and release order

| Phase | Output | Acceptance criterion |
|---|---|---|
| 1. Freeze Taxonomy | taxonomy_v1.yaml + all61归类 + 能力多标签 | 所有61已有候选恰好一个primary；未来新增只填槽 |
| 2. Reuse CoreV2 | 48项CoreV2重用分级与变更清单 | 保留原源码路径/ID/ESI provenance；Evaluator可测 |
| 3. Pick Scenes for Gaps | shortlist按优先级；G0–G10门槛 | 不存在合法物理解时不构造任务；Unknown不得当PASS |
| 4. Build & Verify Remaining Tasks | Mirror8、Size5、Recovery1及针对性CoreV2修复 | 真实合法成功路径、独立评分、正式8路输入、预算收据 |
| 5. Publish V1 | v1.0 task registry, tasks, cost report, three baselines | 数量如实报告，版本冻结；v2新增不改旧榜 |

## Prioritize first implementation examples

1. **GF-2 / D1-1** — `Beechwood_0_garden/utility_room_0` from T26 as the first local width-limited insertion candidate. Validity requires measuring the full allowed orientation set and gripper withdrawal.
2. **GG-3 / R-C1 + R-A2** — real mirror-positive mapping plus direct-view negative control. The original visual-only props, source free camera, and low initial positions are not eligible as real manipulation evidence.
3. **TM-2 / T40–41,T45–48** — preserve one physical layout but construct paired histories requiring different actions at matching current observations.
4. **AP-3 / T42–44** — reuse the native inclined apparatus, enforce counterfactual winner reversal, then test trials-to-decision.
5. **AP-4 / recovery** — use T26 arrangement in `Beechwood_0_garden/utility_room_0` for an exogenous, physically reversible disturbance and verified restoration.

## Versioning policy

- Taxonomy IDs (`AP-1` etc.) are stable in v1 and v2. A new task receives a new canonical `task_id` and records `primary_subcategory`, `capability_tags`, `scene_family`, `source_provenance`, and `scorer_version`.
- V1.0 leaderboard and task contracts are frozen. Replacements that change the scientific task are new variants, not silent overwrites.
- Data split on scene/template family, not solely by task filename. A second operation in one room is useful controlled diagnosis, but not an independent environment.
- Publish core diagnostic tasks and optional controls/stress separately; do not hard-require exactly 60 or 61 items.

## Provenance

- GitHub pinned CoreV2: https://github.com/XuanCheg/Show-Harness-Explore/tree/33bada4bb278471b7a925cd8ddfa3d46a19f52fb
- Source candidate audit: prior `Explore_to_Achieve_Audit.xlsx` and `World_Interactive_Agency_V1_Scene_Audit.xlsx` (2026-10-22).
- Existing source audit / simulator review: `final_report.md` (2026-10-16).
- RoboQuest arXiv:2610.10388 used for distinction between missing-information exploration and broader knowledge-to-action evaluation.

**Not executed by this agent:** no OmniGibson native load, GPU rollout, grasp success trial, reflection render validation, route planning, or numerical experiment. All recommendations are static/provenance analyses; source quantitative observations come from prior saved audits, not this review.
