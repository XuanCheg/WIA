# World-Interactive Agency V1 — 问题索引（供 Codex 独立复查）

**整理日期：** 2026-10-23  
**审核依据：** 原始 ZIP 中的核心故事、Taxonomy YAML、实施方案、3 份审计工作簿；随包的 61 项审核 CSV / 独立审核报告；部分 CoreV2 任务 YAML 与 Converter/Evaluator 源码。  
**代码参照版本：** `XuanCheg/Show-Harness-Explore` commit `33bada4bb278471b7a925cd8ddfa3d46a19f52fb`（原审核取样基线；若检查最新 main，请记录 SHA 和差异）。  
**重要边界：** 本次没有新运行 OmniGibson 原生仿真、8 路真实相机验证或 GPU Agent 实验。以下“风险”不是已确认的运行故障；Codex 应独立确认，不要因为报告已指出就直接视为事实。

## 0. 复查原则

1. **先证明科学机制有必要，再讨论任务数量。** V1 ≈60 是候选规模，不是数量准入要求。
2. **尽量复用 CoreV2。** 保留原任务、场景、初始化、Converter、Evaluator、来源对应关系；能以参数、对照、任务目标或评分小修解决的，不重写整个任务。
3. **场景成本必须服务于科学问题。** 多房间、大面积、更多操作步骤不自动代表认知任务更难。
4. **区分证据等级。** `DOCUMENT`：可从文档、映射表直接核对；`STATIC`：代码/设计风险，须独立核对；`NATIVE`：只有仿真/真实运行可判定；`EXPERIMENT`：需要受控 Agent 对照才能论证科学机制。
5. **不要因为任务处于 pending 就自动判无效。** 静态设计通过、原生初始化通过、机器人合法动作轨迹通过、Agent 科学对照通过，是四个不同阶段。
6. **保留合法 no-op。** 已完成目标时不行动可能是正确决策；但应作为正确停止/对照，不要统计为已完成复杂操作。

## 1. 优先问题台账

| 编号 | 优先级 | 证据等级 | 范围 | 具体疑点与 Codex 应查事项 | 预期处理/验收标准 |
|---|---|---|---|---|---|
| WIA-01 | P0 | DOCUMENT | Taxonomy、两个审计工作簿 | `All_61_Map` 家族分布 **AP/NT/GF/SR/GG/TM = 10/8/9/12/14/8**，`Scene_Audit/Family_Stats` 却是 **10/8/10/14/11/8**；核查版本及3项净归属偏差。 | 选定一个权威任务主归属表，统计从同一来源生成；对每个重新归属保留迁移记录，不擅自改原始文件。 |
| WIA-02 | P0 | DOCUMENT | 任务规模/发布陈述 | 61 候选 = CoreV2保留48（R0=18、R1=20、R2=10） + 新设计13（N1=8、N2=5）；统计值≠物理 PASS 数。 | 检查所有 README、图表、实验计划的状态措辞，不把 R0、N1、`status: ready` 当已完成完整参考轨迹。 |
| WIA-03 | P0 | STATIC+EXPERIMENT | AP-3：T42–T44 | 斜坡试验的候选赢家可能不随条件变化，或 0.15s 容差产生并列。强制完成试验不等于「利用试验决定行动」。核对 `inclined_comparison.py` 的 trial 判定、计时、赢家容差及三套物理配置。 | 至少构造两种经过原生验证、唯一最佳物体不同的物理条件；加入 no-test / incorrect-test 对照；保留已有试验装置与 Evaluator。 |
| WIA-04 | P0 | STATIC+EXPERIMENT | TM-2：T40–T41、T45–T48 | 固定 Phase1 对象位置、事件类型和公开 spare 清单可能使历史记忆不必要；某些答案可由最终快照猜中。核对 Agent 可见输入而非仅 private bindings。 | 构造 `current observation 相同，history 不同，正确物理动作不同` 的配对；无历史基线应劣于有历史版本；T41 no-change 合法。 |
| WIA-05 | P0 | STATIC+NATIVE+EXPERIMENT | Reflection：R-C1–3、R-A1–2、R-S1–3 | 原图像任务可能使用 visual-only 镜面或特殊 camera；真实 R1 8路 mounted view、反射材质、可抓取/避碰、目标身份/空间关系是否需要镜子均不确定。 | 检查真实镜像光学和 mounted camera；镜面信息不可被直接视角、绝对位置或材质纹理绕过；至少一组镜像正例/直视负例与可完成的实体动作。 |
| WIA-06 | P0 | STATIC+NATIVE | GG-4：T15–T17 | Material Transparency 任务的瓶颈/开口是否允许目标与夹爪/机器人手部通过；透明材质可能根本不影响操作决策。 | 对最窄净通路、姿态扫掠体、抓放及撤手作原生核验；若科学机制不足，优先保留素材并通过透明/不透明且尺寸匹配的受控条件重构。 |
| WIA-07 | P0 | STATIC+EXPERIMENT | T09–T11 | T11 的选项包含查询物体的名称，存在词面泄露；T09–T10 可能退化为三选一 QA 而没有真实到访。 | 去除文本/位置捷径，验证真实语义目标到访是主结果；原问答可保留为辅助指标。 |
| WIA-08 | P0 | DOCUMENT+STATIC | AP-4 | 19 子类中 Closed-loop Recovery 是空缺，没有可支持该机制的正式候选。 | 以现有可执行任务加一个外部受控扰动，比较有无反馈修正，保持 Controller/场景资产复用；不要用固定重试算恢复。 |
| WIA-09 | P0 | NATIVE | 所有拟发布任务 | 静态工作簿不等于 `load → settle → observe → legal actions → release → withdrawal → final state` 已成立；需检查真实8路相机、能否抓取及 Evaluator。 | 每任务保留「证据指针 + PASS/FAIL/INCONCLUSIVE + 失败阶段」，没有原生证据一律不宣称已认证。 |
| WIA-10 | P1 | STATIC+NATIVE | NT：T04–T08 | T07 的6房顺序已写进 prompt，更接近路线执行对照；T08 源参考路线 43.831m，不能用房数公平比较；T07储物间标签区域约1.831㎡。 | 用 R1可行轨迹/真实开门与进入/决策预算确认；长程压力与认知机制分开；T07不宣称自主顺序推理。 |
| WIA-11 | P1 | STATIC+NATIVE | AP：T01–T03、T18–T21 | T01 指定顺序≠自主推断；T02 球体滚动；T03 梨掉入锅 vs 上层支撑；T18–20 细书/圆碗/罐操作难；T21 三张约45cm台面产生无关转运。 | 保留 Task/Converter，确认刚体真实支撑与稳定终态、可抓取与退手；T21 优先改一个共同工作站。 |
| WIA-12 | P1 | STATIC+NATIVE | GF：T22–T25、D1-1–3、D2-1–2 | Rigid Containment 已有较完整人工审核，不应重做；但部分任务允许逐个放入、移出，可能被误解为同时容纳。D1 不能只凭 AABB 和默认朝向判断能否穿过，D2 容易退化为定位置排序。 | 严格沿用原 `allow_remove_after_verification` 与独立插入的语义；D1 验证合法旋转/机器人握持与撤手，D2 消除位置→答案捷径。 |
| WIA-13 | P1 | STATIC+NATIVE | SR：T26–T34 | T26–T29 应从非目标初态通过真实物体移动达成几何关系；T30/T32 可 no-op 但不能冒充重排；T31/34 大物体/走廊可达风险；T33 接触不可只看 AABB。 | 原生 contact / stable release / 当前状态判分准确；将已满足目标单列正确停止对照；核对完整动作链。 |
| WIA-14 | P1 | STATIC+NATIVE | GG：T35–T37、T49–T51 | T35 目标可能可见、无同类干扰，测的主要是收集而非主动搜索；T37 多件收集累积低层失败；T49 布料揭示只有单目标，T50 才有身份选择；T51 操作负担增加。 | 分开报告身份决策/搜索发现与抓放完成；T50 必须出现真实手—布接触及先揭布后判定目标的事件。 |
| WIA-15 | P1 | STATIC+EXPERIMENT | TM-1：T38–T39 | 自定首次巡视顺序是否可以被固定路径、物体方位、非历史线索推断？ | 对同一当前状态构造不同真实观察顺序，并测试无历史/扰乱历史；保留合法到访证据。 |
| WIA-16 | P1 | STATIC | Taxonomy：AP-1 / NT-2 / GF-1 vs GF-2 / GG-1 vs GG-2 / GG-3 vs SR-3 | 若子类只是预设动作或素材类别，而非可以改变决策的机制，十九子类的独立性不足；不同大类存在边界模糊。 | 逐子类填写「必要信息→会改变哪种物理决策→反事实对照→和近邻子类差异」；不重做源任务。 |
| WIA-17 | P1 | DOCUMENT+STATIC | V1统计、实验设计 | 同一 `house_single_floor`、`Beechwood_0_garden` 有许多变体；不能把同场景多题视作独立环境；场景面积是标签面积不是导航净面积。 | 增加 `scene_family_id`、source_seed、有效 nav/task workspace、物体与混淆项数量；场景分组报告和预算统计。 |
| WIA-18 | P1 | STATIC+EXPERIMENT | 所有能力归因 | 总任务成功率会混合视觉判断、计划、抓取、路径、物理接触；只跑 Agent 或只跑 oracle 不能证明 Gap 假设。 | 同场景同预算比较有/无关键信息、高层 oracle、合法执行技能成功率、阶段化得分；不把失手归为认知失败。 |
| WIA-19 | P2 | NATIVE | 新设计五项 D1/D2 | 5 个尺寸设计目前没有完整 native scene 身份，构建候选时可能由于空间/夹具约束根本不可执行。 | 按先 scene qualification 后 task authoring 的顺序；优先 D1-1，再 D1-2/D2-1，其余适度延后。 |
| WIA-20 | P2 | EXPERIMENT | 论文对比 RoboQuest | RoboQuest 已研究探索、实验、记忆、执行隔离等；不能声称率先提出这些能力。 | 用完全可见但高难物理决策 + matched counterfactual + 受控 knowledge-to-action 结果证明区别；结论以实验为依据。 |

## 2. Codex 独立审查的输出建议

逐个问题返回：

- **结论**：`CONFIRMED / PARTIALLY_CONFIRMED / NOT_REPRODUCED / BLOCKED / NEEDS_NATIVE_RUN`。
- **证据**：确切文件路径与行号、对应 task ID / scene ID；若依据实验必须注明命令、配置、seed、输出路径和观察结果。
- **归因**：研究设计错误 / 文档失配 / Converter 或 Evaluator bug / 原生初始化问题 / 低层动作限制 / 证据不足。
- **最小改动**：`retain / local_fix / goal_or_metric_revision / controlled_variant / defer_V2`；注明能保留的原资产。
- **发布影响**：是否影响 V1 资格、是否仅影响科学标签，是否可做对照任务。
- **验证证据**：没有真正跑原生环境时标记 `NOT_RUN`，不要推断 PASS。

## 3. 推荐复查顺序

1. 先对 **WIA-01～02** 做数据完整性核对，明确基线版本和状态口径。
2. 对 **WIA-03～08** 做高风险科学机制的源码核对，挑最少量反事实要求。
3. 对 **WIA-09～15** 做分任务执行可行性审查，保留人工已核的 Rigid 与既有场景。
4. 最后检查分类、计费、场景独立性和论文结论 **WIA-16～20**。

> 61项逐任务结论详见 `World_Interactive_Agency_V1_61_Task_Review.csv`；背景分析和实验矩阵详见 `World_Interactive_Agency_V1_Independent_Audit.md`。两份材料是待独立复核的研究意见，不是要求 Codex 直接照单修改。
