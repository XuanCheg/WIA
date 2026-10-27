# World-Interactive Agency V1 — Codex 复查入口

这是在 **2026-10-22 原始研究压缩包**基础上追加独立审核材料后的版本。**原包7项文件保持原内容及相对路径**，没有回写原 Taxonomy、工作簿、实施计划，也没有修改 GitHub 仓库。

## 首先看新增的三个文件

1. `06_Independent_Review/Issues_for_Codex_Recheck.md` —— **20组关键问题/疑点**、P0/P1/P2优先级、证据层级、具体复查动作、验收标准。
2. `06_Independent_Review/World_Interactive_Agency_V1_Independent_Audit.md` —— 科学问题、19子类、任务组、场景指标、对照实验的完整独立审核。
3. `06_Independent_Review/World_Interactive_Agency_V1_61_Task_Review.csv` —— **61项任务逐项**问题、处理建议、最小修复与源码URL，可用Excel/脚本筛选。

## 原始资料（保持不变）

- `README.md`：原始文件的阅读顺序。
- `01_Research_Story/World_Interactive_Agency_Core_Story.md`：核心论文故事。
- `02_Taxonomy/World_Interactive_Agency_Taxonomy_v1.yaml`：6家族/19子类设计。
- `03_Implementation/World_Interactive_Agency_V1_Reorganization.md`：实施思路。
- `04_Audits/*.xlsx`、`05_Reference/*.xlsx`：原有61项映射、场景审计及历史基线。

## 给 Codex 的审核边界

**用户的三项硬原则：** 研究问题优先于任务数；尽可能复用已有 CoreV2；场景服务于科学问题、不引入不相关操作难度。

- 请独立检查 `XuanCheg/Show-Harness-Explore` 的任务 YAML、Converter、Evaluator 和场景配置，**不要将独立审核报告当作已证实事实**；代码基线参见问题索引。
- 优先核实 P0 项，特别是分类统计不一致、T42–44 实验的辨识力、T40–48 记忆必要性、Reflection镜像唯一证据、T15–17瓶口操作可行性、T11泄露、AP-4 缺口。
- **静态审查与原生物理验证必须分开写**。若没有运行 OmniGibson、正式8路R1摄像头和合法物理动作轨迹，标 `NOT_RUN/INCONCLUSIVE`，而非PASS。
- 尽量指出 **最小修复与可以保留的工程资产**；只有无法形成科研上有效的实验机制时才建议调整目标/评分或暂列V2。
- 如果当前任务仅要求复查，先提交发现、证据与建议，不要在未经要求的情况下批量重构任务。

**期望输出：** 逐条复核 WIA-01～20 的结论、具体源码路径/行号、哪项建议应接受或驳回、以及少量最高优先级可执行变更和需要原生实验的列表；**任何候选数量都不能代替正式通过数**。
