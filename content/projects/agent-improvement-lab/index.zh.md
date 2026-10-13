---
title: "Agent Improvement Lab"
description: "一个独立的公开研究项目：探索工具型 Agent 的可信评估、失败诊断与受控改进。"
status: in_progress
status_label: "进行中：Evaluation Harness v0.1 设计"
source_url: "https://github.com/ianchen28/agent-improvement-lab"
source_label: "源代码仓库"
translationKey: agent-improvement-lab
outputs: ["HTML"]
loop:
  title: "Agent 改进闭环"
  development: "开发评估 / 迭代研究"
  stages:
    - title: "任务环境"
      description: "开发任务、策略、工具与状态"
    - title: "Agent"
      description: "模型、提示、技能与执行框架"
    - title: "执行轨迹"
      description: "消息、工具调用与结果"
    - title: "开发评估器"
      description: "明确的标准与评分器身份"
    - title: "失败分析"
      description: "证据与人工提出的假设"
    - title: "优化"
      description: "受控改动；后续实验"
    - title: "改进后的 Agent"
      description: "候选版本 / 研究目标"
  feedback: "改进后的 Agent 回到开发评估"
  feedback_description: "让更新后的候选版本重新进入完整的执行与评估流程。"
  heldout: "独立通道 / 冻结的留出评估"
  heldout_stages:
    - "冻结候选版本与协议"
    - "未暴露的留出任务"
    - "固定评估器"
    - "泛化能力检验"
  heldout_description: "检验前冻结模型、提示、技能、框架、任务划分、Rubric、评分模型、种子、试验次数、指标和预算。留出结果与轨迹不参与提示或技能选择，也不反馈到优化过程。需要审计任务暴露历史，原生 train/test 标签本身并不代表干净的留出集。"
  caveat: "“改进后的 Agent”是待检验的候选版本，不代表已经测得改进。如果留出反馈影响后续开发，应披露暴露情况，并在下一次泛化声明前建立新的未暴露留出集。"
harness:
  title: "Evaluation Harness v0.1 / 目标设计"
  stages:
    - title: "配置与安全预检"
      description: "明确任务、试验次数、模型角色与工作量上限"
    - title: "本地 Python 适配层 / 原生 tau2 交互"
      description: "Agent 与用户模拟器 / Retail 工具与环境"
    - title: "评分前轨迹检查点"
      description: "在评估器运行前保存执行证据"
    - title: "原生评估器 + 明确标识的 DeepSeek NL 评分变体"
      description: "保留原生任务标准与奖励组合"
    - title: "指标、报告与人工标注"
      description: "覆盖率、错误区分与基于证据的诊断"
---

## 动机与研究问题

任务失败是分析的起点，而不是解释。工具型 Agent 的失败可能来自决策、
策略理解、工具交互、运行时问题，或者不确定的评分。

- **可信测量：** 哪些检查与覆盖率指标能够支持可靠的比较？
- **证据驱动的诊断：** 哪些轨迹证据支持失败假设，哪些归因仍不确定？
- **受控改进：** 提示、技能或执行框架的改动能否迁移到开发任务之外？

## 研究架构

{{< agent-loop >}}

## 评估方法

从 **tau2 Retail** 及其原生任务标准、奖励组合开始，在评分前保存交互证据，
区分 Agent、用户模拟器和条件触发的自然语言评分角色，将任务结果与运行、
评分错误分开报告。

**DeepSeek NL 评分模型是明确标识的变体**，不等同于原始 GPT-4.1 评分，也不
宣称两者已经验证等价。OpenRouter/DeepSeek 的合成连通性检查只验证调用路径，
不代表 Retail 成功率或评分质量。比较时必须说明任务集、模型、评分器、
实验协议及证据覆盖率。

{{< harness-flow >}}

## 路线图与当前工作

1. **Evaluation Harness — 设计进行中。** 配置安全、可复现证据、透明评分与
   报告。已有集成与基线包装脚本是前置基础；新实现与离线测试仍为计划阶段。
2. **失败分类 — 计划中。** 将人工审核的类别与假设关联到具体轨迹证据。
3. **技能与提示优化 — 后续研究。** 先进行受控的开发实验，再使用独立、冻结
   的留出评估检验泛化。
4. **后训练 — 后续研究。** 在建立评估基础后探索从审核过的轨迹中学习。
   AutoDL 是可选的计划中算力，并非已有集成。

## 实验摘要

**实验 001：tau2 Retail 基线 — 计划中。** 建立小规模、可检查的基线，固定
任务、模型、评分器来源与覆盖率口径。目前没有基准分数或已验证的改进声明。

摘要将在发布前人工审核。原始轨迹、私有实验产物与凭据不会自动导出到网站。
Python 研究环境保持独立。
