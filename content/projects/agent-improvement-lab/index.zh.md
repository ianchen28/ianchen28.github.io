---
title: "Agent Improvement Lab"
description: "一个独立的公开研究项目：探索工具型 Agent 的可信评估、失败诊断与受控改进。"
status: in_progress
status_label: "进行中：基础工具已完成，首轮候选已复核"
progress_page: progress
progress_label: "用直白的话看开发进度"
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
      description: "小规模提示对照；保留负结果"
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
  title: "Evaluation Harness v0.1 / 已实现的执行与证据流程"
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

任务失败为分析提供具体线索。工具型 Agent 的失败可能来自决策、
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

Agent 和用户模拟器在目前的实际运行中分别使用 DeepSeek，维护独立的提示词与状态。
截至 2026 年 11 月 11 日，已完成的八个 Retail 基线任务均无需要调用自然语言
Judge 的断言，评分采用原生程序检查。系统保留显式标识的 DeepSeek NL 评分变体，
其与原始 GPT-4.1 评分的等价性尚未验证。
比较时记录任务集、模型、评分器、实验协议及证据覆盖率，并分别报告业务结果和流程行为。

{{< harness-flow >}}

## 路线图与当前工作

1. **Evaluation Harness — 已实现。** 已发布 `v0.1-offline`，具备配置检查、
   评分前记录、错误区分、离线报告和实际运行的费用保护。
2. **失败诊断 — 已开展。** 已审核需求变更、写入对象和工具调用纪律，
   将具体行为与任务分数分开记录。
3. **技能与提示实验 — 首个候选已暂停。** 两版单工具提示及其位置调整没有
   达到预定的稳定改善条件，完整保留了负结果和一次未复现的局部信号。
4. **完整收益与泛化 — 待验证。** 同期完整交互对照、预留任务验证、
   Rubric 优化和后训练均尚未开展。

## 实验摘要

截至 **2026 年 11 月 11 日**，完成八个不同 Retail 任务的基线、
一次额外完整 Skill 条件运行和五组共 38 条单步诊断回复。
这些小规模开发实验尚未建立稳定的 Skill 改善或新任务泛化证据。

从模拟顾客和客服的故事开始，逐轮解释见
[开发进度：给 AI 客服一张操作卡]({{< relref "/progress" >}})。

本页和进度页是人工整理的摘要。原始轨迹、私有实验产物与凭据保持在网站之外，
Python 研究环境保持独立。
