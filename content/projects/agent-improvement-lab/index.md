---
title: "Agent Improvement Lab"
description: "An independent public research project exploring reliable evaluation, failure diagnosis and controlled improvements for tool-using agents."
status: in_progress
status_label: "In progress: Evaluation Harness v0.1 design"
source_url: "https://github.com/ianchen28/agent-improvement-lab"
source_label: "Source repository"
translationKey: agent-improvement-lab
outputs: ["HTML"]
loop:
  title: "Agent Improvement Loop"
  development: "Development Evaluation / iterative research"
  stages:
    - title: "Task Environment"
      description: "Development tasks, policy, tools and state"
    - title: "Agent"
      description: "Model, prompts, skills and scaffold"
    - title: "Trajectory"
      description: "Messages, tool calls and outcomes"
    - title: "Development Evaluator"
      description: "Explicit criteria and grader identity"
    - title: "Failure Analysis"
      description: "Evidence and human hypotheses"
    - title: "Optimization"
      description: "Controlled changes; future experiments"
    - title: "Improved Agent"
      description: "Candidate / research objective"
  feedback: "Improved Agent returns to Development Evaluation"
  feedback_description: "Re-run the updated candidate through the full execution and evaluation pipeline."
  heldout: "Separate lane / Frozen Held-out Evaluation"
  heldout_stages:
    - "Freeze candidate and protocol"
    - "Untouched held-out tasks"
    - "Fixed evaluator"
    - "Generalization assessment"
  heldout_description: "Freeze model, prompts, skills, scaffold, split, rubric, judge, seeds, trials, metrics and budget before assessment. Held-out outcomes and traces do not feed prompt/skill selection or optimization. Audit prior task exposure; native train/test labels alone do not establish an untouched holdout."
  caveat: '"Improved Agent" names a candidate to test, not an observed improvement. If held-out feedback informs later development, disclose exposure and establish a fresh holdout before another generalization claim.'
harness:
  title: "Evaluation Harness v0.1 / target design"
  stages:
    - title: "Configuration and safety preflight"
      description: "Explicit tasks, trials, model roles and workload limits"
    - title: "Local Python adapter / native tau2 interaction"
      description: "Agent and User Simulator / Retail tools and environment"
    - title: "Pre-grading trajectory checkpoint"
      description: "Preserve evidence before the evaluator runs"
    - title: "Native evaluator + explicit DeepSeek NL judge variant"
      description: "Native task criteria and reward combination"
    - title: "Metrics, report and human annotation"
      description: "Coverage, error separation and evidence-led diagnosis"
---

## Motivation and research questions

A failed task is a starting point, not an explanation. Tool-using agents can
fail through decisions, policy misunderstandings, tool interactions, runtime
problems or uncertain grading.

- **Trustworthy measurement:** which checks and coverage measures support
  defensible comparisons?
- **Evidence-led diagnosis:** which trajectory evidence supports a failure
  hypothesis, and where is attribution still uncertain?
- **Controlled improvement:** do prompt, skill or scaffold changes transfer
  beyond development tasks?

## Research architecture

{{< agent-loop >}}

## Evaluation methodology

Start with **tau2 Retail** and its native task criteria and reward combination.
Preserve interaction evidence before grading, distinguish agent, simulated
user and conditional judge roles, and separate task outcomes from execution
or grading errors.

The **DeepSeek NL judge is an explicitly identified variant**, not original
GPT-4.1 scoring or a verified equivalent. Synthetic OpenRouter/DeepSeek
connectivity checks establish a call path, not Retail success or grader quality.
Comparisons must identify the task set, model, grader, protocol and coverage.

{{< harness-flow >}}

## Roadmap and current work

1. **Evaluation Harness — design in progress.** Configuration safety,
   reproducible evidence, transparent grading and reports. Integration and
   the existing baseline wrapper establish prerequisites; new implementation
   and offline tests remain planned.
2. **Failure taxonomy — planned.** Human-reviewed categories and hypotheses
   tied to concrete trajectory evidence.
3. **Skill and prompt optimization — future research.** Controlled development
   experiments, followed by separate frozen held-out assessment.
4. **Post-training — future research.** Explore learning from reviewed
   trajectories after establishing evaluation. AutoDL is optional planned
   compute, not an existing integration.

## Experiment summaries

**Experiment 001: tau2 Retail baseline — planned.** Establish a small,
inspectable baseline with fixed task/model/grader provenance and explicit
coverage. No benchmark scores or improvement claims are available yet.

Summaries will be manually reviewed before publication. Raw trajectories,
private experiment artifacts and credentials are never automatically exported
to this website. The Python research environment remains independent.
