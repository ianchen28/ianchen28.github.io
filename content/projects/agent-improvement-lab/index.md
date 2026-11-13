---
title: "Agent Improvement Lab"
description: "An independent public research project exploring reliable evaluation, failure diagnosis and controlled improvements for tool-using agents."
status: in_progress
status_label: "In progress: evaluation tools built, first candidate reviewed"
progress_page: progress
progress_label: "Read the plain-language progress account"
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
      description: "Small prompt comparisons; negative results retained"
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
  title: "Evaluation Harness v0.1 / implemented execution and evidence flow"
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

A failed task provides a concrete starting point for analysis. Tool-using agents can
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

The agent and user simulator currently use separate DeepSeek conversations.
Through 11 November 2026, the eight completed Retail baseline tasks required
no natural-language judge calls; native programmatic checks supplied their scores.
The configured DeepSeek NL judge remains an explicitly identified variant whose
equivalence to original GPT-4.1 scoring has not been established.
Comparisons record the task set, model, grader, protocol and coverage, with business
outcomes and procedural behavior reported separately.

{{< harness-flow >}}

## Roadmap and current work

1. **Evaluation Harness — implemented.** Published `v0.1-offline`, with strict
   configuration, pre-grading records, error separation, offline reports and
   spending protection for approved live runs.
2. **Failure diagnosis — underway.** Reviewed changed customer requests, write
   targets and tool-call discipline, keeping specific behaviors separate from scores.
3. **Skill experiments — first candidate paused.** Two single-tool prompt versions
   and location changes did not meet the predefined stability criteria. Negative
   results and a one-off improvement that did not repeat remain in the record.
4. **Full-task gains and generalization — unverified.** Contemporaneous complete
   interactions, reserved-task validation, rubric optimization and post-training
   have not yet been performed.

## Experiment summaries

Through **11 November 2026**, work includes eight different Retail baseline tasks,
one additional full Skill-condition conversation and five groups totalling
38 single-response diagnostic samples. These small development experiments have
not established stable Skill improvement or transfer to new tasks.

The [plain-language progress account]({{< relref "/progress" >}})
explains each stage through the simulated customer-service example.

These pages are manually prepared summaries. Raw trajectories, private artifacts
and credentials remain outside this website. The Python research environment
remains independent.
