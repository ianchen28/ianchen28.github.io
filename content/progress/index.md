---
title: "Can an instruction card improve an AI support agent?"
linkTitle: "Development progress, explained simply"
description: "A beginner-friendly account of our simulated shop, the experiments we ran, and why we have paused a candidate that did not produce a repeatable improvement."
type: progress
translationKey: agent-improvement-lab-progress
date: 2026-11-13
lastmod: 2026-11-13
outputs: ["HTML"]
comments: false
eyebrow: "Agent Improvement Lab · A guide for first-time readers"
evidence_date: "2026-11-11"
evidence_label: "Experiments through"
updated_label: "Page updated"
status_label: "Evaluation tools built; current instruction-card candidate paused"
snapshot_heading: "The one-minute version"
snapshot:
  - title: "The goal"
    text: "Learn from a specific AI mistake, write a short instruction card, and check whether it helps the agent do better next time."
  - title: "What we have built"
    text: "An experiment setup that records conversations, software actions, scores and charges, with several rounds of real model tests completed."
  - title: "The current finding"
    text: "Two instruction-card versions did not produce a stable single-call improvement. We preserved the results and paused this candidate."
reading_heading: "Read by question"
reading_links:
  - {label: "What are we trying to improve?", id: goal}
  - {label: "Who plays the customer and the shop?", id: setup}
  - {label: "What problems did we find?", id: finding}
  - {label: "How did we test the instruction cards?", id: experiments}
  - {label: "How much have we done?", id: counts}
  - {label: "What can we conclude?", id: conclusions}
  - {label: "What happens next?", id: next}
project_page: projects/agent-improvement-lab
project_link_label: "Project overview"
note_heading: "How this page uses the evidence"
note: "This is a manually prepared summary of saved results. Account details, order records and private raw logs stay outside the website. Failures, untriggered cases and the improvement that did not repeat remain part of the account."
---

## What are we trying to improve? {#goal}

Imagine buying a helmet online and asking customer support to exchange it for a medium size.
The representative needs to identify you, find the order, check availability, explain the change,
get your confirmation and submit the exchange. This is a practical job with several decisions
and an outcome we can inspect.

**We want to understand how an AI can get better at work like this.**
The AI can talk and issue software commands such as “look up this order” or “request an exchange.”
That combination is what we mean by an **agent**. On this page, think of it as an AI support representative.

The research idea is straightforward:

1. Let the representative try a task and record what happens.
2. Identify one concrete mistake or rule violation.
3. Write the desired procedure on a short instruction card.
4. Test again under comparable conditions, checking both the behavior and the task outcome.
5. If it helps on the development example, freeze the card and test it on fresh problems.

We call this additional instruction text a **Skill**.
The representative reads the extra guidance before working; the model's internal parameters
remain unchanged.

**People currently review the mistakes, write the cards, design comparisons and decide when
to stop.** A system that independently finds problems and writes effective improvements
remains a future research goal.

Our eventual goal is repeatable improvement that also helps on tasks beyond the example used to develop the card.
That goal remains unverified.

## Who plays the customer and the shop? {#setup}

We use **tau2 Retail**, an existing test environment with fictional customers, products, orders
and support policies. Exchanges and returns affect a test database. Real customers' shopping
orders are not involved.

### Four parts take different roles

- **Simulated customer:** a separate AI conversation receives a scenario and the information the customer is allowed to know.
- **AI representative:** reads the support policy, available tools and conversation, then decides what to ask or do.
- **Simulated shop:** software and a test database execute commands and return their results.
- **Evaluator:** the test environment checks the resulting state, such as whether the intended order was changed correctly.

The customer and representative use the same DeepSeek model type with separate prompts and
conversation states. Neither receives the evaluator's reference answers. The tasks described
here required no additional natural-language judge calls; native checks, including database
checks, supplied their task scores.

A typical interaction looks like this:

> Customer requests an exchange → representative asks for identity information →
> software returns account details → representative explains the proposal →
> customer confirms → representative submits an action → evaluator checks the result.

We save the conversation, commands and tool results so that a failure can be traced to a particular step.

### The first thing we built

The first evaluation tools are implemented and published as `v0.1-offline`.
They freeze settings, preserve conversations before grading, generate reports and distinguish
task failures from execution or grading problems.

We also added explicit approval for paid runs, request and spending limits, and per-request
billing checks. These provide a dependable record for comparisons. Building that record was
a substantial part of the work.

## What problems did we find? {#finding}

### First, check the complete path

We started with a small shopping task. The two AI roles conversed, software executed the
requested actions, the native evaluator scored the outcome, and billing was reconciled.

**The task received full native reward, and the complete evidence was saved.**
Earlier network rate limiting and a local pre-send guard failure remain separate startup records.

### Next, look for actions based on an outdated customer request

A customer might initially ask to return two items and then confirm only one.
We wanted to see whether the representative would submit the customer's latest request.

In the three tasks that reached this decision, the representative followed the updated request.
**We did not observe the intended business mistake in these samples.**
That small set does not establish reliability on every similar problem.

A fourth task stopped at identity verification. Its scenario supplied an account string and ZIP
code, without separately spelling out the name. The simulated customer was also told to treat
unspecified information as unknown. It repeatedly said it could not provide the requested name.

That result highlighted a limitation in the test scenario and the simulator's interpretation.
We kept the zero task score and labelled the behavior we originally wanted to inspect as
“not reached.”

### Then, look for changing the wrong order or item

We selected three tasks requiring order or product disambiguation: choosing the right purchased
item, keeping a reference order separate from the target order, or selecting an available replacement.

All three used the correct order and item arguments and received full native reward.
There was still no qualifying wrong-write example.

However, reading the conversations revealed a different problem: the representative sometimes
put several tool calls in a single message.

### What does “several calls in one message” mean?

For four orders, the AI might produce:

> One assistant message: look up order A; look up B; look up C; look up D.

The native shop policy asks for at most one call at a time. The desired pattern is:

> Look up A → wait for the real result → decide whether to look up B.

We count the commands packaged in **one assistant message**.
The receiving software may still execute that group sequentially; this measurement makes
no claim about simultaneous computer execution.

The selected example completed its business changes and received full native reward,
while also containing two grouped-call messages.
**Task completion and adherence to the operating procedure became separate measurements.**

## How did we test the instruction cards? {#experiments}

### Version 1: a direct reminder

We appended a two-sentence Chinese instruction asking the representative to issue at most one
tool call per assistant message, wait for the result, and then choose its next action.

We kept the other settings fixed and ran one complete conversation. The representative still
decided how to call tools; no executor forced it to produce single calls.

**Grouped-call messages increased from two in the historical baseline to three.**
Native reward stayed at 1.0. After the business changes, the simulated customer raised a
return-label issue and requested a human transfer, so our predefined normal-ending condition
also failed.

The joint check did not pass. A single historical comparison cannot establish that the card
caused the extra grouped message or the transfer ending.

### Pause at the same moment and inspect only the next decision

A changed representative reply can change the customer's next response.
We therefore froze the history immediately before a decision and asked the model for one
next reply with and without the card.

Think of this as placing the same exam paper at the same question and comparing only the next step.
We checked actual request bodies to establish that the paired inputs differed only in the card.

Six replies covered three decision points. Each number below is the number of commands in
one assistant reply:

| Decision point | Without the card | With version 1 |
| --- | ---: | ---: |
| Which order to inspect next | 4 | 4 |
| Which product catalogue to inspect next | 3 | 3 |
| Which confirmed change to submit first | 1 | 1 |

The two query points did not improve. At the write point, the fresh baseline already used
one call, leaving no observed call-count gap. Returned tools were not executed in these probes.

### Version 2: a procedure and a generic example

We expanded the card into:

> Choose one useful next step → issue one native tool call → end this message →
> wait for a real tool result → decide what to do next.

A generic example showed querying object A, receiving its result, and only then querying B.
The card contained no task-specific order IDs or reference answers.

Four new replies tested the two query points. **The counts remained four versus four for
orders and three versus three for product catalogues.** The extra procedure and example
did not achieve the target in those contexts.

### Check the simplest instruction controls

We also tested the API route with four small synthetic requests:

- Return one exact phrase, once without tools and once with a tool available.
- Handle a simple request that calls for a lookup tool.
- Repeat the lookup request with the API's tool-use switch set to `none`.

**All four checks passed.** The exact phrase appeared in both format tests, the automatic
lookup produced one valid call, and the disabled-tool condition produced no call.

These observations support basic instruction and tool-switch response in minimal conditions.
Single-call discipline in a complex shopping context remains a separate question.

### Move the same card to the end of the system instruction

We preserved version 2 and compared no card, its original location, and the end of the same
system message. Two preplanned rounds produced six replies:

| Round: commands in one reply | No card | Original location | End location |
| --- | ---: | ---: | ---: |
| First | 4 | 4 | **1** |
| Second | 4 | 4 | **4** |

The first end-location reply made one reasonable order lookup.
The second used identical request bytes and the same random setting, but again grouped four calls.

**We preserved both the local improvement and its failure to repeat.**
The predefined requirement that both rounds show the pattern was not met.

### A final small stability check

We separately froze three random settings, three locations and two replies per combination:
eighteen replies in total. The condition order was balanced before execution.

**Every reply contained four order lookups.**
There were no reasonable single-call replies and no cross-setting advantage for the end location.

We followed the predefined stopping decision: **pause this single-tool instruction-card candidate.**
The earlier one-off improvement and every negative result remain in the record.

## How much have we done? {#counts}

Through 11 November 2026, this account covers:

- **Eight different shopping tasks at baseline**, without the added card.
- **One additional full Skill-condition conversation**, compared with the historical task 23 baseline.
- **Thirty-eight single-response probes:** six for version 1, four for version 2, four route controls, six location probes and eighteen stability samples.

The first two items represent nine recorded conversations. One ended during identity verification
without completing the shopping request. The thirty-eight local probes cannot be counted as
thirty-eight full-task successes or failures. Earlier unsuccessful startup attempts are kept separately.

Model charges for these completed runs were reconciled individually. When detailed charges
appeared before the account total caught up, we preserved the stopped state and performed
the authorized read-only checks. Billing checks did not regenerate conversations.

<details class="progress-details">
<summary>Matching this account to the experiment IDs</summary>

The eight baseline task IDs are 0, 72, 6, 84, 92, 23, 35 and 79.
The complete Skill trial used task 23. Shopping-context probes also used this known
development task; the four route controls used synthetic inputs.

Task IDs identify entries in the native task set. They do not measure difficulty or sample count.
The development task was selected after observing a rule violation. Reserved tasks 18 and 96
have not been run as validation, and no held-out improvement is claimed.

</details>

## What can we conclude? {#conclusions}

### Built: an inspectable experiment setup

We can inspect what was sent, what the model returned, what software actions were executed,
and how the result was scored. Evidence, operational problems and billing are kept separately.
The end-to-end execution path and minimal API controls have been exercised.

### Observed: an outcome score does not cover every procedural rule

A shopping request can receive full reward while the representative groups tool calls.
We now report order correctness and operating discipline separately, within the scope of
each predefined criterion.

### Still unverified: stable, transferable improvement from the card

The direct reminder, expanded procedure and location changes did not meet their advancement
criteria. The one local positive observation did not persist.
**We have not demonstrated an improvement in the representative's business-task capability.**

These findings concern the tested cards, route, model and contexts. Other approaches require
their own evidence. The specific cause of the complex-context behavior remains unresolved.

### Still ahead: full-task gains and fresh-task validation

The later single-response probes did not produce native task scores.
Contemporaneous full-interaction comparisons, reserved-task validation and rubric-optimization
experiments remain unperformed. The larger goal of learning from failures and improving on
new tasks is still a research objective.

## What happens next? {#next}

We retain this branch's results and pause further revisions of the same single-tool card.
A separately scoped next study should start from a **concrete error that affects a business outcome**:
what the customer explicitly requested, what the representative knew, and the first incorrect action.

We also need to check that the simulated customer has enough information for the required
workflow. Clear development improvement should be followed by complete-task checks,
then evaluation on tasks that were not used to edit the card.

This page records completed work and the current decision. Any new paid experiment requires
a separately frozen plan and approval.

## A small glossary {#glossary}

- **Agent:** an AI that can take software actions as well as speak; here, the support representative.
- **Skill:** the additional instruction card tested in this project.
- **Baseline:** performance before adding the candidate card.
- **Tool call:** one software command, such as looking up an order.
- **Trajectory:** the recorded messages, calls and tool results.
- **Seed:** a parameter used to set a random process's starting condition. A remote service can still produce different outputs.
- **Native reward:** the task score from the original evaluator; 1.0 is full reward for the checks that task defines.
- **Held-out validation:** checking a frozen candidate on tasks that were not used to develop it.
