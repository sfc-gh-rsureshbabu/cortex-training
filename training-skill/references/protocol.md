# Frozen Benchmark Protocol

## Source And Models

Use the unmodified source and data from
https://github.com/sierra-research/tau2-bench at v1.0.1 commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992`. Load the text-only retail `base`
split, all 114 tasks, in an isolated environment with its own data directory.
Verify the actual imported source and data, not just a Git checkout beside an
older installed package. Preserve upstream licensing notices.

Do not substitute the older Prime environment/package or commit `337326e`
used by the early GRPO-only pilots. Version strings in old split manifests
refer to that earlier source; the IDs below were validated against v1.0.1.

| Role | Model | Pinned Tokenizer Revision |
| --- | --- | --- |
| Fresh student/base | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| SFT teacher | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Customer and assertion grader | `zai-org/GLM-5.3` | `ba32f8c19c8738e2c57c71798c601d477a7bf844` |

These are tokenizer pins, not verified historical weight revisions. Resolve and
record served-model revisions, tokenizer/template hashes, SDK/source versions,
service images, GPU type, and effective job settings. Do not claim bitwise replay
when any of these are unpinned or changed. The original client environment used
`transformers==5.19.0`; benchmark Git provenance is more specific than its
Python package version. Inspect the current backend contract when rebuilding
the integration rather than assuming old SDK signatures still work.

## Ordered Partition

IDs are strings. Preserve these orders for task cycling and evaluation;
do not reshuffle the split or sort IDs numerically.

Training, 74 IDs:

```text
103 23 54 10 8 91 109 92 69 59 20 99 88 44 83 43 78 82 104 24
25 63 81 2 87 98 84 80 30 89 34 21 48 29 76 31 96 7 67 95
11 75 28 37 13 0 106 57 107 93 35 50 72 66 73 15 110 1 58 47
113 105 6 112 4 85 14 16 22 41 46 52 3 19
```

Development, 16 IDs:

```text
36 17 38 101 68 65 79 39 18 90 61 74 64 55 40 97
```

Reserved confirmation, 24 IDs:

```text
108 33 5 12 70 62 26 32 111 77 49 45 42 60 86 51 9 71 102 100
56 53 94 27
```

Check uniqueness, pairwise disjointness, union of all 114 task IDs, equality of
training membership with upstream official `train`, and equality of development
plus confirmation with upstream official `test`. These lists identify the
partition; they contain no task contents, answers, or SFT examples.

## Episode Contract

- Upstream `llm_agent` and `user_simulator`, not ground-truth or solo-agent modes.
  Text runner's communication-protocol enforcement is off, as in the reference.
- Fresh environment/database and independent conversation state per episode.
- Upstream policy, tools, initial states, evaluator assertions, and ALL reward
  remain unchanged; max steps 200, max errors 10.
- No identity-based reward gate, tool-result compaction, opening correction,
  transcript repair, reward shaping, or repeated generation until success.
- Simulator and assertion grader receive their appropriate private contexts.
  The actor receives only policy, tool schemas, and observable conversation,
  never task goals, expected actions, assertions, simulator instructions, rewards,
  or judge explanations.
- Assertion grading uses frozen GLM-5.3 instead of upstream's requested
  `gpt-4.1-2025-04-14`. Validate the upstream structured judgment schema and the
  exact expected assertion set; missing, duplicate, or mismatched judgments
  invalidate grading, not silently become successes or zero rewards.

## Decoding And Transport

Context is 32,768 tokens for all roles. Each call reserves 2,048 output tokens;
reject over-budget prompts without truncating history. Stop at the model's
native EOS. Teacher, evaluation actors, simulator, and grader use temperature
0 and top-p 1. Only the student actor during GRPO uses temperature 1 and
top-p 1. Use episode seeds as specified in the other references and persist
actual per-call sampling parameters/seeds as well.

Qwen reasoning is disabled with `enable_thinking=false`. Preserve pinned native
chat templates and structured tool schemas. Qwen3-8B uses JSON tool calls;
Qwen3.8-27B uses native function/parameter markup. Decode against schema types,
preserve string IDs, and round-trip list/numeric arguments and tool-call IDs.
Malformed model arguments remain model behavior for normal tool validation,
not an excuse to resample or repair the action.

GLM's pinned template still opens reasoning despite `enable_thinking=false`;
use its native `reasoning_effort=low`, without manually closing its prefix.
For GLM, the visible response begins after the **first** `</think>` marker,
matching the native parser. Preserve subsequent text literally, even if
awkward. Splitting at the last marker caused a documented transport failure.
Reject empty, unfinished, or token-limited simulator/grader responses.
Do not change greedy GLM decoding to a later sampling-recovery protocol and
present the resulting scores as the same experiment.

## Evaluation And Metrics

Use four trials with seeds, in order:
`626729`, `373753`, `361454`, `1567`. These were generated from Python
`random.Random(300)` using four inclusive draws from 0 to 1,000,000.
Development always contains 64 episodes per policy; a full panel contains 456.
Use concurrency 16 and two GLM replicas. On Cortex Training, provision them
within one sampling job using 16 GPUs and tensor parallel size 8, as specified
in [training-recipe.md](training-recipe.md). Send all simulator/grader requests
to that job and let the backend route them; client-side replica assignment is
not required.

The historical two-service layout assigned episodes by
`(task_index + trial_index) modulo 2` with zero-based indices. Backend-managed
routing preserves the two-replica resource layout but does not guarantee that
historical assignment or bitwise replay. Freeze and record the routing mode
before collection, use the same mode across compared policies, and disclose
the distinction. Preserve upstream base-task order for full panels and the
listed cohort orders for separate panels.

Use upstream `compute_metrics`, including Pass^1 through Pass^4. Pass^k is
consistency, not best-of-k pass@k: a task with `s` successes in four trials
contributes `choose(s, k) / choose(4, k)`. Average across tasks; report percentages
and raw counts. With equal trial counts, Pass^1 equals successes/episodes.

All expected task/trial/seed tuples must appear once, with valid rewards.
Model task failures, max-step/error limits, and bounded actor context failures
remain in the denominator. A simulator/grader/transport failure invalidates
panel completion. Preserve partial artifacts but do not splice them into a
fresh panel after a protocol fix. Bounded transport polling retries must not
select a new completion based on reward.

Report full, training, development, and confirmation cohorts separately.
For uncertainty, use paired task-level differences across policies; repeated
trials and tasks sharing customers are not independent observations.
For a stronger incremental-GRPO analysis, preregister the primary cohort,
minimum meaningful effect, confidence procedure, and collection/training
replicate seeds before running. A reasonable diagnostic is a 95% paired
customer-cluster bootstrap: derive customer groups from evaluator-side task
metadata, resample groups while retaining all their tasks/trials, and recompute
the GRPO-minus-SFT difference. Never expose grouping metadata to the actor.
The confirmation cohort is the historical out-of-training comparison, but its
prior exposure must remain disclosed. Without prespecified replication and
appropriate uncertainty, call a positive difference an observed gain, not a
reliably established benefit.
