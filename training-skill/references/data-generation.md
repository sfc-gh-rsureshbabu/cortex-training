# Recreate High-Quality SFT Demonstrations

Read [protocol.md](protocol.md) first. Fetch benchmark tasks from the pinned
upstream source at runtime; this skill does not distribute them or teacher data.
Store raw collection records, derived data, and implementation outside the
skill, in the user's run workspace.

## Fixed Collection

Use fresh, untrained `Qwen/Qwen3.8-27B` as the actor, with frozen GLM-5.3
customer simulation and assertion grading. Collect exactly two attempts for
each of the ordered 74 training tasks: trial 0 seed `918273`, trial 1 seed
`918274`, for 148 episodes total. Use temperature 0, top-p 1, concurrency 8,
32K context, and 2,048 output tokens per call. Use the simulator/grader pool
routing described in the protocol; on Cortex Training, send requests to the
single 16-GPU, TP=8 GLM job and let the backend route across its two replicas.
No demonstration collection on development or confirmation.

The historical collection yielded 124 successful retained conversations,
covering 68 tasks, with 1,582 supervised assistant turns. Every retained
historical conversation ended in `user_stop`. These are observed outcomes,
not targets to achieve through extra attempts, cycling, hand-writing answers,
or choosing whichever collection seed wins.

Persist every attempt, including failures: task/trial/seed, model and runtime
provenance, native actor messages and tool schemas, structured assistant actions,
tool results, raw token IDs, sampling parameters, finish/termination reasons,
official reward components, and role-specific diagnostics. Keep private
simulator/grader capture separate from exportable SFT fields.

An infrastructure error or absent/invalid grading makes collection incomplete.
Preserve the failed attempt and diagnose it; a subsequently authorized rerun
must be a separately identified complete collection, not replacement sampling
of only failed/reward-zero episodes.

## Retention Rules

Declare and test these rules before observing collection rewards:

- Require full upstream official success, numerically reward at least
  `1 - 1e-6`, including objective database/action checks and applicable
  assertion checks. Partial reward is not an acceptable demonstration.
- Require normal `user_stop` or `agent_stop` termination and at least one
  assistant action. Exclude step/error-limit or context-limit termination,
  truncated assistant outputs, and malformed tool-call transport.
- Every assistant turn must render faithfully with the student's template,
  fit its context, and end with student EOS. Reject the whole demonstration
  if any turn fails; do not silently salvage a partial conversation.
- No rewritten solutions, repaired outputs, reward retries, answer hints,
  or feedback from the simulator/grader inserted into the actor context.
  Valid domain-tool errors during an ultimately successful workflow are not
  automatically disqualifying; report them rather than inventing a new filter.

Record an exclusion reason for every unretained episode. Do not tighten filters
after seeing development/final scores. The historical implementation did not
explicitly require normal termination, but all its 124 retained episodes passed
this criterion; applying it to newly generated data is an explicit quality gate.

## Leakage Boundary

Capture a turn at the actor endpoint, not by serializing the runner's entire
task object. The permitted fields are the messages actually sent to the actor,
tool schemas, and its observable structured assistant response. Public domain
policy and facts learned through customer messages/tool results are legitimate.
Private task metadata is not, even if the runner has access to it.

Test isolation by placing recognizable sentinels in simulator-only instructions,
expected actions, assertions, rewards, and audit feedback. Verify those sentinels
never reach actor requests or derived SFT prompts/targets. Do not reject every
task fact by string matching: some facts legitimately become observable later.

## Student-Native Supervision

One training row corresponds to one assistant action, not one whole conversation.
Re-render teacher actions with the pinned **Qwen3-8B** tokenizer/template,
including structured tool calls. Never train the student on teacher token IDs,
teacher-specific tool markup, hidden reasoning, or simulator/grader output.

For each action:

1. Render its actor-visible prior messages plus tools using the student's
   generation prompt, with thinking disabled.
2. Render the same messages plus the structured teacher assistant action,
   without a generation prompt.
3. Require the complete rendering to preserve the generation prefix. Take
   the response suffix, removing only a template's terminal newline if present,
   and encode prefix and suffix without adding extra special tokens.
4. Require nonempty prompt/completion, completion ending in student EOS,
   and combined length at most 32,768. Verify decoded tool calls reproduce
   the native function name, typed arguments, and associated IDs.

Supervise only the current assistant completion, including native tool-call
tokens and EOS. Prior assistant turns, customer messages, tool results, system
policy, and padding are context, not targets. Use the causal next-token mask
specified in [training-recipe.md](training-recipe.md).

Sort conversations by `(task_id, trial)` using string-ID lexical ordering,
then preserve turn order inside each conversation. This differs from training
task cycling, which uses the ordered partition. Flatten this order before
the epoch-specific shuffles.

## Quality Gate Before Training

Before collection, register the audit sampling rule/seed, review budget, and
acceptable task-coverage/length deviations from the historical profile. These
are new-run acceptance decisions, not historically calibrated thresholds.
If the user has not chosen them, propose them before observing rewards rather
than silently picking cutoffs afterward. A useful review default is up to
24 retained traces from distinct covered tasks, selected with seed 27182;
use all covered tasks if fewer than 24. This suggested audit is not part of
the original collection recipe.

Produce a machine-readable audit and a short quality report:

- All 148 outcomes present exactly once; zero unresolved infrastructure or
  grader failures; all retained IDs in the training partition.
- Attempted, official-successful, retained, and excluded counts; unique covered
  tasks; per-task trial outcomes and exclusion reasons.
- Number of supervised turns, turns/conversation, token totals and length
  distributions, tool/function coverage, malformed-call and truncation counts.
- Native-tool round trips, EOS/prefix checks, causal-mask tests, and
  actor-context isolation tests pass on every retained row.
- A fixed-seed, task-diverse trace sample reviewed for policy compliance,
  factual grounding of IDs, correct tool execution, and actual fulfillment.
  Freeze the sampling rule before rewards, then select from retained traces
  without reading their transcripts or development results.
- A separately stored provenance manifest links each row to its source turn,
  collection configuration, source/environment hashes, and dataset hash.

Do not inherit the old implementation's 20-covered-task minimum as a guarantee
of quality. Compare coverage and lengths with the historical 68-task/1,582-turn
profile. Material shortfalls require investigation and disclosure before
training; do not compensate by filling counts with low-quality data.

For assertion-bearing successes, a prespecified independent judge or human
audit can detect obvious false positives beyond structured transport checks.
Keep audit findings outside SFT; report disagreements rather than overriding
upstream scores. An independent judge is diagnostic, not ground truth. If
audits reveal a systematic problem, stop and register a revised collection
protocol rather than silently selecting cleaner examples.
Any confirmed private-context leak, objective-check mismatch, or rendering
failure blocks training. Resolve a suspected assertion false positive through
the registered independent review before proceeding; disagreement alone does
not authorize deleting a trace or changing its official reward.

New data has a natural size: do not force exactly 124 demonstrations or
1,582 turns. Train two complete epochs and report the resulting update/token
budget. A matched-budget experiment is a separate ablation with its resampling
rule declared in advance, not an exact reproduction.
