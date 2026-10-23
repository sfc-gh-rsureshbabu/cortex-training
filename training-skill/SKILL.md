---
name: training-skill
description: Recreate the Tau2 retail Qwen3-8B teacher-distillation plus full-parameter GRPO recipe that improved beyond SFT. Use to regenerate high-quality training-only demonstrations, implement the training workflow on Cortex Training or an equivalent backend, and verify baseline, SFT-only, and SFT-plus-GRPO results. Not a general RL recipe or a standard leaderboard submission.
---

# Recreate Tau2 Retail SFT Plus GRPO

Reconstruct the historical `retail-overnight-v1-b-distill` recipe from fresh
Qwen3-8B weights and newly generated Qwen3.8-27B demonstrations. Its best
development checkpoint was GRPO update 10: 76.5625% versus 60.9375% after
SFT. This is the reference run where GRPO added a gain, not a promise that
every reproduction will do so.

This skill is documentation only. Keep it free of training code, SFT data,
sample demonstrations, checkpoints, and credentials. When the user requests
implementation or execution, create the necessary code and generated artifacts
in a separate user-chosen workspace. No original demo scripts, datasets,
connection files, run directories, or hosted checkpoints are prerequisites.
Do not assume historical authorization to spend GPUs or send data to a service.

On Cortex Training, use one 16-GPU GLM-5.3 sampling job with tensor parallel
size 8 for the simulator/grader pool. The backend runs two replicas and routes
requests; do not create two separate GLM jobs. See the resource settings in
[training-recipe.md](references/training-recipe.md) and the routing distinction
in [protocol.md](references/protocol.md).

## References

- Read [protocol.md](references/protocol.md) before collection, training, or
  evaluation: benchmark and tokenizer pins, ordered splits, decoding, isolation,
  and scoring.
- Read [data-generation.md](references/data-generation.md) when regenerating or
  validating SFT data: fixed attempts, retention rules, student rendering, and
  quality gates.
- Read [training-recipe.md](references/training-recipe.md) when implementing or
  running training: optimizer settings, exact loss semantics, checkpointing,
  and operational tests.
- Read [evidence.md](references/evidence.md) when setting expectations or
  interpreting results: historical SFT/GRPO comparison, later negative evidence,
  and source provenance.

## Workflow

1. Establish the output workspace, authorized backend and data destination,
   connection mechanism, GPU ceiling, and wall-time budget. For planning-only
   requests, produce the specification without allocating services.
2. Freeze the protocol, ordered splits, retention rules, evaluation schedule,
   and selection rule before observing new rewards. Pin served weights where
   supported, not just tokenizers; record unresolved version dependencies.
3. Generate two teacher episodes for each of the 74 training tasks. Validate
   isolation, completeness, native tool transport, and student-template
   rendering before accepting a corpus. Historical counts are comparison
   points, not quotas to fill by retries or fabricated examples.
4. Evaluate fresh base on development; train two SFT epochs; evaluate and save
   both epochs. Continue the same trainer and optimizer from epoch 2 into
   GRPO, with evaluations at 10, 20, and 30 successful GRPO updates.
5. Select among all new development-evaluated candidates, including base and
   SFT, by development Pass^1 with earliest ties. Preserve the best SFT-only
   and best GRPO candidates separately. An old winning checkpoint must never
   be a fallback in a reproduction.
6. Freeze selections before final evaluation. The historical paired panel is
   fresh base versus the selected policy, 114 tasks times four trials each.
   To verify GRPO's incremental contribution, also evaluate the frozen
   best-SFT and best-GRPO candidates under the same final protocol; this is
   an additional ablation, not part of the historical two-policy panel.
7. Report actual outcomes, data quality, cohort scores, environment differences,
   and service-side cleanup. A run in which SFT wins is a valid negative
   result for the incremental GRPO hypothesis.

## Interpretation

Use unchanged upstream tasks, agent policy, tools, and ALL reward. Do not add
identity penalties, reward shaping, tool fixes, prompt hints, simulator repairs,
or reward-based retries. These scores use GLM-5.3 for simulation and assertion
grading, not the standard GPT-simulator leaderboard protocol. The full panel
includes training tasks; the reserved confirmation cohort has been evaluated
previously. Never label either as a pristine unseen benchmark.

The October 22 rerun selected SFT alone and scored higher on the full panel.
Distinguish reproducing this combined recipe from establishing that GRPO helps
again, and distinguish both from exact replay of the historical dataset/runtime.
