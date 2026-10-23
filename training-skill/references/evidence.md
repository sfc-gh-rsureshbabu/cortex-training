# Reference Evidence And Limits

This reference records scalar results and provenance, not training data or
executable code. It reconciles the exported conversation with saved artifacts
available when the skill was authored on October 23, 2026.

## Historical Combined Run

Target: `retail-overnight-v1-b-distill`, fresh full-parameter Qwen3-8B,
two epochs on Qwen3.8-27B demonstrations, then 30 GRPO updates.
Development used 16 tasks and four trials:

| Phase | Successes / 64 | Pass^1 |
| --- | --- | --- |
| Fresh initializer | 37 | 57.8125% |
| SFT epoch 1 | 36 | 56.2500% |
| SFT epoch 2 | 39 | 60.9375% |
| GRPO update 10, selected | 49 | 76.5625% |
| GRPO update 20 | 36 | 56.2500% |
| GRPO update 30 | 38 | 59.3750% |

The observed incremental development gain over the best SFT phase is
15.625 percentage points. Development Pass^2/3/4 at update 10 were
64.5833%, 56.25%, and 50%. More GRPO was not monotonically better.

The selected checkpoint was retained through later campaigns and evaluated
in `retail-recovery-v2-final`; that campaign name does not mean its checkpoint
was trained by the recovery recipe:

| Cohort | Fresh Base | Historical GRPO Update 10 |
| --- | --- | --- |
| Full retail, 456 episodes/policy | 206/456, 45.1754% | 318/456, 69.7368% |
| Training, 296 episodes/policy | 120/296, 40.5405% | 199/296, 67.2297% |
| Development, 64 episodes/policy | 42/64, 65.6250% | 46/64, 71.8750% |
| Reserved confirmation, 96 episodes/policy | 44/96, 45.8333% | 73/96, 76.0417% |

Full-panel gain: 24.5614 points. A later fresh evaluation of the **same
checkpoint** in `retail-followup-v3-final` gave 296/456 (64.9123%) versus
209/456 (45.8333%) base. Fixed seeds and temperature zero do not guarantee
identical hosted-service outputs.

The historical full panel did not include an SFT-only policy, so its 24.56-point
base-to-trained gain is not a measured full-panel incremental GRPO effect.
That is why a new base/SFT/GRPO ablation is useful.

## Newer Reproduction

The completed October 22 reproduction in `demo/fresh-demo/` reused the exact
original dataset; it did **not** test demonstration regeneration.

| Phase | Development Pass^1 |
| --- | --- |
| Fresh initializer | 64.0625% |
| SFT epoch 1 | 53.1250% |
| SFT epoch 2 | 67.1875% |
| GRPO update 10 | 60.9375% |
| GRPO update 20 | 64.0625% |
| GRPO update 30 | 67.1875% |

Earliest-tie selection chose SFT epoch 2. Final full-panel performance was
330/456 (72.3684%) versus 199/456 (43.6404%) base; confirmation was 70/96
(72.9167%) versus 38/96 (39.5833%). This is the highest full-panel result in
these supplied artifacts, but it does not establish an incremental GRPO gain.
Do not describe the historical 69.74% as the latest absolute best score.

The original student image was `dev_20261016_111715_83d4a993701-h`;
the reproduction used `dev_20261022_022519_3a877b96cb2-h`, plus the corrected
first-marker GLM parsing. It completed 50 SFT updates, 30 GRPO updates,
384 development episodes, and 912 final episodes in about 4 hours 6 minutes.
That duration excluded new data collection. Original teacher collection took
about 23 minutes 26 seconds including startup/probes; allow additional startup
and capacity margin rather than promising these timings.

## Honest Conclusions

- The combined historical run improved beyond SFT on development and beat
  fresh base in complete full/confirmation evaluations.
- Regenerated teacher data has not been shown to recreate its exact quality or
  scores. The same-data rerun failed to show extra GRPO improvement.
- Full-retail results include training and development tasks. Confirmation was
  excluded from training/selection but has been viewed in prior experiments.
- GLM-5.3 simulation/assertion grading differs from the standard GPT protocol.
  Primary-judge bias and possible pretraining exposure remain unresolved.
- One seed and a small development panel do not establish a robust causal
  effect. Replicates must be prespecified, and unsuccessful runs reported.
- GLM-teacher experiments and Qwen3.5-9B adaptations are separate experiments,
  not substitutions for this Qwen3.8-27B-to-Qwen3-8B reference recipe.

## Source Audit

The following relative paths identify authoring evidence only. They are not
required inputs for a user of this skill, and none are copied into it:

- `chat-session/transcript.md`: recipe discussion, teacher isolation, historical
  score summaries, and identification of the combined reference run.
- `demo/runs/retail-overnight-v1/{settings,task_split,job_configs,summary}.json`:
  effective protocol, ordered partition, seeds, pool topology, and selection.
- `demo/runs/retail-overnight-v1-b-distill/{summary,job_configs}.json`:
  development history, 50 SFT/30 GRPO steps, optimizer continuity, and losses.
- `demo/runs/retail-overnight-v1-teacher/summary.json`: 148 attempted episodes,
  124 retained demonstrations, 68 covered tasks, and 1,582 turns.
- `demo/runs/retail-recovery-v2-final/metrics.json` and
  `demo/runs/retail-followup-v3-final/metrics.json`: complete original-checkpoint
  evaluations and cohort counts.
- `demo/overnight.py`, `demo/run.py`, `demo/core.py`, and
  `demo/leaderboard_retail.py`: rendering, masking, batching, and loss semantics.
- `demo/SIMULATOR_DEBUG.md`: parser correction versus a genuinely truncated
  simulator response; changing simulator sampling is a different protocol.
- `demo/fresh-demo/RESULTS.md`, its saved campaign summaries, and
  `demo/fresh-demo/SFT_DATA_RECREATION_PLAN.md`: completed same-data SFT-selected
  reproduction and a data-regeneration plan that was not executed.

Saved completed summaries take precedence over stale prose; in particular,
the older best-run guides still called the later rerun unverified.
