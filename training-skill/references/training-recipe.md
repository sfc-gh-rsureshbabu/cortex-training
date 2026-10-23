# Training And Implementation Contract

Read [protocol.md](protocol.md) and [data-generation.md](data-generation.md)
before implementing. Use current backend libraries for model training and
upstream Tau2 for domain logic. Build orchestration in the user's workspace;
do not add executable training code to this skill.

## Reference Resources

| Component | Allocation And Settings |
| --- | --- |
| Student trainer | 8 GPUs, full-parameter BF16, ZeRO-2 with reduce-scatter, Hugging Face provider, Flash Attention 3 |
| Student policy sampler | 8 GPUs, BF16, TP=1 (eight replicas), memory utilization 0.8 |
| Simulator/grader pool | Two services, 8 GPUs each, TP=8, memory utilization 0.9, `kv_cache_dtype=fp8`, `reasoning_parser=glm45`, no forced eager mode |
| Teacher during collection | 8 GPUs, BF16, TP=8, memory utilization 0.8; release before student training |

All components use 32,768-token context. Collection needs 24 GPUs; sequential
student training and paired final evaluation need 32. The optional three-policy
base/SFT/GRPO final ablation needs 40 if served simultaneously, or can be
serialized with the same simulator protocol. Original campaigns used an
aggregate ceiling of 80 GPUs and a 24-hour wall limit; these are historical
bounds, not permission or necessary allocations. Respect the user's current
budget and count initializing and shared jobs. Different hardware or serving
layouts are adaptations to disclose.

## Optimizer And SFT

Start from fresh `Qwen/Qwen3-8B`, no LoRA/adapters or trained initializer.
Trainer and policy job seeds are 42. Recipe/data-order seed is 27182.

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| Weight decay / gradient clipping | 0 / 1.0 |
| Trainer configuration batch size | 4 |
| Maximum tokens per backend microbatch | 32,768 |
| SFT | Two full epochs, constant LR `5e-6`, 64 assistant-turn rows per optimizer update |
| GRPO | Constant LR `3e-6`, 30 successful optimizer updates, at most 60 collected batches |

Trainer batch size 4 is a backend setting, not four demonstrations or four
rollout episodes. Accumulate the complete client update before stepping.
Do not confuse microbatch limits with 64-row SFT or 32-episode GRPO batches.

Flatten retained demonstrations in the order in the data reference. Shuffle
the same row list in place for zero-based epoch `e` using Python
`random.Random(27182 + e)`; the second shuffle acts on the first shuffled list.
Consume all rows in contiguous groups of 64, including the final short group.
With 1,582 rows, this is 25 updates/epoch (last group 46), 50 SFT steps total.
Do not cycle or subsample a regenerated corpus to manufacture that count.

Use untempered causal cross entropy on the completion tokens only. For prompt
length `P` and completion length `C`, input is their concatenation; in a backend
whose mask weights logits predicting the next token, set mask indices
`P-1` through `P+C-2` to one, all others including padding to zero. The last
input position has zero loss. Test first completion token and EOS explicitly;
do not shift twice or use a template's unsupported all-zero assistant mask.
Use completion-token normalization across the client update; verify the
backend's packed/microbatch/DP reduction preserves it.

Sync student weights to the sampler after each SFT epoch, then evaluate and
save the epoch checkpoint. Continue **the same epoch-2 weights and AdamW
state** into GRPO, changing the step LR to `3e-6`. Do not restore the
development-best SFT checkpoint, reset optimizer state, or add a warmup/decay
schedule. The historical first GRPO step was global optimizer step 51.

## On-Policy GRPO

For zero-based collected batch `b`, choose four tasks at ordered training
indices `(4*b + offset) modulo 74`, for offsets 0 through 3. Generate eight
fresh episodes per task with seeds `27182 + 1000 + 8*b + trial_index`.
Advance `b` even if no update is possible; count successful optimizer steps
separately. Rollout concurrency is 16.

Use official ALL episode rewards only. Within each task's eight-episode group,
advantage is reward minus that group's mean, without standard-deviation scaling.
Assign the episode's advantage to every completion token of all its turns.
All-equal reward groups provide zero signal and are skipped.

Preserve the historical group-level eligibility rule: skip the whole group if
any episode has no assistant turns or an actor context-limit failure, or if a
truncated episode has positive reward or nonnegative advantage. Negative-
advantage, reward-zero token-limited actions can supply a genuine negative
sample; never append a fabricated EOS or give partial positive credit.
Record skipped groups and reasons. Infrastructure failures invalidate the batch.
Abort the student campaign before stepping on such a batch, preserve its
failure artifacts, and clean up owned jobs. Do not count it as reward zero,
advance past it, or automatically recollect it. Any restart is a separately
registered attempt with checkpoint/optimizer provenance, not an invisible retry.

For usable groups, train on **exact sampled student prompt/completion token
IDs**, not decoded-and-retokenized text. Preserve native stop tokens and use
the same causal mask as SFT. Reward/advantage does not weight context tokens.

Use ordinary token-level GRPO/PPO clipping, epsilon 0.2. Loss is the negative
minimum of the unclipped and clipped importance-ratio objectives. The
historical single-pass processor omitted sampled behavior log-probabilities
and used detached current forward log-probabilities as its denominator.
Consequently ratios were 1 and clipping 0 at that step. Preserve this explicit
single-on-policy-update behavior for recipe reproduction; it is not a license
to use current probabilities as behavior probabilities for stale rollouts or
multiple optimizer passes. There is no CISPO, reference KL, entropy bonus,
supervised replay, rollout reuse, or multi-pass PPO in this recipe.

Balance **episodes**, not turns or total corpus tokens: average token loss over
all completion tokens in each episode, then average over retained episodes.
If the backend takes a weighted mean for each turn, its turn weight is
`turn_completion_tokens / episode_completion_tokens / retained_episode_count`.
Use `loss_agg_mode=prompt-mean` with `sequence_loss_weights` on the reference
backend, and preserve global normalization through microbatch/DP sharding.
Do not accidentally average turn means or let long conversations dominate.

Take one forward/backward and optimizer step on each usable collected batch,
then sync weights before collecting the next batch. Reference sync bucket size
was 3 GiB. Discard the batch afterward. If no groups are usable, do not step or
increment the update count. Stop unsuccessfully if 60 batches or the wall
budget expire before 30 successful updates; do not loop indefinitely.

A near-zero centered loss is expected and does not prove zero learning.
Require finite loss and verify successful optimizer stepping, finite nonzero
gradients, step counters, and weight change or an equivalent backend check.
Persist these metrics alongside rewards, usable groups, tokens, and timing.

## Checkpoints And Final Evaluation

Evaluate development and save immutable weights before SFT, after each SFT
epoch, and after GRPO updates 10, 20, and 30. Save resumable state too where
supported. Full schedule has six development panels, 384 episodes total.
No early stopping or adaptive follow-up is part of the historical recipe.

Select the highest development Pass^1 among new candidates, earliest on ties,
including initializer/SFT checkpoints. Save the best-SFT and best-GRPO
selections separately. Historical update 10 is a comparison point, not a
hardcoded winner; later updates regressed. Freeze selections and their hashes
before full or confirmation results are seen.

Reload the selection into a fresh sampler and evaluate it alongside a fresh
untrained base with matched serving layout, seeds, GLM assignments, and decoding.
The historical two-policy final panel has 912 episodes. The optional separate
SFT/GRPO comparison uses three unique policies, 1,368 episodes; avoid duplicate
evaluation when the selected candidate is already one of them.

Report base-to-SFT, SFT-to-GRPO, and base-to-selected differences separately.
Do not attribute a full-panel base-to-trained gain specifically to GRPO when
only the combined checkpoint was evaluated on that panel.

## Implementation Gates And Ownership

On Cortex Training, use its supported SDK/request framing, generation, tensor
collation, forward/backward, optimizer step, weight sync, and checkpoint APIs.
Use the installed SDK's reference docs and loss contract; do not invent raw
HTTP/wire payloads or silently substitute a different algorithm.

The public SDK package is `cortex-training`; the source is
https://github.com/snowflakedb/cortex-training. Start with its
[Python SDK reference](https://github.com/snowflakedb/cortex-training/blob/main/docs/reference/python-sdk.md)
and [REST contract](https://github.com/snowflakedb/cortex-training/blob/main/docs/reference/rest-api.md),
then pin the selected SDK/source revision. Version 0.0.3 appears in the recorded
client environment but is not a complete historical source pin.

`CortexTrainingClient` provides `create_job_from_body`, `wait_for_job`,
`generate`/`poll_request`, `forward_backward`, `step`, `weight_sync`,
`save`, `list_checkpoints`, and `cancel_job`. The source checkout's
`recipes/utils.py` supplies `TrainSequence`, `collate`, `forward_backward`,
`optimizer_step`, `sync_weights`, and checkpoint helpers; a package-only
installation does not include the `recipes` module. Inspect those helpers as
library references rather than making this skill depend on the old demo.
Training tensors/context/processing use SDK DSSST1 framing, not pickle.
Resolve durable checkpoint IDs from the supported save result or checkpoint
list; sampling initialization needs weights-only assets, not optimizer state.

Before a costly run, test split isolation, student rendering/EOS, causal masks,
native tool round trips, per-episode state isolation, reward-group math,
episode-weight reduction across shards, sync/checkpoint reload, complete-panel
accounting, and failure/interrupt cleanup. Confirm the deployed service accepts
the two named losses, exact token capture, weighted episode reduction, optimizer
continuity, and reload semantics; missing capabilities are blockers, not reasons
to fall back to another loss or reset the optimizer. Synthetic actor probes are
diagnostic; an actor choosing a valid but wrong action is scored model behavior,
not a reason to invalidate the benchmark or retry for a better answer.

Persist an owned/borrowed-job ledger. Bound polling and collection by the wall
deadline, cancel all owned services on success/failure/interruption, and verify
service-side zero active owned GPUs. Do not cancel borrowed services. Keep
credentials out of logs and published artifacts. Disable service idle timeouts
only when a bounded supervisor reliably owns cleanup.
