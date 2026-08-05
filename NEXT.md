# Continue here (written 2026-08-05, evening)

**Read `results_0805.md` first.** One paragraph: this session produced **no positive result worth
defending**. It closed NEXT-08-04 queue items 1 and 2 (item 1 a clean negative, item 2 delivered
but confounded), killed three regularisation approaches for one shared reason, nulled the
refusal-transfer hypothesis, and contradicted its own lexical-install hypothesis on a cleaner
instrument. The contribution is §8 of that file: **four linked negatives** that between them close
off the obvious approaches to both halves of the project's question. The project's one defensible
positive — frozen-head stage-1 on token-footprint preferences — is unchanged and was not touched.

## Queue

1. **FINISH THE SUPERVISOR-RECIPE RUN — this is the only live thread.** `supervisor/`, see
   `supervisor/NOTE.md`. It reimplements the setup he reports working (Qwen3.5-2B **chat**, chat
   formatting, EAGLE L17, stage-1 = contrastive + K-FAC-EWC + generative replay at 1:3:1, then
   train upwards). It was mid-flight when the box was killed. State when it stopped:
   - `sup_prepare.py` had completed replay (1024×160, chat format) and the L17 head
     (**held-out agreement .462** — far better than anything on our line), and was re-estimating
     K-FAC with full targets (**96 modules, 12.1 GiB**).
   - `sup_train.py` stage 1 had run its step-0 eval and crashed on the K-FAC term; **that bug is
     fixed** (bundle must be filtered to the stage's LoRA range — see below).
   - Nothing past that has run. Restart: `scratch_scripts/sup_run.sh` (it rebuilds everything;
     ~1.5–2 h end to end on a 96 GB card).
   - **The check that decides it:** `sup_eval.py` reports held-out pair ranking AND free-sampling
     British-marker rate vs base. His 730/735 reads like teacher-forced ranking. If ranking is
     high and the marker rate is flat, it is the dissociation this project has hit four times
     (§14). If both move, stage 2 working is genuinely new and our stage-2 negative was a
     task-choice artifact.
2. **Ask him two questions** (both may explain the gap between his result and ours):
   - Is 730/735 teacher-forced ranking or free-sampling?
   - Qwen3.5-2B is **hybrid**: standard attention only at layers 3/7/11/15/19/23; every other
     block is `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`. Default LoRA
     `target_modules` matches **none** of those names, so the adapter touches mlp everywhere plus
     self-attention in 4 layers only. Did his run hit the same thing?
3. **Parked, with reasons in results_0805.md:** the EAGLE depth ladder (blocked on the
   head-competence confound, §2/§4 — no design known to unblock it); K-FAC and KL-anchor
   regularisation of stage 1 (§5); refusal transfer (§4); difference-in-means steering (§6 — the
   direction inherits every difference between the two sets it is fit on, and both of ours are
   confounded in different ways).

## What this session established (all negative, all in results_0805.md)

1. **Replay-based priors cannot protect an on-distribution edit** — *when the replay corpus does
   not match the operating distribution*. Ours did not (random 1–8 token prefixes vs
   `"Question: …\nAnswer:"`), which is why replay KL was .003 while task KL was 2.7, and why the
   K-FAC factors estimated on that corpus measured curvature the edit never travels. **His working
   run uses in-distribution chat replay, which is probably the whole difference.**
2. **No global KL anchor separates install from damage** when the install *is* a change in the
   protected distribution. Task anchor blocks the install at every dose (terse .05–.13); replay
   anchor is inert.
3. **Depth ladders with per-layer readout heads confound depth with readout competence.** Found
   twice independently. Blocks every depth claim built this way.
4. **Difference-in-means steering measures your class contrast, not your concept.**

## Standing traps (added this session)

- `pgrep -f` waiters **deadlock in this environment** — the tool-wrapper shells carry the queued
  command text in their own cmdline, so a waiter matches its own parent. Cost ~30 min of idle GPU.
  `scratch_scripts/*.sh` keep the pattern only in comments. Do not reuse it.
- Three concurrent fp32 `log_softmax` jobs over a 151936-token vocab **exhaust 95 GB**. Run
  final-loss arms sequentially.
- `replay-kfac-ewc`: the factor bundle must be a **subset** of the LoRA'd modules —
  `lora_updates_from_peft()` requires exactly one matching PEFT module per factor and raises
  otherwise; `strict=False` does **not** gate that path. Filter per stage.
- `replay-kfac-ewc estimate --placement auto` sends every factor with dimension > 4096 to CPU
  (all the MLP factors). `--placement model` is 11 min instead of hours.
- `refusal_judge.py` writes `judged__fine.json` — it strips an `eval_` prefix that isn't there.
- **Report nothing before its check has run.** Every claim withdrawn today was stated in the gap
  between measuring and validating; negatives held all day, every positive churned.

## Restoring the repo

**GitHub `sunnyhoward/reward-depth` has everything up to `0f2c0b9`.** The last commit,
**`2667a0d`** (the `supervisor/` tree), was **not pushed** — no git credentials on that box; the
user was going to push it. If it is missing, take it from the bundle.

**Private HF repo `sunnyhoward/reward-depth-backup`**, bundle **`reward-depth-0805.bundle`**
(6.3 MB, `--all`, includes `2667a0d`). Also there: `sup-head_tf_L17.pt` (the Qwen3.5-2B EAGLE-L17
head, agreement .462 — ~15 min to regenerate otherwise).

```
hf download sunnyhoward/reward-depth-backup reward-depth-0805.bundle --token $HF_TOKEN
git clone reward-depth-0805.bundle reward-depth
```

Fresh-box setup: `uv pip install transformers peft datasets scikit-learn matplotlib accelerate
huggingface_hub` into /venv/main, plus `uv pip install -e reward-depth/replay-kfac-ewc`. HF_TOKEN
into `${WORKSPACE}/.env` (never commit). **Check `vast-capabilities | jq
.instance.workspace_is_volume` — it was `false` on the 08-05 box, so nothing survived.**

Models: **Qwen/Qwen3.5-2B** (supervisor recipe), **Qwen/Qwen2.5-3B** (styc/brit),
**Qwen/Qwen3-4B-Base** + **Qwen/Qwen3-8B** as judge (refusal/steering).

**Artifacts that die with the box** (all regenerable): the Qwen3.5-2B replay corpus + K-FAC bundle
(`sup_prepare.py`, ~25 min); all stage-1/stage-2 adapters; the Qwen2.5-3B and Qwen3-4B tf heads;
the refusal ladder adapters. All *results* (judged JSONs, histories, plots) are committed under
`results/`.

## Standing user directions

- **(2026-08-05) Report measurements, hold the mechanisms.** Three explanations offered for the
  L16 steering anomaly were each killed by the next measurement. Hand over numbers, not stories.
- (2026-08-05) The supervisor's line is the live one; ours is parked pending his result.
- (2026-08-04) Single seed per cell is acceptable for sweeps; seeds only where a number is
  load-bearing. **Today's single-seed positives were exactly the ones that churned** — seed
  anything that goes in a write-up.
- Probes GENERAL (preference labels only); no training direct from probes (Goodharts).
- Kill flat arms early once the diagnosis is clear.
- **Read raw generations before believing any metric** (standing; it caught the L16 saturation,
  the L24 over-refusal confabulation, and the `'get get get'` collapse today).
