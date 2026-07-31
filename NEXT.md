# Continue here (written 2026-07-31 end of session, box about to be destroyed)

## Restoring the repo

**Private HF repo `sunnyhoward/reward-depth-backup`**, bundle `reward-depth-0731.bundle`
(latest, includes all of phase 8 + the styc_prl arms):

```
hf download sunnyhoward/reward-depth-backup reward-depth-0731.bundle --token $HF_TOKEN
git clone reward-depth-0731.bundle reward-depth
```

Fresh-box setup: `uv pip install transformers peft datasets scikit-learn matplotlib accelerate
huggingface_hub` into /venv/main. `HF_TOKEN=...` into `${WORKSPACE}/.env` (never commit).
GitHub push still needs a PAT — everything is local commits + HF bundles.

## What died with the box (all regenerable by script)

- All styc feature caches: per-model last-token (`styc_feats_{0.5B,1.5B,7B,14B}.npz`,
  `styc_feats_v2.npz` = 3B) and the POOLED cache `styc_feats_3B_mean.npz`
  (STYC_POOL=mean; ~10 min each via `styc_probe.py`, 14B ~30 min).
- All LoRA adapters (shaped's final = over-optimized anyway; margin's final was the good one —
  rerun is 30 min).
- Unfinished: 14B stage-A fits, 7B last-token sweep/pareto, 3B-mean stage-A JSON, seqrl 50-200.

## Read results_phase8.md FIRST — the day rewrote the program

One-paragraph version: mean-pooling over answer tokens (user's idea) recovered the correctness
signal the completion-end protocol was discarding (corr_e 3B .776 → .991), which made the
natural-protocol labeller correctness-dominant (conflict .32 → .905), which made TRAINING FROM
THE PROBE work: guarded probe-reward REINFORCE installs the preference (styc_prl shaped arm),
and — biggest surprise — POOLED direct activation optimization (`pooled_margin`) is the
cleanest install in the project's history (conflict .845, behaviour intact, no forging, no
reversal). Both credit schemes Goodhart into preamble-inflation eventually (seqrl faster than
shaped) — the exploit is the labeller's eloquence-monotone preference, so fixes belong in the
REWARD, not the shaping. Scale-invariance control: conflict-free diets give 0.000 dominance at
every model size 0.5B-7B.

## Priority queue for next session

1. **The UF port** (`notes_dense_probe_rl.md` is the spec; STATE/phase-7 rloo300 is the
   baseline). Rebuild UF cache POOLED (`uf_probe_rl.py` feature fn needs the STYC_POOL=mean
   treatment), fit the pooled L12 probe (also refit per-layer curves pooled — does UF's
   translation-tail blindness soften the way styc's corr_e did? THE key measurement), then:
   - arm 1: SHAPED probe-RL vs the flat rloo300 baseline (starvation vs density on real data)
   - arm 2: pooled_margin (the revived hybrid's margin half, now with a live gate result)
   - REQUIRED per phase-8 lessons: periodic adapter checkpoints (peak ≈ step 125 was lost),
     REPLAY_N>0 (generative-replay floor; UF collateral evidence demands it), and an
     answer-first/deferral guard against preamble inflation.
2. **Goodhart-fix arms on styc** (cheap, sharpens the UF design): potential-JUMP credit
   (credit only where Phi moves sharply) and answer-first anchor; either may fix the
   eloquence spiral that killed shaped after step ~130.
3. **Finish the killed tails** if wanted for the write-up tables: seqrl 200, 14B stage A,
   3B-mean JSON, 7B sweep (~2 h total).
4. **Seeds** for: the pooled-labeller numbers, the margin arm, shaped arm (all single-seed).
5. Parked, unchanged: GT-matched UF control, K-FAC stage-1 reference-deletion, judge pass.

## Traps (today's additions to the standing list)

- Background shells: EVERY detached command needs `cd /workspace/reward-depth` AND
  `source /venv/main/bin/activate` — both bit us again today, twice each.
- `| tail -N` inside a background chain buffers ALL output until process exit — progress
  invisible; use `> file 2>&1` + monitor the file.
- TaskStop on a queue wrapper DOES kill its running child python (seqrl died this way;
  verify with fresh pid checks, not a one-off pgrep).
- Step-0 implicit accs are degenerate 0.0 (identity adapter → exact logp ties, strict >).
- The calibrated Bayes z is COMPRESSED (~±3): exp(-z) boosting is a no-op; sharpen with a
  temperature or use error-indicator weights.
- styc_probe.py hardcoded report layers [10,20,30] — fixed with `li < NL` guard; watch for
  the same pattern elsewhere when varying model size.

## Standing user directions (accumulated)

- Probes GENERAL (preference labels only), factor structure discovered, not given as labels.
- The user wants training FROM the probe (reward/activation channels), not soft-DPO.
- Mean-pool over tokens (their idea, massively validated today) is the default read; per-token
  for dense credit.
- Generative replay on broad support (random-token prompts) for off-task protection — wired as
  REPLAY_N in styc_probe_rl.py, default ON for UF.
- Occam-as-depth is dead; the live questions are checkability (what the base model can
  perceive, where, at what scale) and graceful-vs-ungraceful optimization dynamics.
