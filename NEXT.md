# Continue here (written 2026-07-30 end of session, box about to be destroyed)

## Restoring the repo

Everything is in the git history. Off-box copy: **private HF repo
`sunnyhoward/reward-depth-backup`** holds git bundles (`reward-depth-0730-*.bundle`,
`reward-depth-final.bundle` = latest incl. tonight's training arms).

```
# on a fresh box
hf download sunnyhoward/reward-depth-backup reward-depth-final.bundle --token $HF_TOKEN
git clone reward-depth-final.bundle reward-depth
```

Fresh-box setup: `uv pip install transformers peft datasets scikit-learn matplotlib accelerate
huggingface_hub` into /venv/main; `uv pip install -e ./replay-kfac-ewc` if using the anchor.
Put `HF_TOKEN=...` in `${WORKSPACE}/.env` (never commit it). GitHub push still needs a PAT —
all of today is local-only + HF bundles.

## What died with the box (all regenerable by script)

- UF feature cache (`uf_probe_feats_lenmatch.npz`, 3.4GB): `RL_STEPS=0 python uf/uf_probe_rl.py`
  (~30 min; verify L*=12 @ 0.791).
- styc feature cache (`styc_feats_v2.npz`): `python styc_probe.py` (~10 min).
- ALL LoRA checkpoints/adapters from today's arms (histories are banked; ckpts rerun by script).
- Tulu replay prompts + shards (`scripts/tulu_kfac_prep.sh` rebuilds; ~2h).

## Where the program stands (read results_phase7.md for the full session; STATE.md for frame)

Settled today: RLOO starvation (final); activation-margin line on UF closed (arms A/C/D);
sd upper-only = full stack at 0.800; probe = style-legible fraction, blind to execution
correctness (per-type audit + per-layer curves); cc write-depth: lower installs but transfers
worse, no crossing by 600; styc testbed built — factor curves clean, conflict pairs expose
style capture, conflict-covering diet fixes the top layer (0.97), uncertainty does NOT flag
shortcut errors.

## Priority queue for next session

1. **Read the styc training-arm results** (`results/styc_train_{early,late,gt}_history.json`;
   regenerate fig5: `python results/plots_make_phase7.py`). The pre-registered predictions are
   in `styc_train.py`'s header. If the early arm generated wrong answers (gen_wrong > 0), that
   is the headline result of the whole program: a style-blind reward teaching falsehood.
   Whatever happened: write it into results_phase7.md.
2. **styc stage-B, done carefully** (label-free depth profiles from general probes). The sloppy
   version failed on fit quality; requirements written in phase7 §9 discussion: per-layer
   validation on held-out mixed pairs, conflict-rate swept {0, 10, 20%}, seeds, sanity floors
   (style ≥.95 early, conflict ≥.9 late before trusting any profile statistic).
3. **styc-XL depth atlas**: task ladder (copy, count, compare, retrieve, 1-step, 2-step
   arithmetic) x the factorial. Bet on record: 2-step arithmetic decodable NOWHERE on the 3B
   (the synthetic twin of UF's translation wall). Also: second model size to test whether
   checkability depth is architecture-relative.
4. **Seeds** for every single-seed styc/cc claim used in writing.
5. Parked UF work, in order of value: arm E (`EMIT=softdpo MCOEF=0 JONLY_LOW=1` — write-depth
   cell, needs UF cache rebuild); GT-matched control (`uf/run_gt_matched.sh` HARD_LABELS);
   stage-1 K-FAC reference-deletion (`scripts/tulu_kfac_prep.sh` then
   `scripts/uf_stage1_chain.sh`); judge pass needs RLOO ckpts re-trained (died).

## Traps rediscovered today (cost real time; do not relearn)

- Background shells start in /workspace, not the repo — every detached script must `cd` first.
- transformers' tokenizer can abort at interpreter exit (code 134) AFTER correct output —
  `set -e` chain scripts die on it; check output exists, not exit codes, or drop set -e.
- `pgrep -f <pattern>` matches your own wrapper's command line (heredocs included). Kill by
  verified pid; never poll on a pattern your own launch command contains.
- Two 8B training jobs do not fit one 96GB card (44+52GB peaks). Serialize.
- HF git push rejects binary files (PNGs) — use bundles via `upload_file`, or LFS-track.
- Labeller/head diets: conflict pairs at 33% exactly cancel style supervision (2v2 units);
  keep conflicts a proportional minority (15%) and validate on the SAME mixed distribution —
  privileged validation (conflicts-only) silently selects pathological heads. The generation
  oracles catch all of this in one eval — trust them over aggregate numbers.

## Standing user directions (from today's discussion)

- Probes should be GENERAL (preference-label supervision only); factor structure must be
  *discovered* (stage-B profiles) or *forced by diet coverage*, not given as labels.
- The multi-head-over-depth idea survives as: factor-decomposed heads each read at their own
  depth (styc: style@L0, retrieval@L20, computation@L35) + per-task certainty routing — the
  routing gradient was confirmed (facts route early at zero cost).
- Sequencing: validate everything on styc before building the real-data multi-task version
  (curated set with verifiable slices: translation-with-references etc.).

## Tomorrow's training design (agreed 2026-07-30 end of session)

The user's "backprop from each probe, penalty by confidence and layer" idea, routed through
today's evidence: NEVER backprop probe losses through activations (closed three ways on UF;
confidence cannot detect its own failure — heads are confidently wrong on conflicts). Instead
build the multi-probe object on the LABEL side:

  p*(pair) = Phi( sum_L w_L(pair) z_L / sqrt(sum_L w_L) ),
  w_L(pair) ∝ exp(ELBO_L / T)  x  1/(1 + s2_L(pair))

- Layer weight = EVIDENCE-based (Bayesian Occam), not a hand-coded early bias (rejected by data).
- Per-pair precision = the routing term; real opportunity because no single layer is best for
  everything (retrieval peaks L20 and FADES by L35; computation only at L35; style early).
- GATE before training: fit the ensemble labeller on the styc cache, compare per-family acc vs
  the best single layer. Train only if it wins. Then: soft-DPO vs ensemble / single-L35 / GT
  with today's oracle evals (styc_train.py).
- On-policy variant later: multi-layer confidence-weighted reward read from the FROZEN base of
  emitted text (no forging risk), with the guarded-srloo lessons.

## The three-arm result (2026-07-30 evening, results/styc_train_*.json, fixed parser)

Dose-response: labeller conflict-competence -> policy conflict ranking:
early L10 0.06 -> 0.06-0.10 | late L35 (natural fit) 0.32 -> 0.50 | gt 1.0 -> (see history).
Generations stayed ~intact in all arms (0.98 correct; base 1.0) — preference corruption
precedes behavioural corruption; the corrupted implicit reward would already mis-select in
best-of-N. KEY NUANCE: the natural-protocol L35 labeller is style-first (conflict 0.32), not
the 0.97 that conflict-privileged validation extracts — information existing at a depth does
not mean a natural fit finds it. Depth capability necessary, diet/validation coverage
sufficient-maker.
