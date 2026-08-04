# Continue here (written 2026-08-04, second session — evening)

**Read `eagle/RESULTS.md` §17-§19 first** (that file's §8-§16 are the morning session; §1-§7 are
08-03). One paragraph: the frozen-head **L-sweep is done** (single seed on user direction; L12
keeps 3 seeds) — stage-1 installs at EVERY depth at the first eval, and depth buys speed at the
price of stability: L24 styc installs perfectly at step 5 and is dead by 15; styc L32 is the one
true failure (damage outruns install); brit L32 installs at 25 then degenerates to token spam.
"Checkpoint finely, select early" is load-bearing at L24+. **UF is closed as a fully-diagnosed
triple negative** (§18-§19): the quality preference has no token-likelihood footprint (.578/token
ceiling for the FULL model; pooled probe reads the same h_12 at .79 — probes read what LM heads
cannot say); a competent readout lets layers 0..12 fit the margin (.742) but the fit lives in the
lower-edit x upper-stack interaction (invisible at L12: .35/.39) and is behaviourally null (7B
judge 12:12:24). EAGLE's scope = token-footprint preferences (style, dialect, format — real
datasets included); holistic-quality preferences are out of reach of likelihood-margin training
at any write depth in this regime.

## EAGLE queue

1. **Finish the K-FAC leash test** — the one experiment left mid-flight when the box was killed.
   Everything is built: `replay-kfac-ewc` installs from the repo dir; `eagle_dpo.py` has
   `KFAC_LAMBDA`/`KFAC_DIR`; the test driver is `/workspace/sweep_kfac_l24.py` (copy below if
   lost — styc L24, lambda {1,10} vs the existing lambda-0 collapse; success = the step-5 install
   surviving to 50). Factor estimation was ~half done after ~3h CPU: 175 modules (layers 0..24,
   all 7 projections) was TOO MANY — restart with attention-only targets (q,k,v,o: ~70MB/layer
   vs 1.6GB) or cap `--max-positions 20000`; that alone should cut it to ~20-30 min. Replay
   shards are uploaded next to the bundle (mixed + styc-prompt-seeded, 768 seqs each; the merge
   needs the renumbered prompt shard or distinct `--seq-start`). If the leash widens the L24
   window, extend to brit L24/32 and consider it for the decay problem generally.
2. **Remeasure §1's encoding-depth table with the tf head** (unchanged from morning; the repo's
   core claim still rests on an mlp-head measurement).
3. **The brit residual** (§16): stage 1 plateaus at brit_rate .70-.85. Per-prompt breakdown of
   remaining American markers; the targeted argmax metric (item 5, morning list) is ~15 lines.
4. **Upweight guard rows** to 1:1 before concluding truth and dialect are inseparable.
5. Parked: **UF in all forms** (dataset pairs AND the probe-margin idea — the direct-from-probe
   family Goodharts; user explicitly declined the on-policy synthesis). **Stage 2 in all forms**
   (delta/head measured dead §11-§12; pairwise safe-but-inert; the entropy-gate design in §12 of
   the 08-04 morning notes remains the insurance IF a genuine silent install ever appears —
   base-entropy-gated, frozen-base side, tau~1.0).

## Standing traps (additions this session)

- This volume has **stale-file-handle dentries** (`/workspace/.env`, one HF blob): unfixable
  in-container. HF cache moved to `/workspace/.hf_home2` (token file inside). Check
  `vast-capabilities | jq .instance.workspace_is_volume` on the next box before trusting.
- `replay-kfac-ewc merge` requires disjoint seq_ids across shards (use `--seq-start`, or
  renumber); `estimate --resume` errors if no checkpoint exists yet.
- 175-module dense estimation on CPU ≈ hours. Attention-only or `--max-positions` first.
- The 7B judge produced 50% position-swap disagreement (24/48 ties) — fine for a null, but any
  POSITIVE judge claim needs the 32B (phase-9 protocol) before it goes in RESULTS.
- (Morning list still applies: read raw generations; brit step-0 accs are 0 by construction;
  head_acc means nothing with a trainable head; CKPT_EVERY<=10 at L>=24.)

## Restoring the repo

**Private HF repo `sunnyhoward/reward-depth-backup`**, bundle **`reward-depth-0804b.bundle`**
(latest — L-sweep, UF closure §17-§19, K-FAC wiring, all histories in `results/runs/`; prior
states: `-0804` morning, `-0803`). Replay shards `kfac-shard-mixed.jsonl` /
`kfac-shard-prompt-renum.jsonl` sit next to it.

```
hf download sunnyhoward/reward-depth-backup reward-depth-0804b.bundle --token $HF_TOKEN
git clone reward-depth-0804b.bundle reward-depth
```

Fresh-box setup: `uv pip install transformers peft datasets scikit-learn matplotlib accelerate
huggingface_hub` into /venv/main, plus `uv pip install -e reward-depth/replay-kfac-ewc` for the
K-FAC line. HF_TOKEN into `${WORKSPACE}/.env` (never commit). Ask for 100GB+ disk.
Models: **Qwen/Qwen2.5-3B** (EAGLE testbeds). UF is parked — Tulu/judge models not needed unless
reopened.

**Artifacts that die with the box** (all regenerable): eagle heads (`eagle_head.py`, HEAD_ARCH=tf,
~5 min; replay corpus `eagle_replay.py` ~5 min; NOTE the canonical `eagle_head_tf_L*.pt` slots
currently hold the REPLAY-distilled heads — styc-data originals in `*_styc.pt`), brit heads
(lazy-distilled on first `eagle_brit.py` run per L), all stage-1 adapters (a working styc cell is
~4 min: `HEAD_ARCH=tf FREEZE_HEAD=1 FACTOR=style L=12 STEPS=50 CKPT_EVERY=5 EVAL_EVERY=5 python
eagle/eagle_dpo.py`), K-FAC factors (restart estimation as in queue item 1), the lower12 adapter
(`LORA_LAYERS=0-12 DPO_STEPS=200 RUN_TAG=lower12 python uf/uf_dpo_train.py`, ~40 min, only if §19
needs revisiting — histories/gens/judgments are all in `results/runs/`).

## Standing user directions (accumulated)

- **(2026-08-04) All experiments stay on the EAGLE line.** UF parked (user call, evening session:
  "keep going with the eagle idea"); the on-policy synthesis was offered and declined.
- Single seed per cell is acceptable for sweeps ("don't care about seeds too much"); seeds only
  where a number is load-bearing.
- Probes GENERAL (preference labels only); no training direct from probes (Goodharts — reaffirmed
  by user this session).
- Kill flat arms early once the diagnosis is clear (UF frozen-head arm killed at 120 on user
  call; diagnostic run instead — the right move, do it again).
- Read raw generations before believing any metric (standing; two more collapses caught by it
  this session: brit L32 spam, styc L24 death).
