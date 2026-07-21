# reward-depth

**Where should the reward signal attach to a language model — and what does the attachment depth
buy you?** We train policies against a linear (Bayesian) *reward probe* reading hidden state at
layer L, and study how the installed preference changes with L: its generalization, its
coherence, and how gracefully it Goodharts under over-optimization.

The working hypothesis (Bayesian Occam's razor): attach the reward at the **earliest layer where
the preference is fully decodable**. A head there can only read the simple, semantic core of the
preference — dataset idiosyncrasies that are only decodable late can't enter the reward — so the
policy trained against it should overfit the preference less and degrade more gracefully than
output-level methods (DPO / top-attached reward models).

**Headline findings so far** (details: `results_phase1.md`, `results_phase2.md`, `methods.md`):

- **Self-read backprop from the probe is gameable everywhere we tried it** (top layer, mid-depth,
  3B and 7B, frozen or co-adapted head): the policy forges the probe's feature instead of
  changing behavior. Every working method couples the probe to behavior through *emitted text*.
- **Anchored RL-from-probe works**: RLOO with the probe scoring frozen-base reads + a DPOP
  anchor installs the full preference at the top *and* at the elbow, with less proxy inflation
  and far less collateral damage than DPO (see `results/plots/fig3_final_bars.png`).
- **Two cheap-escape attractors dominate naive runs** on menu-format testbeds: the
  *letter-policy* attractor (answer "A" always) and *likelihood displacement* (both completions
  sunk, mass drains off-menu). The DPOP anchor addresses both; `fracA` is logged at every
  checkpoint so the letter attractor can't hide.
- **On UltraFeedback**, preference decodability plateaus at **L11/32** of the frozen SFT model,
  at an accuracy (0.80) equal to what a 400-step DPO run installs — the preference DPO trains in
  is already linearly present a third of the way up the untrained model
  (`results/plots/fig5_uf_probe_rl.png`). The plateau survives length matching (0.799 @ L12,
  vs a 0.62 length-only cheat floor) — see `results_phase3.md`.
- **Soft-label DPO from the frozen L12 probe matches ground-truth DPO on UF** (implicit acc
  0.800 ± 0.021 vs 0.805) with far less collateral: chosen-side likelihood preserved (Δlp +0.7 vs
  −9), margin inflation +9.8 vs +33 nats, using only the 3k probe-fit pairs. The weak RL-from-probe
  result (0.571) was a harness artifact — a pad-token reward read (reward ≈ noise, corr ≈ 0 with
  the true probe score) plus three smaller defects, each isolated in `results_phase3.md`. For
  scale: the official Tulu-3-8B-DPO checkpoint scores 0.623 on this split.

- **Decision-position probes (phase 4)**: moving the read upstream of the decision. Pure
  activation-only training (no likelihood terms) genuinely installs the preference but never
  stabilizes (oscillates 0.18–0.65 for 600 steps; frozen head is fully forged even upstream);
  pure candidate-REINFORCE letter-locks (replicating phase 2). The **two-head hybrid** —
  activation margin from a decision-position *plan-reader* below L\*=23, exact-expectation
  REINFORCE from a completion-end *outcome-judge* above it — installs the complete preference:
  flip 1.000 letter-balanced, OOD 1.000/0.847, know 1.000, easy 0.99, off-menu 0.000, stable.
  Each half is load-bearing (six-arm ablation + failure taxonomy: `results_phase4.md`,
  `decision_probe.py`).

## Layout

- `probe_vs_dpo.ipynb` — the A/B testbed experiment, arm menu: `dpo`, `rl_top`, `rl_elbow`,
  optional `hybrid`. Pick arms in the `RUN` list; figures generated at the end.
  `probe_vs_dpo.out.ipynb` is the executed record (all three default arms validated).
- `helpers.py` — model/data loading, the A/B wrongness testbed, per-layer Bayesian probes,
  LoRA policy, training signals (`margin_step`, `dpo_step`, `sampled_rl_step`), head filtering,
  eval/Goodhart instrumentation.
- `hybrid_deep.py` — standalone hybrid (margin backprop ≤ L + candidate/RLOO REINFORCE > L),
  A/B testbed; all knobs from the phase-2 reconstruction (`--rl_mode`, `--anchor_mode`, …).
- `uf/` — UltraFeedback (Tulu-3-8B-SFT, `allenai/ultrafeedback_binarized_cleaned`):
  - `uf_dpo_train.py` — DPO baseline (LoRA; saves adapter + merged model + history)
  - `uf_probe_rl.py` — per-layer probe sweep → plateau layer L* → anchored RLOO from the frozen
    probe (truncation-masked rewards, pessimism LCB, KL-in-reward, checkpoints)
  - `uf_soft_dpo.py` — soft-label DPO from the frozen probe (the working method; phase 3)
  - `uf_bigN_eval.py` — large-N held-out implicit-reward accuracy for saved checkpoints (`CKPTS`/`OUT` env)
  - `uf_tulu_dpo_eval.py` — official Tulu-3-DPO checkpoint as a general-DPO reference point
  - `uf_readpos_diag.py` — quantifies the pad-read bug (probe acc at sentinel vs trailing pads)
  - `uf_spread_diag.py` — reward-spread diagnostic (within-prompt spread vs pair gap, truncation)
  - `uf_hybrid.py` — UF port of the hybrid (see header caveats: the margin half is the gameable
    coupling; it exists to *measure* whether it adds anything over pure RL)
- `results/` — run JSONs, logs, figures (`plots/`), LoRA adapters (`adapters/`, gitignored).
- `attic/` — superseded one-off drivers, kept for provenance.
- `isotropy_check.py` — unembedding row-cloud isotropy measurement (phase-1 premise). CPU.

## Testbeds

**A/B wrongness (synthetic oracle).** Train the model to prefer *wrong* answers to 2-choice
questions it answers ≥99% correctly (Qwen2.5-3B; 2,030 questions × 2 formats). Deterministic
oracle, exact targeted-flip metrics, full Goodhart instrumentation. Caveat learned the hard way:
the relational (letter-randomized) preference makes letter policies a strong attractor — always
check `fracA` and per-type flips before believing any aggregate.

**UltraFeedback (realistic).** Tulu-3-8B-SFT + AllenAI binarized-cleaned pairs (score-margin
filter, by-prompt split). The probe defines the reward; DPO is the baseline; evaluation via
reference-corrected implicit-reward accuracy on held-out pairs (raw log-prob ranking is length-
confounded: base raw acc 0.40).

## Reproducing

Notebook: open `probe_vs_dpo.ipynb`, edit `RUN`, run top to bottom (features/probes are
disk-cached after the first run). UF: `python uf/uf_dpo_train.py`, then `python uf/uf_probe_rl.py`
(builds the shared feature cache), then optionally `uf/uf_hybrid.py`; evaluate with
`uf/uf_bigN_eval.py`. All UF model artifacts are written under `/workspace/` (not the repo).

## Provenance

Extracted from a larger study (`preface` repo): a wrongness-preference sandbox on Qwen2.5-7B and
an UltraFeedback replication on Llama-3.1-Tulu-3-8B-SFT. The phase-2 hybrid archaeology
(`results_phase2.md` §4) reconstructs that study's `04_compare_L24` figure: the working
ingredient was candidate-based REINFORCE above the probe layer; its collateral profile was
never fully reproduced and is attributed to early stopping pending a 150-step rerun.
