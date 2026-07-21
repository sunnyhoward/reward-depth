# Phase 3 — UF v2 postmortem, and soft-label DPO from the frozen probe

*2026-07-21. Follow-up to `results_phase2.md` §5 (the weak UF RL result). Code:
`uf/uf_probe_rl.py` (fixed), `uf/uf_soft_dpo.py`, `uf/uf_readpos_diag.py`, `uf/uf_tulu_dpo_eval.py`.
All big-N numbers: 350 held-out pairs, reference-corrected implicit-reward accuracy, SE ≈ 0.022–0.027.*

## Headline

**Soft-label DPO from the frozen L12 probe matches ground-truth-label DPO on installed preference
(0.800 ± 0.021 vs 0.805) with a far better Goodhart profile** — chosen-side likelihood *preserved*
(Δlp +0.7 vs −9), margin inflation +9.8 nats vs +33, trained on only the 3,000 probe-fit pairs.
The probe's compressed, uncertainty-aware representation of the preference is as installable as the
raw labels, and installs more cleanly. The RLOO-from-probe arm's weakness (v2's 0.571) was a
harness artifact, not a probe deficiency: four stacked bugs, each isolated and fixed or falsified
below.

| arm | acc_implicit | margin (nats) | Δlp chosen / rejected |
|---|---|---|---|
| DPO baseline (12k pairs, hard labels) | 0.805 | +33 | −9 / −30 |
| **soft-DPO from probe, ckpt200** | **0.800 ± 0.021** | **+9.8** | **+0.7 / −9.2** |
| soft-DPO final (400) | 0.760 ± 0.023 | +8.8 | +0.2 / −8.6 |
| official Tulu-3-8B-DPO (left-trunc eval) | 0.623 ± 0.026 | +18.3 | −67.8 / −86.1 |
| RLOO v3 (all fixes, aborted @100) | 0.489 ± 0.027 | +0.3 | +3.6 / +3.3 |
| RLOO v2 (historical) | 0.571 ± 0.026 | +1.3 | +5.7 / +4.4 |

Soft-DPO decays past ckpt200 (0.800 → 0.760 at 400) — the phase-2 early-stopping theme again.

## 1. The v2 postmortem: four stacked defects

**(a) Read position: the reward was read at a `<pad>` token.** `generate()` left-pads the prompt
but right-pads completions to the batch max, so `hidden[:, -1]` is a `<pad>` position for every
rollout shorter than the longest in the batch (≈ (K−1)/K of them). Worse, Tulu's `<pad>` (id
128256) is an *appended, untrained* embedding: ‖e‖ = 1.28 vs 0.21 for the `<|end_of_text|>`
sentinel the probe was fit on (rows 128256–128263 all sit at ‖e‖ ≈ 1.26–1.30, the signature of
never-trained init). Measured on-policy (`read_diag`, v3 run): **corr(old read, correct read) =
−0.16…+0.05 ≈ 0** — v2's reward was noise, not attenuated signal. `uf_readpos_diag.py` on natural
pairs: pairwise acc 0.793 (correct) → 0.595 (one trailing pad). v2's truncation mask compounded
it: masking capped rollouts *selects for* the pad-read ones. Fix: decode each rollout and
re-render through `render_full`, reading the true eos sentinel (`rollout_feats`; ported from the
source repo's `onpolicy_feat`, which had already found and fixed this). Capped rollouts become
scoreable (an abruptly-ending response, legitimately low reward) instead of discarded.

**(b) Truncation side.** The uf scripts never set `truncation_side`; the default (`right`) cuts
the completion *end* — the probe's read position — on ~6% of pairs at MAX_LEN=1024. The source
repo set `left` explicitly. Now set everywhere (helpers + all uf scripts).

**(c) Length confound (Stage A).** UF's chosen side is longer (268 vs 193 response tokens; chosen
longer in 61% of pairs); length alone scores 0.619. The sweep had no control (the source repo's
IPW matching had been dropped in the port). **Result of the matched re-sweep: the plateau
survives — 0.799 from L\*=12 vs 0.803 from L11 unmatched** (IPW, bucket 16, Kish ESS 13,056/15,283).
The deep probe was not riding the length cheat; the early layers partly were (L0–2: 0.70 → 0.66,
so the early-vs-plateau gap *widens* under matching). `train_bayes_head` now takes IPW weights
(unweighted path verified bit-identical).

**(d) Reward-scale compression (falsified as the binding constraint).** `probe_reward` scored
*absolute* features `f/sd` uncentered; the head was fit on *differences*, where the mean cancels.
The uncentered mean inflates the predictive variance (s2 = 648 vs 38 centered ≈ 48 fit-time), and
`√(1+s2)` + the pessimism LCB squash the reward spread ×3.3 (chosen-vs-rejected gap 0.26 → 0.08;
ranking unaffected: 0.808 vs 0.810). The A/B testbed never hit this because `sampled_rl_step`
scores `g(f_right − f_cand)` — a difference read (`helpers.py`). Fixed (center with the Stage-A
pooled mean) and ablated: **v3 (squashed) and v4 (centered) are both flat** — reward ~0.42–0.49
with no trend over 50–100 steps, acc_implicit ~0.5 at big-N (v3 ckpt100: 0.489 ± 0.027). So scale
was real but not binding; with (a)–(c) fixed, RLOO at this throughput still does not move the
reward.

## 2. Why RLOO stalls and dense coupling doesn't

Per step the RLOO arm gets 32 sequence-level advantages over ~200-token completions (4 prompts ×
8 rollouts) at lr 5e-5 — against Goodfire-scale RL throughput this is starvation, and the A/B arm
that worked had few-token completions (trivial credit assignment), lr 1e-4, and a 3B model. The
discriminating experiment (methods.md §3, first proposal): **soft-label DPO from the frozen
probe** — per-pair soft label p = Φ(z/√(1+s²)) from the head's posterior predictive on its native
difference features, loss −[p log σ(βΔ) + (1−p) log σ(−βΔ)] on the implicit-reward margin. Dense,
offline, zero sampling variance, unfakeable (loss reads emitted-text likelihoods). Config mirrors
`uf_dpo_train.py` exactly (β 0.1, lr 5e-5, 400 × 4×4, LoRA r16); labels: mean p 0.739, 15.4%
side against the dataset, 47% genuinely soft (0.2–0.8).

It installs immediately (in-run n=128 evals): acc_implicit 0.695 @ 50, probe-agreement at the
probe's own ceiling (0.781 vs 0.791) by step 250, big-N 0.800 @ ckpt200. **Conclusion: the probe
signal was never the problem; the on-policy REINFORCE estimator at this throughput was.** The
positive-side profile (Δlp_chosen ≈ 0 throughout, vs the baseline's −9) is consistent with the
soft labels acting as a regularizer: the probe-uncertain half of the pairs pushes both ways and
caps margin growth (+9.8 vs +33 nats).

## 3. Reference points

- **Official Tulu-3-8B-DPO scores only 0.623 ± 0.026 on this split** (trained on the full
  tulu-3-pref-mixture, of which UF is one component; far more compute than any arm here). The
  in-domain 400-step LoRA DPO's 0.805 is therefore substantially *dataset-specific* fit — an
  in-domain/general gap of ~0.18 that big DPO training does not close. Its displacement is also
  extreme (−68/−86 nats). Caveat: different data, scale, and full-FT vs LoRA — an upper
  *reference*, not a matched arm. (Right- vs left-truncation eval: 0.614 vs 0.623 — no material
  difference.)
- Base SFT raw ranking: 0.409 (length-confounded, matches phase-2's 0.403).

## 4. Artifacts

- `/workspace/uf_softdpo_{ckpt200,ckpt400,lora}`, `uf_softdpo_history.json`
- `/workspace/uf_probe_rl_{v3aborted,v4aborted}_history.json` (the scale ablation pair),
  `uf_probe_rl_v3aborted_ckpt100`
- `/workspace/uf_probe_feats_lenmatch.npz`, `uf_probe_curve_lenmatch.json` (matched sweep)
- `/workspace/uf_readpos_diag.json`, `uf_bigN_softdpo.json`, `uf_tulu_dpo_eval.json`
  (+`_righttrunc` variant), logs `/workspace/{rl_v3,rl_v4,softdpo,eval_pass,stageA*}.log`

## 5. Open items

- **The depth claim proper**: soft-DPO from an L31 (top) probe vs L12 — same script, LSTAR
  override; the phase-3 result shows L12-probe labels ≈ dataset labels, but the depth
  *differential* (Occam: earlier attach ⇒ less idiosyncrasy absorbed, gentler Goodhart) is untested
  on UF. Also worth an unmatched-probe arm (does length-matching the labels change what installs?).
- Early stopping for soft-DPO (ckpt200 > final; sweep the knee).
- RLOO with real throughput (larger K·batch, lr 1e-4, micro-batched KL pass) — now that the
  reward path is verified, the harness question is honest to ask.
- Judge-based rollout comparison (paired, both-order, length-controlled) for soft-DPO ckpt200 vs
  DPO baseline — implicit-reward acc can't see generation quality.
- Port `rollout_feats` + centered reward into `uf_hybrid.py` (still has the v2 read bug at its
  line 158) and `uf_spread_diag.py` (line 73); both also still load the unmatched cache.
- Cache-key footgun: `uf_probe_feats_lenmatch.npz` does not encode `UF_LEN_BUCKET`/`UF_POOL` —
  delete it when changing those knobs.
