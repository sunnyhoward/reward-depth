# The phase-8 mechanism on britishness: it installs, it does not beat DPO, and it splits by family

*2026-08-14. `Qwen3.5-4B`, read block 20 pooled, `brit_dose20.jsonl`, seed 0, 600 steps,
LoRA 0-31, DPOP floor 1.0. Scored on `false_friend` and `style` at ckpt100/300/600. All eleven
arms — base, the replay 2x2, and six meandiff cells — judged in ONE blind `Qwen3-32B` invocation,
528 items per family. Scripts: `probefix/run_pf4b_meandiff.sh`, `pf_train.py MODE=meandiff`.*

## 0. What was ported and why

`results_phase8.md:266` (`pooled_margin`) is the only arm in this project that installed a
preference by optimising activations directly: conflict .845, `gen_correct` .891, no forging,
stable plateau. Phase 9 ported it to UltraFeedback and got a **behavioural null** (acc at chance
at every checkpoint while the meter was driven past its target), and diagnosed it:

> the margin mechanism works exactly when the direction is causally load-bearing; pooling fixed
> forging, not direction quality.

Britishness had never been tried. What probefix has been running instead — `MODE=probe` — is a
continuously-refitted probe read at the **completion end** with the gradient through a scalar
score: the objective family phases 1-5 established forges, and probefix reproduced the signature
(decodability rises everywhere, output ranking and generation stay at base).

Three differences, each load-bearing per phase 8: **pooled** reads, a **lag-1** direction, and a
**saturating** hinge.

## 1. It installs. It is not the UF null.

British score among engaged items, judge 0-100, against base:

| family | arm | Δ vs base | SEs |
|---|---|---|---|
| false_friend | MD r0 @100 | **+29.4 ± 7.9** | +3.7 |
| false_friend | MD r0 @600 | **+30.3 ± 8.2** | +3.7 |
| false_friend | MD r1 @100 | +25.3 ± 7.6 | +3.3 |
| false_friend | MD r1 @600 | +22.5 ± 8.0 | +2.8 |
| style | MD r0 @100 | +6.5 ± 3.4 | +1.9 |
| style | MD r1 @600 | +9.5 ± 3.2 | +3.0 |

On `false_friend` the mechanism moves free generation from 36.0 to 66.4 — a real behavioural
install from an activation objective, on a naturalistic task. **That is the first time the
phase-8 mechanism has installed outside the styc oracle.**

Nothing degenerates. Coherence is 94.9-97.1 on `false_friend` (base 93.1, P1 93.9) and 81.8-84.0
on `style` (base 78.2). The forging channel was open by design and the text stayed healthy.

## 2. It does not beat DPO, on either family

| family | meandiff best | P1 (plain DPOP) | C1 (two-stage) | MD − P1 |
|---|---|---|---|---|
| false_friend | 66.4 | **78.8** | 75.9 | −12.5 ± 7.4 (−1.7 SE) |
| style | 56.8 | **73.4** | 67.4 | −16.7 ± 2.8 (**−6.0 SE**) |

Ordinary DPO at the output is better on both, decisively so on `style`. Whatever the activation
objective buys, it is not install strength.

## 3. The result worth keeping: a WITHIN-TASK dissociation

Expressed as the fraction of DPO's gain the mechanism recovers:

| family | base | meandiff | P1 | fraction of the DPO gain recovered |
|---|---|---|---|---|
| **false_friend** (terminology) | 36.0 | 66.4 | 78.8 | **71%** |
| **style** (register) | 47.3 | 56.8 | 73.4 | **36%** |

Same model, same read layer, same objective, same run — and the mechanism recovers twice as much
of the terminology gain as the register gain.

Phase 9's formulation had to be established ACROSS tasks: the pooled direction is load-bearing on
styc and is the style-legible-but-inert one on UF. Here the same contrast appears **within one
task, between two families, in a single training run.** The pooled direction at block 20 carries
which *word* to use and carries register much more weakly — so driving it moves terminology and
barely moves register, while output DPO moves both.

That is a cleaner experimental design for the claim than the cross-task version, because model,
data, objective, layer and seed are all held.

## 4. The trajectory, and why the checkpoints mattered

The hinge saturates hard and early. By step 200: `pref` exactly 0.0000, `frac_sat` 1.00, `proj`
**+64.7** against M0 9.29 — seven times the target, and no preference gradient at all after that.
Raw ranking peaks near step 100 (`legacy` .087 -> .493) and settles back to ~.24 by 600 while the
implicit columns stay .84-.96.

Scoring was extended to ckpt100/300/600 on the strength of that trajectory, since phase 8's
`shaped` arm peaked at ~125 and banked only the over-optimised adapter. **In the event the
checkpoint barely matters** — `false_friend` reads 65.4 / 61.5 / 66.4 across 100/300/600 — so the
ranking peak at step 100 does not correspond to a generation peak. The check was cheap and the
null result on it is itself worth recording: on this task the over-optimisation that ruined phase
8's shaped arm does not appear.

Replay makes little difference and what there is favours OFF on `false_friend` (66.4 vs 58.5) and
ON on `style` (56.8 vs 50.9) — inconsistent, both within noise, and consistent with 0814's
finding that the term is not doing much either way.

## 5. Scope limits

1. **Single seed per cell**, as everywhere in this project.
2. **One depth.** Block 20 only. The depth ladder — which is the experiment this mechanism was
   ported to make possible — has not been run. `false_friend` at 71% recovery is the first
   objective in probefix with enough behavioural headroom to make such a ladder readable.
3. **M0 was calibrated, not copied** (`M0_MULT=1.15`, M0 9.29 against a base projection of 8.08).
   styc's M0=4.0 would have been satisfied at initialisation here. The ratio was matched to the UF
   port's 1.14x; no sweep over it was run, and the 7x overshoot suggests the saturation point is
   not where the interesting dynamics are.
4. Two families. `false_friend` (n=48, SE 6-8) carries the terminology half alone.
5. The comparison arms are the same-environment retrains from the replay sweep, not the banked
   0813 adapters.
