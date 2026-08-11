# Where this got to — start here tomorrow

*2026-08-11. Read `RESULTS.md` for the evidence, `NOTE.md` for the design. This file is the
orientation and the honest list of what is and is not established.*

## The one-paragraph version

Trained the encoder against a continuously-refitted linear probe at L\*, with a replay anchor at
the output, then handed it to ordinary DPO — against plain DPO on all layers. Two models
(Qwen3.5-2B at block 12, Qwen3.5-4B at block 20), dosed britishness with the guard at 20%, single
seed. **Stage 1 raises linear decodability everywhere including the final block and installs
nothing at the output, on both models, even when the probe already reads the composite rule at
0.92.** Whether the two-stage pipeline beats plain DPO depends on the attach point: at 2B it lost
badly, at 4B it matches on install and holds the guard better. On both models it converts ranking
into *generation* worse than the control.

## The two claims that survived everything

1. **The surrogate gap is real and is not forging.** An independently refitted probe on the trained
   model reads guard 0.62 → 0.76 at L\* and 0.58 → 0.74 at block 23 (2B), while output ranking and
   free generation stay at base. The 4B replicates it with a near-ceiling probe. `pf_audit.py`.
2. **Every arm's largest weight change lands at L\*, including plain DPO**, which was never told
   where L\* is (2B: A peaks at block 12 after subtracting the replay-only null). What differs is
   shape — probe-attached writes a delta function there, output DPO a broad peak — and only the
   broad one produces behaviour. `pf_audit.py` + `R_replay_only`.

## What I retracted during the session — do not resurrect these

| claim | why it died |
|---|---|
| "A single direction can't express 'British unless false'" | `brit_guard_dose.py`: one head reads guard 0.843 / install 0.944 at 0.2 dose |
| "Stage 1 trains against a dialect-dominated direction" | `pf_axis_decomp.py`: probe leans toward truth (cos +0.203 vs +0.112); truth axis amplified x4.41 vs dialect x1.48 |
| "Decoder-only DPO cannot install truth_dialect (0.360)" | Stopped inside an oscillation. At 600 steps D reaches 0.85 (2B) / 0.89 (4B) |
| "C1 destroyed the truth axis at the readout (x0.15)" | Base-fitted frame measuring a rotation; C1 prefers truth by +66.9 nats |
| "The guard is paid for" | Nobody loses truth. Truth preference stays 0.96–1.00 and its pull GROWS 2.5–11x |

**The mechanism for stage 1 costing generation is still not established.** One hypothesis has
support and is unverified: stage 1 makes the preference more linearly separable, which is a
*discriminative* property, so stage 2 satisfies a *ranking* objective while moving the generative
distribution less. Checkable consequence, and it holds — C1_4b reaches equal-or-better ranking than
A_4b at lower drift (replay KL 0.147 vs 0.209 at step 300).

## The thing I would look at first tomorrow

**`results/probefix/ROLLOUTS_Qwen3.5-4B.md` and `..._2B.md`.** Every behavioural number here is a
marker-count ratio, and it scored **A_4b@600 at brit_rate 0.988 with diversity collapsed to 0.46**
(base 0.87, C1_4b 0.90). The winning arm on the headline meter is the one degrading most on a meter
nobody was watching. The rollouts include six **guard prompts** where writing British would require
a falsehood — read those by eye, because nothing in this study scores them automatically.

## Open, in the order I would do them

1. **Seeds.** Everything is single-seed and the guard bucket is 50 rows. The 4B "C1 holds the guard
   better" result (0.940 flat vs A's 0.780, swinging 0.68–0.94) is ~1.5 SE on checkpoint spread.
2. **A diversity meter as a first-class column**, not a field in a JSON. The collapse above would
   have changed how I read every table.
3. **The by-layer weight analysis on the 4B is not done.** `R_4b_replay_only` finished (600 steps,
   ckpt100..600) so the step-matched null exists; it needs `pf_audit.py` over B4 / C1_4b / A_4b /
   D_4b with `OUT_DIR=/workspace/probefix4b/audit`, then the subtraction. ~15 min.
4. **Fix the 2B §3.5 magnitudes.** They subtract R@300 from A@600 — not step-matched, so A's
   preference-attributable numbers are inflated by 300 steps of unsubtracted replay. Shapes are
   unaffected.
5. **A generation-side guard meter.** Needs a fact checker; the prompts are generic so free samples
   need not touch the fact under test.
6. **Attach depth as a variable.** Both runs used one depth. The 2B-vs-4B difference is confounded
   between model scale and attach quality (0.68 vs 0.92 decodable) — one 2B run at a deeper attach
   would separate them, and it is the cheapest informative run left.

## Layout

- `pf_probe_curve.py` stage 0 · `pf_train.py` all arms · `pf_audit.py` ranking/drift/ΔW/fresh probe
- `pf_guard_axes.py` un-crosses the guard into truth and dialect · `pf_axis_decomp.py` what the
  probe encoded and what got amplified · `pf_rollouts.py` samples · `pf_report.py` figures
- `run_pf.sh` 2B queue · `run_pf600.sh` step-matched reruns · `run_pf2.sh` nulls/ablations ·
  `run_pf4b*.sh` the 4B port
- Artifacts: `results/probefix/` (2B) and `results/probefix4b/` (4B), `runs/` holds every history,
  audit, guard-axis and behaviour JSON. Adapters are on `/workspace` only and do NOT survive.
