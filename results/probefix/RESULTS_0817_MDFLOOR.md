# An absolute anchor for the activation objective: the mechanism is fixed, the install is not

*2026-08-17, last experiment of the session. `MD_FLOOR=1.0` adds `relu(ref_chosen·u − chosen·u)` —
DPOP's floor applied to the **chosen side's own projection** in SD-normalised units, reference taken
adapter-disabled. Two arms: `MD_L20_floor` (single attach) and `MDSTACK_floor` (multi-depth severed,
floor computed per segment inside the severed forward). 300 steps, seed 0, `Qwen3.5-4B`, replay off,
LoRA 0–31. 768 items (8 arms × 96) judged blind by 16 Claude subagents, shuffled across arms and
families, with `MDSTACK_s300`, `MD_L20_s300`, `P1_r1` and `base` in-pass. Diagnostic re-run:
`results/probefix/mdsides_floor.json`.*

**Why it was run.** `RESULTS_0817_MDSTACK.md` §2 measured the activation objective winning its margin
the way plain DPO wins its: `MD_L30` doubled its projection with the **chosen side falling 11.7** and
the rejected side falling 48.7, at unchanged read norms. `relu(M0 − proj)` constrains only the
difference; the λ floor acts on output logps, so the activation term had no absolute anchor. This
adds one.

**Verdict: the mechanism is repaired and it buys nothing.** Δchosen at L30 goes from **−51.7 to
+59.5** — a 111-point swing, exactly as designed — and the judged install does not improve. At 300
steps the arms are wrecked. The one bright spot is early-stopped single attach, at 1.9 SE.

---

## 1. The mechanism is fixed (`pf_mdsides.py`, base-fitted frame, 128 pairs)

| arm | read | proj (base) | Δchosen | Δrejected | ‖read‖ (base) |
|---|---|---|---|---|---|
| MDSTACK (no floor) | L30 | +45.0 (+25.0) | **−51.7** | −71.7 | 31.3 (30.2) |
| **MDSTACK_floor** | L30 | +23.0 (+25.0) | **+59.5** | +61.5 | 33.8 (30.2) |
| MDSTACK_floor | L6 | +59.3 (+21.2) | +93.0 | +54.9 | 33.9 (33.6) |
| P3 (DPOP at output) | L30 | +8.9 (+25.0) | +138.4 | +154.5 | 32.0 (30.2) |

The sign flips where it was gamed, and the profile now resembles the output-attached control. Note
the honest counterpart: `proj` at L30 **falls back to base** (+45.0 → +23.0). The floor buys absolute
movement by giving up the difference — the same trade DPOP makes in logit space. L18 also goes
negative (−27.9), so the repair is not uniform across segments.

## 2. And it wrecks the text

Chat-template leakage (`pf_leakage.py`) and judged coherence:

| arm | leakage | coherence ff / style |
|---|---|---|
| MD_L20_s300 (no floor) | 0.146 | 79.4 / 66.3 |
| MDSTACK_s300 (no floor) | 0.094 | 71.3 / 76.0 |
| **MD_L20_floor_s300** | **0.625** | **29.3 / 48.1** |
| **MDSTACK_floor_s300** | **0.896** (47/48 ff) | **10.4 / 32.1** |
| MDSTACK_floor_s100 | 0.375 | 52.4 / 60.1 |
| **MD_L20_floor_s100** | **0.000** | **78.9 / 84.5** |

**The reason is structural and was foreseeable.** `relu(M0 − proj)` **saturates** — that is what `M0`
is for, and phase 8 identified saturation as the anti-forging resource. `relu(ref_chosen·u − chosen·u)`
does **not**: it pushes the chosen side's absolute projection up without bound, and inflating
activations along a direction is precisely the forging channel saturation was introduced to close.
The floor fixed one unbounded channel by opening another.

## 3. Judged install (unconditional, paired per prompt)

| arm | `false_friend` | `style` |
|---|---|---|
| base | 40.1 | 25.3 |
| MD_L20_s300 | 57.0 | 43.9 |
| **MD_L20_floor_s100** | **65.1** | **48.9** |
| MD_L20_floor_s300 | 60.5 | 44.4 |
| MDSTACK_s300 | 59.8 | **57.5** |
| MDSTACK_floor_s100 | 60.3 | 45.9 |
| MDSTACK_floor_s300 | 51.8 | 53.1 |
| **P1** (plain DPOP at output) | **71.4** | **72.8** |

| contrast | `false_friend` | `style` |
|---|---|---|
| **MD_L20_floor_s100 − MD_L20_s300** | **+8.1 ± 4.2 (+1.9 SE)** | **+5.0 ± 3.0 (+1.7 SE)** |
| MD_L20_floor_s300 − MD_L20_s300 | +3.5 ± 3.5 | +0.5 ± 3.2 |
| MDSTACK_floor_s300 − MDSTACK_s300 | **−8.0 ± 4.0 (−2.0 SE)** | −4.4 ± 3.3 |
| MDSTACK_floor_s100 − MDSTACK_s300 | +0.5 ± 4.2 | −11.6 ± 3.1 |
| **P1 − MDSTACK_floor_s300** | **+19.6 ± 5.3** | **+19.7 ± 2.8** |

- **On the multi-depth arm the floor makes things worse** (−8.0 ff, −2.0 SE), which is what the
  coherence collapse predicts.
- **On single attach with early stopping it is the best mean-diff cell in the project**: +8.1 and
  +5.0 over `MD_L20_s300` with **zero leakage** and the highest `style` coherence of any arm here
  (84.5). At 1.9 and 1.7 SE on one seed, that is *suggestive*, not established — and run spread on
  `style` is ~11 points (`RESULTS_0817_MEANDIFF_REJUDGE.md` §4), larger than the effect.
- **Everything still loses to plain DPOP at the output** by ~20 points.

## 4. What this settles, and the one variant left

The pathology was real, measured, and repairable — and repairing it did not change the conclusion.
That makes the activation-space negative **stronger** than it was before this run, because the
obvious rescue has now been tried and fails for a stated reason: the objective family has two
unbounded channels (the difference, and each side's absolute magnitude) and bounding one opens the
other.

**The one variant that follows from this**: a **saturating** floor, `relu(min(ref_chosen, cap) −
chosen)`, which is the actual analogue of what `M0` does for the difference — bound how far the
chosen side is pushed rather than pushing it without limit. Combined with early stopping (the clean
`MD_L20_floor_s100` cell), that is the version worth one run. Prediction, for the record: it
recovers the s100 numbers at s300 and still loses to DPOP by 15–20 points.

## 5. Scope limits

Single seed; `MD_FLOOR=1.0` unswept (the weight is a free parameter and the s300 collapse may be a
weight problem as much as a design one); Claude judge only, unconditional primary; coherence
comparable within this pass and to the 0814-worded passes. `MD_FLOOR=0` is the default, so every
prior arm reproduces.
