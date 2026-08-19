# The finer `z` grid: `style` saturates at 0.7, and the cost lands on the family that gains nothing

*2026-08-19. `RESULTS_0818_ZSWEEP.md` §2 named this run and did not get to it. `A_L` is the banked
`addon_L20` vector (2560 params, `PREF=dpo`, 300 steps, seed 0) on a frozen `Qwen3.5-4B`; `z` scales
it at generation, so every cell is one generation pass and NOTHING was retrained —
`results/probefix4b_addon_dpo/addon_L20.pt` is committed and `pf_addon.py` loads it. Script:
`probefix/run_pf4b_addon_zfine.sh` (the addon runner that never existed; the 0818 cells were
launched by inline env vars). Judge: `probefix/pf_cjudge.py`, 8 arms × 96 items = **768, 16 blind
agents, 768/768 merged, 0 unparsed** — no truncation instruction, so coherence is comparable to the
0818 passes. `results/probefix_cjudge_zfine/`. Single seed. Unconditional is the primary column.*

| arm | ff | ff coh | style | style coh |
|---|---|---|---|---|
| base | 40.6 ± 4 | 79.6 ± 1.6 | 29.5 ± 2 | 77.5 ± 1.4 |
| `L20_z0.50` | 68.6 ± 4 | 76.6 ± 1.8 | 55.9 ± 3 | 79.0 ± 1.3 |
| `L20_z0.55` | 70.9 ± 3 | 75.3 ± 2.0 | 56.5 ± 2 | 78.2 ± 1.2 |
| `L20_z0.60` | 70.2 ± 4 | 76.5 ± 1.9 | 60.2 ± 2 | 79.1 ± 1.2 |
| `L20_z0.65` | 69.9 ± 3 | 74.5 ± 1.9 | 63.0 ± 2 | 76.4 ± 1.3 |
| **`L20_z0.70`** | 67.1 ± 4 | **69.2 ± 2.5** | **68.6 ± 2** | 76.1 ± 1.5 |
| `L20_z0.75` | 67.6 ± 4 | 69.2 ± 2.4 | 68.5 ± 2 | 74.2 ± 2.2 |
| `P1_r1` (plain DPOP) | 67.8 ± 4 | 76.1 ± 2.3 | 68.2 ± 2 | 77.7 ± 3.0 |

---

## 1. `style` has a saturation point, and it is 0.70 — not "keeps climbing"

`RESULTS_0818_ZSWEEP.md` §1 read `style` as climbing monotonically with `z` and taking coherence
with it, from a three-point grid (0.5 / 0.75 / 1.0 at L20 and L4). With the gaps filled, paired
item-by-item:

| contrast | `false_friend` | `style` |
|---|---|---|
| z0.60 − z0.50 | +1.6 ± 2.0 | +4.3 ± 1.8 (2.3 SE) |
| z0.65 − z0.50 | +1.3 ± 1.9 | +7.1 ± 1.5 (4.8 SE) |
| z0.70 − z0.50 | −1.5 ± 2.4 | **+12.7 ± 1.9 (6.7 SE)** |
| z0.75 − z0.70 | +0.5 ± 2.2 | **−0.1 ± 1.6 (0.0 SE)** |

**`style` rises smoothly to 0.70 and then stops.** The last step of the 0818 grid was flat and its
three points could not show it. **`false_friend` is flat across the entire range** — it was already
saturated at z=0.5, exactly as 0818 §1 said.

Both families match plain DPOP from z=0.70 up, paired: `style` +0.4 ± 2.5, `false_friend`
−0.7 ± 4.0. That reproduces the 0818 headline at a dose 0.05 lower, on a second judge pass.

## 2. THE COST IS ON `false_friend`, WHICH GETS NOTHING FOR IT

Coherence, paired against base and against z=0.50 (the dose that already maxes `false_friend`):

| arm | ff coh vs base | ff coh vs z0.50 | style coh vs base |
|---|---|---|---|
| z0.50 | −3.0 ± 1.8 | — | +1.6 ± 1.4 |
| z0.60 | −3.1 ± 1.9 | −0.1 ± 1.9 | +1.7 ± 1.8 |
| z0.65 | −5.1 ± 1.9 (2.6 SE) | −2.1 ± 1.8 | −1.0 ± 1.8 |
| **z0.70** | **−10.4 ± 2.1 (4.9 SE)** | **−7.4 ± 2.1 (3.5 SE)** | −1.4 ± 1.7 |
| z0.75 | −10.4 ± 2.4 (4.3 SE) | −7.4 ± 2.4 (3.1 SE) | −3.3 ± 2.1 |
| `P1_r1` | −3.5 ± 2.6 | — | +0.3 ± 3.0 |

**`false_friend` text falls off a cliff between z=0.65 and z=0.70** — flat to 0.60, −2.1 at 0.65,
then −7.4 against z0.50 — while its install does not move at all. **The dose `style` needs to reach
DPOP costs the lexical family seven coherence points for zero gain.** Plain DPOP pays −3.5 ± 2.6
(noise) for the same install on both families.

This is the 0818 family dissociation (§3 there: "the families are reached by different mechanisms")
showing up as a **dose conflict** on a single shared vector. `false_friend` wants z≈0.5–0.6; `style`
wants 0.70. One scalar cannot serve both, and `A_20` has only the one.

## 3. Against the pre-registered bar

The bar carried over from `RESULTS_0818_MULTI_ADDON.md` §4.2 was **`style` ≥ 66 at coherence ≥ 77**.
`L20_z0.70` gives `style` **68.6** at `style` coherence **76.1** — install met, coherence **missed
by 0.9 points**. Recorded as a miss, as 0818 recorded its 5-point one.

But the honest reading is that **the bar named the wrong column**. On `style` coherence z0.70 is
−1.4 ± 1.7 against base, i.e. indistinguishable from the unmodified model; the 77 threshold is
essentially "base level" and it is met within noise. What actually fails is `false_friend`
coherence, at −10.4 ± 2.1 (4.9 SE) — a column the bar never mentioned, on the family the bar was
not about. **The add-on does not dominate DPOP; it matches DPOP's install on both families while
damaging one family's text in a way DPOP does not.**

The compromise cell is **z0.65**: `style` 63.0 (−5.2 ± 2.5 against DPOP, 2.1 SE) with `false_friend`
coherence −2.1 ± 1.8 against z0.50 (noise). It gives up 5 `style` points to keep the text.

## 4. The judge reproduced on 384 identical items

Four arms (base, `P1_r1`, z0.50, z0.75) were re-judged here on byte-identical generations, in a
fresh invocation with a fresh shuffle and different agents. This is the judge replication
`RESULTS_0817_JUDGE_VALIDATION.md` asked for, and it is a replication of the *instrument*, not of
the training:

| arm | ff 0818 → 0819 | style 0818 → 0819 | ff coh | style coh |
|---|---|---|---|---|
| base | 39.1 → 40.6 | 27.9 → 29.5 | 77.4 → 79.6 | 78.0 → 77.5 |
| z0.50 | 68.4 → 68.6 | 51.0 → **55.9** | 77.0 → 76.6 | 79.4 → 79.0 |
| z0.75 | 68.4 → 67.6 | 69.3 → 68.5 | 69.2 → 69.2 | 72.2 → 74.2 |
| `P1_r1` | 69.7 → 67.8 | 66.0 → 68.2 | 76.6 → 76.1 | 78.2 → 77.7 |

**Install scores reproduce within 2 points except z0.50 `style` (+4.9)**; coherence reproduces
within 2.2 everywhere, which is the point of holding the rubric wording fixed. Arm ORDERING is
identical under both passes. Note the one drifting cell is the one the §1 conclusion leans on least.

## 5. Limits

1. **Single seed, one layer, one trained `A_L`.** The whole ladder is one vector; nothing here says
   a differently-seeded `A_20` has its knee in the same place.
2. **`style` is the least reliable judged axis** (25.8 MAE against hand labels,
   `RESULTS_0817_JUDGE_VALIDATION.md` §1). Ordering carries; magnitudes do not.
3. **Claude judging Claude-written references.** `results/probefix_cjudge/blind_for_human.json` is
   still unlabelled — that remains the only check that breaks the circle, and §4 above does NOT
   break it (it is the same instrument twice).
4. **Coherence is a rubric line, not a task-success measure.** The §2 cliff is read off a judge
   column; `results/probefix/ROLLOUTS_0819_ADDON_Z.md` shows every item so the failure mode can be
   inspected directly rather than inferred from the number.
5. `pf_leakage.py` is **0.000 for every cell in the grid** — the damage carries no chat-template
   artefacts, exactly as in 0818. Generation length is flat across `z` (275 / 420 characters), so
   it is not the `PREF=nll` length collapse either.

## 6. What follows

- **The dose conflict is the result worth keeping**, and it is testable: fit or train a SECOND
  vector on `false_friend` alone and steer the two families with separate `z`. If per-family doses
  clear both bars, "one direction, one scalar" was the binding constraint, not activation space.
- **A seed is now load-bearing.** §1's saturation point and §2's cliff are both single-seed
  statements about one trained vector, and they are what the write-up would rest on. Three more
  `A_20` vectors at 300 steps is ~15 minutes of GPU each and needs no new judging protocol.
- The bar should be **restated per family** before any further z work: an install target and a
  coherence floor for `false_friend` and for `style` separately. The single-column bar produced a
  0.9-point "miss" that hides a 10-point failure.
