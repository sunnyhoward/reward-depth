# Multi-depth severed supervision: the best activation arm yet, still beaten — and the margin is won the DPO way

*2026-08-17. **`MODE=mdstack`** — the mean-diff objective at five read points (6/12/18/24/30) with the
stack **severed between segments**, so the gradient from read L reaches only that segment's blocks.
Deep supervision with gradient truncation (cf. Lee et al. 2015; Belilovsky et al. 2019).
`Qwen3.5-4B`, `brit_dose20.jsonl`, replay off, LoRA 0–31, λ=1.0 output floor, seed 0, 300 steps.
480 items (5 arms × 96) judged blind by 10 Claude subagents, shuffled across arms and families, with
`MD_L20_s300`, `P1_r1` and `base` judged in-pass. Plus a new diagnostic, `probefix/pf_mdsides.py`.*

**Two results.** (1) Multi-depth severed supervision is the **best activation-space design tested in
this project** — and still loses to plain DPOP at the output. (2) The diagnostic shows *why* the
whole family underperforms: the objective is won the way plain DPO wins its margin, by **dropping the
rejected side while the chosen side falls too**. That is a fixable defect, and `MD_FLOOR` (below) is
the fix now under test.

---

## 1. Judged result

Unconditional (non-engagement scored 50), paired per prompt over the same 48 held-out prompts:

| arm | `false_friend` | `style` | coherence ff / style | leakage |
|---|---|---|---|---|
| base | 39.1 | 26.2 | 74 / 70 | 0.000 |
| MDSTACK s100 | 47.2 | 46.7 | **55 / 48** | **0.448** |
| **MDSTACK s300** | **59.5** | **56.6** | 73 / 77 | 0.094 |
| MD_L20_s300 (best single-attach) | 57.7 | 42.9 | 76 / 66 | 0.146 |
| **P1** (plain DPOP at output) | **71.6** | **71.8** | 77 / 78 | 0.000 |

| contrast | `false_friend` | `style` |
|---|---|---|
| MDSTACK s300 − base | **+20.4 ± 5.3 (+3.8 SE)** | **+30.4 ± 3.0 (+10.2 SE)** |
| MDSTACK s300 − MD_L20_s300 | +1.6 ± 4.8 | **+3.8 ± 3.7** |
| MDSTACK s300 − P1 | **−12.1 ± 5.6 (−2.1 SE)** | **−15.2 ± 3.2 (−4.8 SE)** |
| MDSTACK s100 − P1 | −24.3 ± 5.8 | −25.2 ± 3.4 |

**The step axis inverts.** Every single-attach mean-diff arm peaks at s100 and decays
(`RESULTS_0817_MEANDIFF_REJUDGE.md` §3d). mdstack does the opposite: s100 is its *worst* cell, with
leakage 0.448 and coherence 48–55, and s300 is clean (0.094) and its best. Consistent with segments
needing time to reach agreement with each other when none can see the output.

**It is the best activation arm, by a little.** +3.8 on `style` over MD_L20 (1.0 SE) and level on
`false_friend`. Its `style` install is the largest any activation objective has produced here.

**It still loses to output DPO** by 12–15 points, 2.1 and 4.8 SE. The deficit is the smallest of any
activation arm, which is the honest way to state it.

## 2. Why — the chosen side falls (`probefix/pf_mdsides.py`)

The activation analogue of the `d_chosen`/`d_rejected` split that `RESULTS_DPOP_4B.md` introduced for
DPO. Projections in a **fixed base frame** (`u` and `SD` fitted on the base policy), 128 pairs:

| arm | read | proj (base) | Δchosen | Δrejected | ‖read‖ (base) |
|---|---|---|---|---|---|
| **MD_L30_s300** | L30 | +62.0 (+25.0) | **−11.7** | **−48.7** | 30.5 (30.2) |
| **MDSTACK_s300** | L6 | +60.2 (+21.2) | +55.5 | +16.4 | 33.0 (33.6) |
| MDSTACK_s300 | L18 | +23.5 (+18.2) | +67.4 | +62.1 | 32.7 (30.6) |
| **MDSTACK_s300** | L30 | +45.0 (+25.0) | **−51.7** | **−71.7** | 31.3 (30.2) |
| **P3 (DPOP, output)** | L30 | +8.9 (+25.0) | **+138.4** | +154.5 | 32.0 (30.2) |

- **`MD_L30` more than doubles its projection with the chosen side going DOWN.** Read norms are
  unchanged, so this is not the frame rotation that `probefix/RESULTS.md` warns about — it is the
  scale-free-margin pathology, in activation space. `relu(M0 − proj)` constrains only the
  *difference*; nothing pins either side.
- **mdstack raises both sides in its lower segments** (Δchosen +55 to +67 at L6–L24) — it moves the
  representation wholesale rather than winning a difference. **But its top segment reverts to the
  failure** (Δchosen −51.7). The segment nearest the output is where the objective is gamed, which is
  also where `RESULTS_0817_MDLATE.md` found single-attach arms inert.
- **The output-attached control does the opposite**: P3's DPOP raises the chosen projection **+138**
  at L30. The λ floor in these runs acts on output logps, so the activation term has had **no
  absolute anchor at all**.

## 3. The fix now under test: `MD_FLOOR`

`pf_train.py MD_FLOOR=w` adds `w · relu(ref_chosen·u − chosen·u)` — DPOP's floor, on the chosen
side's own projection, in the same SD-normalised units, with the reference taken adapter-disabled.
For `mdstack` it is computed **inside the severed forward**, so each segment's anchor is local too.
`MD_FLOOR=0` is the default and reproduces every previously run arm exactly.

Running at the time of writing: `MDSTACK_floor` (multi-depth + per-segment floor) and `MD_L20_floor`
(single-attach + the same floor, isolating the floor from the multi-depth design).

**Pre-registered reading.** The floor should first show up in `pf_mdsides.py` as **Δchosen turning
positive at L30**, where it is −51.7 without it. If it does and the judged install does not move,
that is the phase-1 null-space result again (`cos(μ, W_A−W_B) = −0.003` at the final layer) and the
line is finished. If the install does move, the read-depth negative gets reframed: the top of the
stack looked inert partly because that is where the unanchored objective was most gamed.

## 4. Scope limits

1. **Single seed.** Run-to-run spread on `style` is ~11 points
   (`RESULTS_0817_MEANDIFF_REJUDGE.md` §4) — larger than the +3.8 mdstack-over-L20 gap, which should
   therefore be read as "no worse", not "better".
2. `pf_mdsides.py` uses a **base-fitted frame**; the norm column is what distinguishes a genuine fall
   from a rotation, and it is flat here. n = 128 pairs, no error bars.
3. **Claude judge only**, in-pass anchors, unconditional scoring primary. Coherence comparable within
   this pass and to the 0814-worded passes.
4. `GRAD_CKPT` is unavailable for `mdstack` (the severance pre-hook breaks checkpoint recompute), so
   the arm needs ~81 GiB.
