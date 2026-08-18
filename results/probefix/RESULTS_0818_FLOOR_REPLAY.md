# The floor's damage was drift, not magnitude: a KL anchor takes leakage to zero

*2026-08-18. `NEXT_0817.md` §6.0b and §6.1, run together as that doc asked
(`probefix/run_pf4b_mdfloor_kl.sh`). Three arms, `MODE=meandiff` at L20, LoRA 0–31, `LAMBDA=1.0`,
300 steps, seed 0, Qwen3.5-4B, `CKPT_EVERY=100`. Comparators are banked from 0817:
`MD_L20_floor` (unbounded floor, no replay, leakage .625) and `MD_L20_s300` (no floor, no replay,
.146). Meters here are `pf_leakage.py`, `pf_mdsides.py` and the training-time ranking buckets —
**no judged install**; that needs a `pf_cjudge.py` pass and was not run. Single seed.*

| arm | floor | replay | **leak s100** | **leak s300** |
|---|---|---|---|---|
| `MD_L20_s300` (banked) | none | none | — | .146 |
| `MD_L20_floor` (banked) | unbounded | none | .000 | **.625** |
| **`MDF_kl`** | unbounded | **`kl`** | .094 | **.000** |
| `MDF_sat` | capped | none | .333 | **.583** |
| `MDF_sat_kl` | capped | `kl` | .188 | .125 |

**§6.0b is answered YES, and emphatically.** The same unbounded floor that leaked **.625** leaks
**.000** — 0/48 on both families — once the replay term is an actual anchor. This is the first
zero-leakage floor cell at 300 steps in the project; the only other clean floor cell was
`MD_L20_floor_s100`, i.e. one that stopped early.

**And §6.1's prediction fails, along with the diagnosis behind it.** `RESULTS_0817_MDFLOOR.md` §4
put on record that a **saturating** floor would "recover the s100 numbers at s300", because the
wreckage was attributed to `relu(ref_chosen·u − chosen·u)` being unbounded where `relu(M0 − proj)`
saturates. Capping it changes **nothing**: .583 against the uncapped .625. The wreckage was never
about that channel.

---

## 1. Leakage tracks how far the representation moved, and the anchor is what restrains it

`pf_mdsides.py`, base frame, 128 pairs, `READ=mean` (`results/probefix/mdsides_floorkl.json`).
Δchosen at s300 against the leakage in the same row:

| arm | Δchosen L20 | Δchosen L30 | leak s300 |
|---|---|---|---|
| `MD_L20_floor` (banked) | — | **+54.52** | .625 |
| `MDF_sat` | **+98.55** | +41.89 | .583 |
| `MDF_sat_kl` | +71.65 | +35.36 | .125 |
| **`MDF_kl`** | +47.47 | **+22.63** | **.000** |

Monotone across four cells at L30: the further the chosen side's absolute projection travels, the
more the text leaks. **The KL anchor is the only thing in the grid that reduces the travel** — and
it does so while leaving the floor itself unbounded. Note also that `MDF_kl` moves *less at 300
steps than at 100* (L20 +81.50 → +47.47): the anchor pulls back over training rather than merely
slowing the departure.

**The cap does not bound the overshoot at all.** `MDF_sat` has the **largest** Δchosen in the grid
(+98.55 at L20) despite being the arm whose floor was supposed to be bounded.

## 2. Why the cap was the wrong instrument, and it is a general point

`MD_CAP_MULT=1.0` resolved to a cap of **−2.17**, because the base model's chosen-side projection at
L20 is −2.17. `u` is a **difference** direction, so an absolute projection onto it carries an
arbitrary offset and its sign is not meaningful. Consequences:

- `relu(min(ref_chosen, cap) − chosen)` with the cap at the base level is not a *saturating lift*.
  It is a **don't-fall-below-base floor that switches off** the moment the chosen side rises above
  where it started — which the objective makes it do immediately.
- It is nevertheless **not inert**: `MDF_sat` leaks .583 where the floor-free `MD_L20_s300` leaks
  .146, so the capped term still binds early and still does damage. It just cannot deliver the
  bounded lift §6.1 wanted.
- **`MD_CAP_MULT` is only interpretable at 1.0** when the base projection is negative; any other
  multiplier moves the cap the wrong way. This is unlike `M0_MULT`, whose quantity is a difference
  and positive by construction. The warning `pf_train.py` prints now says this.

The general point for the write-up: **`M0_MULT`-style ratio calibration is valid for the difference
and invalid for either side's absolute projection.** Any future term on an absolute position needs
a reference expressed as an offset, not a multiple.

## 3. What the anchor costs

Training-time ranking buckets — **teacher-forced, not generation**:

| arm | ff raw s300 | style raw s300 | ff implicit | replay KL s300 |
|---|---|---|---|---|
| `MD_L20_floor` (banked) | **.60** | .22 | .84 | .114 |
| `MDF_sat` | .58 | **.26** | .82 | .076 |
| `MDF_sat_kl` | .53 | .19 | .86 | .011 |
| `MDF_kl` | **.49** | .21 | .77 | **.010** |

- **The anchor holds**: replay KL flat at .010–.015 across all 300 steps where the unbounded floor
  arm drifts .046 → .114. That is the difference between an anchor and a competing objective, in the
  drift meter, and it is the first time `REPLAY_LOSS=kl` has been run at all.
- **It costs ranking install**: `false_friend` .60 → .49, implicit .84 → .77. Same direction as the
  banked `MD_r1` vs `r0` replay contrast (58.0 vs 62.1 judged). `style` is a wash.
- **Cap plus anchor is worse than anchor alone** on the meter that matters here (.125 vs .000
  leakage) and better on install (.53 vs .49). One seed; do not build on the ordering.

## 4. What this settles

1. **The rescue exists and it is `REPLAY_LOSS=kl`.** Every mean-diff and floor arm in this project
   ran without it. The pathology that made the floor unusable is gone — not reduced, zero.
2. **`RESULTS_0817_MDFLOOR.md` §2's account is retracted.** The floor did not wreck the text by
   pushing the chosen side up without bound; bounding that channel leaves the leakage where it was.
   The damage is **drift from the base distribution**, and an anchor on the output distribution is
   what stops it. §2 of that doc should be read with this one.
3. **The negative is unchanged and now has fewer excuses.** Both remaining objections to the 0817
   result — "the floor was never given an anchor", "the floor was never bounded" — are now closed,
   and ranking install did not improve in any arm. Judged install is not measured here.
4. **The one cell worth judging** is `MDF_kl_s300`: zero leakage at full training length, the
   smallest representational drift in the family, and an install meter 11 points below the banked
   floor arm's. Whether that trade is real needs `pf_cjudge.py` against `MD_L20_s300`, `P1_r1` and
   `base` — one pass, and the only way to know whether "clean but ranks worse" is also "clean but
   generates worse".

**Limits.** Single seed on every cell. Leakage is a regex over 96 generations per cell, exact for
what it catches and blind to everything else — it is not a coherence meter. The ranking buckets are
teacher-forced, the failure mode this project has been burned by twice. `GRAD_CKPT=1` was on for all
three arms (activation recomputation, semantically identical) because the `kl` arm OOMs without it.
