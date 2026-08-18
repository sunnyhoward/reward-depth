# The `z` knee: a 2560-parameter add-on matches plain DPOP on BOTH families

*2026-08-18, last experiment of the session. `A_L` is trained once; `z` scales it at generation with
no retraining, so the whole trade-off curve is one generation pass per point. `z=1.0` in
`RESULTS_0818_MULTI_ADDON.md` was arbitrary and past the knee. 768 items, 16 blind agents —
**720/768 merged**: batch 13's agent died twice on the same API error mid-response, so ~90 items per
arm instead of 96. Items are shuffled across arms before chunking, so the loss is not arm-correlated.
Single seed.*

| arm | ff | ff coh | style | style coh |
|---|---|---|---|---|
| base | 39.1 ± 4 | 77.4 | 27.9 ± 2 | 78.0 |
| `L4_z0.25` | 54.8 ± 5 | 77.1 | 37.4 ± 3 | 77.6 |
| `L4_z0.5` | 63.6 ± 4 | 75.9 | 58.5 ± 3 | 78.4 |
| `L4_z0.75` | 62.9 ± 4 | 63.7 | 80.0 ± 1 | 70.2 |
| `L4_z1.0` | 58.2 ± 3 | **55.1** | **85.2 ± 1** | **52.6** |
| `L20_z0.5` | 68.4 ± 4 | **77.0** | 51.0 ± 3 | **79.4** |
| **`L20_z0.75`** | **68.4 ± 3** | 69.2 | **69.3 ± 2** | 72.2 |
| `P1_r1` (plain DPOP) | 69.7 ± 4 | 76.6 | 66.0 ± 3 | 78.2 |

**`addon_L20` at z=0.75 matches plain DPOP on both families at once**, paired on the same prompts:

| contrast | `false_friend` | `style` |
|---|---|---|
| `L20_z0.75` − `P1_r1` | **−0.4 ± 4.3** | **+3.2 ± 2.4** |
| `L20_z0.5` − `P1_r1` | **+0.0 ± 4.5** | −14.9 ± 2.8 |
| `L20_z0.75` − base | +29.2 ± 5.4 | +40.9 ± 2.3 |

That is **2560 trained parameters on a frozen model** reaching the judged install of a full LoRA
DPOP run (~21M parameters, 300 steps) on both the lexical and the register family simultaneously.

---

## 1. The knee is real, and the two families have different ones

- **`false_friend` saturates at z=0.5** and does not improve after: `L20` reads 68.4 at both z=0.5
  and z=0.75, i.e. DPOP-level, and at z=0.5 it costs **nothing** — coherence 77.0/79.4 against
  base's 77.4/78.0.
- **`style` keeps climbing with z** and takes coherence with it: `L20` 51.0 → 69.3 as coherence
  falls 79.4 → 72.2; `L4` 58.5 → 80.0 → 85.2 as coherence falls 78.4 → 70.2 → **52.6**.

So the honest form of `RESULTS_0818_MULTI_ADDON.md` §2's headline is: the `style` ceiling is broken,
but the part above DPOP's 66.0 is bought with coherence. **`z=1.0` (style 85.2, coherence 52.6) is
Goodhart; `z=0.75` at L20 (style 69.3, coherence 72.2) is not.**

## 2. Against the pre-registered bar

The bar set before the run was **style ≥ 66 at coherence ≥ 77**. `L20_z0.75` gives style **69.3** at
coherence **72.2** — the install bar is met, the coherence bar is missed by ~5 points. Reported as a
miss, not rounded into a pass. `L20_z0.5` is the cell that meets the coherence bar exactly
(77.0/79.4) and it reaches DPOP on `false_friend` only.

**A finer sweep between z=0.5 and z=0.75 has not been run** and is the obvious next point: one
generation pass per z, no retraining.

## 3. What this changes about the project's central negative

Every activation-space result in `probefix/` concluded that activation methods lose to output DPOP.
That conclusion was drawn from **training** arms whose objectives optimise a difference between
chosen and rejected reads. A single vector, trained on generation likelihood and scaled at inference,
is level with DPOP on both families. The negative was about **the objectives**, not about
activation space.

Note also what did NOT work, from the same session: the fitted mean-difference direction tops out at
`style` 32.1 (`RESULTS_0818_STEER.md` §5), simultaneous multi-layer steering adds nothing
(`RESULTS_0818_MULTI_ADDON.md` §1), and `PREF=nll` collapses generation length. The result depends on
**learning the vector against a length-cancelling margin**, not on activation editing per se.

## 4. Limits

1. **720/768 verdicts** (see header). ~90 items per arm.
2. **Single seed**, one `z` grid, two layers.
3. **`style` is the least reliable judged axis**: 25.8 mean absolute error against hand labels,
   band agreement 0.667 (`RESULTS_0817_JUDGE_VALIDATION.md` §1). Ordering carries; magnitudes do not.
4. One judge (batch 00) reported applying its own uniform coherence deduction for truncation,
   against the instruction. Shuffling makes that noise rather than arm-bias, but it is a real
   deviation and it is on the record.
5. Claude judging Claude-written references. `blind_for_human.json` remains unlabelled.
