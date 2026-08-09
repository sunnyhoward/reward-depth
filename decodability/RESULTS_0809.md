# hops, re-examined — the construction fix, and a lookup-free control (2026-08-09)

Scoped to the depth dial. `decodability/NEXT.md` (08-07) carried two required fixes to `hops` and
called the `L*(k)` curve the live thread. This file is what happened when they were applied, plus
a control built to answer a specific objection: **that `hops` may be solvable by a chain of
retrieval heads, in which case "L\* rises with k" restates a fact about attention rather than
locating a preference.**

## 1. The banked ladder reproduces exactly, so the environment is not a variable

`HOPS_LEGACY=1 HOPS_CHAIN=6` restores the exact pre-fix construction. On this box, under
transformers 5.14.1 and a fresh cache:

| k | banked (08-07) | legacy repro (08-09) |
|---|---|---|
| 1 | 2 | **2** |
| 2 | 8 | **8** |
| 3 | 12 | **12** |
| 4 | 14 | **14** |
| 5 | *7* | ***7*** |

Bit-for-bit, including the k=5 break. **Any change below is caused by the construction, not by
the environment.** This control is the reason the rest of the file can say anything.

## 2. The k=5 break was a construction flaw — confirmed and removed

`NEXT.md` diagnosed k=5 as a flaw rather than a fact: at `chain=6` its distractor pool is {4, 6},
and index 6 is the last name in the premise, so "second-to-last vs last" is solvable by recency
without composing 5 hops. Under the corrected construction k=5 lands at **14 at both scales**
instead of collapsing to the k=2 level. The diagnosis was right.

## 3. But the corrected ladder is scale-dependent

`L*` (mean read, linear rung), corrected construction — chain 10, span 2, direction-balanced:

| k | 1.7B old | **1.7B fixed** | 4B old | **4B fixed** |
|---|---|---|---|---|
| 1 | 2 | 3 | 3 | 3 |
| 2 | 8 | **9** | 10 | **10** |
| 3 | 12 | **9** | 14 | **12** |
| 4 | 14 | **9** | 16 | **14** |
| 5 | *7* | **14** | *10* | **14** |

- **At 4B the dial turns and the fix improves it**: 3 → 10 → 12 → 14 → 14, monotone through k=4
  with the k=5 break gone.
- **At 1.7B it flattens**: k=2,3,4 all collapse to L\* = 9.

So the monotone 1.7B ladder in the banked results was substantially an artifact of the flawed
construction. The dial is real at 4B and not at 1.7B, which is a claim about scale that the
original single-model result could not have made.

### 3a. Attributed: it is the chain length, not the shortcut fixes

The fix changed three things at once — chain 6→10, span 1→2, direction balancing — so a
`chain=6, span=2, balanced` control holds chain at the old value and moves only span+balance.
(`chain=10, span=1` is impossible: k=1 would have a single distractor, which the degeneracy
assert rejects — correctly.)

| config (1.7B) | k2 | k3 | k4 |
|---|---|---|---|
| legacy — chain 6, span 1, unbalanced | 8 | 12 | 14 |
| chain 6, span 2, **balanced** | 5 | 12 | 14 |
| chain 10, span 2, balanced (the fix) | **9** | **9** | **9** |

With chain held at 6 the ladder still rises through k=2–4 under the wider, balanced construction.
It flattens only when the chain goes to 10. **The two changes made for shortcut reasons are not
what cost the ladder; premise length is.** Why a longer premise compresses L\* at 1.7B while 4B
keeps its ladder is not something these measurements can say, and this file does not guess.

**This control is itself flawed at k=5 and its k=5 row is not usable.** At `chain=6, span=2`,
k=5's pool is {3, 4} — two distractors, so the size assert passed, but both *backward*, so
"prefer the later-mentioned candidate" solves it. That is §4's positional shortcut mirrored. The
assert now checks pool *direction* as well as size for every k > 1, and this config is rejected
by it.

## 4. A shortcut the existing floor battery could not see

Found by auditing the *fixed* set before spending GPU on it. Hop 0 is excluded from the
distractor pool (the question quotes the starting name verbatim), so at low k every distractor is
*further along the chain* — and **"prefer the earlier-mentioned candidate" solved k=1 outright
(P = 1.000) and reached 0.593 at k=2.**

This is a **positional** cue. The lexical floor is a bag-of-token-ids probe and the length floor
counts tokens; neither can represent premise order, so both would have passed this set. §1c's
standing conclusion — the target is |surface signal| ≈ 0, not signal reversed — extends here: the
surface channels that need auditing are not only vocabulary and length.

Distractor direction is now drawn before magnitude, giving P(distractor later) = 0.50–0.55 for
k ≥ 2. **k = 1 cannot be fixed** — its only backward neighbour is hop 0 — and is documented as an
unreliable rung. It is also the ladder's shallow anchor, so the bottom of every `hops` curve
carries this caveat.

## 5. The lookup-free control: `arith_hops`

Same chain skeleton, but only the *start* value is stated and every later value must be computed,
so no hop is answerable by copying a premise token:

```
Bruno has 52. Zach has 2 more than Bruno. Uma has 1 more than Zach. ...
Starting at Bruno and following 3 steps, what number do you reach?
chosen " 55."   rejected " 58."
```

Surface floors are 0.5 by construction and measured so before any GPU: every value is held to two
digits, so the **length floor is exactly 0.500 with no variance to fit** (against `hops`'
0.482–0.566, since names differ in token length), and the larger side alternates by item index.

**At its default difficulty it is unusable.** Mixed +/- with deltas 1–9 puts both 1.7B and 4B at
the chance floor for k ≥ 2 — peaks 0.53–0.60 against a shuffled null of 0.45–0.55. By
`dec_plots.py:466`'s own rule that makes L\* a position in noise.

**Add-only with deltas 1–2 produces signal**, and the result is a contrast:

| k | `hops` L\* (4B) | `arith_hops` L\* (4B) | arith peak |
|---|---|---|---|
| 2 | 10 | 12 | 0.890 |
| 3 | 12 | 12 | 0.913 |
| 4 | 14 | 13 | 0.861 |
| 5 | 14 | 13 | 0.801 |

**`hops` rises 10 → 12 → 14 → 14; `arith_hops` is flat at 12–13.** Read cautiously — arith peaks
decline with k (1.00 → 0.80) and L\* is the earliest read within 1 SE of *that curve's own* max,
so a lower ceiling can be reached earlier for reasons unrelated to depth. The comparison is not
clean until the arith curves sit nearer ceiling. At 1.7B the arith ladder is noisy (3, 6, 7, 13,
5) on peaks of 0.72–0.85 and should not be read at all.

If it holds up, it says the rising ladder in `hops` is specific to serial *lookup*, and that
composition-by-computation saturates at roughly a third of depth regardless of hop count — which
is the objection this control was built for, answered in the direction that limits `hops`.

**One thing not held constant**, stated because it bounds the comparison: in `hops` both
completions appear verbatim in the premise and the task is selection among present tokens; in
`arith_hops` neither does and the task is to produce a computed value. That is unavoidable — it
is the manipulation — but a difference between the ladders is lookup-vs-computation confounded
with select-vs-produce.

Add-only also makes **k=1 a dead rung**: the walk is monotone and every k=1 distractor is forward,
so P(chosen > rejected) = 0.000 and "pick the smaller number" is exact. Drop it, do not report it.

## 6. `knowcomp` — the contrast that actually has range

`hops` gives a dial with a **narrow** span. Extending it to k=8 at chain=12 (4B) shows it
saturates:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| L\*/D | 0.11 | 0.31 | 0.42 | 0.39 | 0.42 | 0.44 | 0.33 | 0.36 |
| peak | 1.000 | 1.000 | 0.996 | 0.957 | 0.869 | 0.837 | 0.871 | 0.872 |

L\* plateaus at ~0.4 from k=3 and never goes deeper, while peak accuracy falls to 0.87 so the
k ≥ 5 rungs measure a task the model cannot reliably do. **Usable range is k=2→3, i.e. 0.31→0.42.**
Adding hops does not buy depth.

The contrast with range was already in `RESULTS.md` §1a — computation-correctness is the only
family that starts at chance and it resolves ~3/4 of the way up, against retrieval at ~0.27 — but
it rested on n=14 test pairs and was a sub-split of `styc` by `meta.typ` rather than a dataset.
`load_knowcomp` makes it one: two families, **one** prompt template (`load_styc`'s exact
`Question: {q}\nAnswer:` with a terse completion), correct vs near-miss wrong answer.

| scale | retrieval L\*/D | computation L\*/D | spread |
|---|---|---|---|
| 1.7B (28L) | 0.36 | 0.86 | 0.50 |
| 4B (36L) | 0.33 | 0.75 | 0.42 |
| 8B (36L) | 0.28 | 0.75 | 0.47 |

Both families are at chance at L0 and reach ceiling at peak, at every scale. Floors after the fix
below: retrieval lexical(group) 0.395 / length 0.526; computation lexical 0.568 / length **0.500
exactly** (`mean_abs_len_diff` 0.0 — the near-miss distractor is built at the same digit width, so
there is no "which number looks more like an answer" cue at all).

**A length cue, found and fixed.** The retrieval side started with 65 items whose true answer was
shorter than the distractor against 23 longer (122 equal), so "prefer the shorter answer" scored
~0.600 and the measured length floor was 0.585. Balancing on **character** length was tried first
and is not an adequate proxy — it dropped 6 items and left the token imbalance untouched
(64/118/22, rule still 0.603), because char-equal and token-equal are different partitions.
Balancing on **token** length with a fixed reference tokenizer gives 23/122/23 and a rule at
exactly 0.500, costing 210 → 168 retrieval items.

`helpers.KNOW_BANK` is untouched: it feeds `load_styc`, so expanding it in place would have
changed the item mix behind every banked styc number. `KNOW_EXT` extends it only here.

### The catch, for anyone building an attach experiment on this

L\* above is measured **pairwise** — a probe comparing two completions of the same prompt. A
*reward* needs an **absolute** score for one completion in isolation, and that is a harder problem
that matures later. Absolute-probe held-out accuracy by layer (1.7B, `kc_rl.py`):

| layer | 2 | 6 | 10 | 14 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| retrieval | 0.474 | 0.579 | 0.776 | 0.868 | **0.895** | 0.829 | 0.763 |
| computation | — | — | 0.659 (L12) | — | 0.705 (L18) | **0.909** | — |

The two coincide for computation (~L24) and **disagree for retrieval by 4–10 layers** (pairwise
L\*=10, absolute peak L20). So "attach at L\*" is ambiguous until L\* is defined by the readout
being trained against. Defined that way: retrieval L\*≈14 (0.50D) vs computation L\*≈24 (0.86D) — a
0.36-of-depth contrast, smaller than the 0.50 the pairwise numbers advertise.

Worth noting for the de-confounding argument: the retrieval probe **peaks at L20 and declines** to
0.763, whereas EAGLE head agreement climbs monotonically (0.361/0.394/0.435/0.653/0.812 at
L5/9/13/17/21, identical 5000-step budget). A readout that falls off with depth cannot manufacture
a spurious "deeper is better"; one that climbs can.

## 7. Where this leaves the depth dial

The instrument is in better shape than the banked results and supports a narrower claim:

- Use **`knowcomp`**, not `hops`, when the experiment needs range: 0.28-0.36 vs 0.75-0.86 against
  hops' usable 0.31-0.42, and it replicates at three scales.
- If using `hops` anyway, use **4B or larger**. The dial does not turn at 1.7B under a sound
  construction.
- Drop **k=1** at all times, and under `AHOPS_OPS=add` treat it as invalid rather than weak.
- `hops` alone still cannot separate serial-lookup depth from composition depth; `arith_hops` is
  the control that can, once its curves are nearer ceiling.

Unchanged from `NEXT.md`: a `hops` result is a proof-of-concept for the (attach − L\*) design, not
a claim about how preferences are represented, and `KNOW_BANK` remains the naturalistic anchor
that would let one generalise (retrieval L\*/D 0.25–0.29 vs computation 0.75–0.86, on n=14 test
pairs — queue item 5 is to expand it).

## Files

- `dec_data.py` — `load_hops` fixed (chain 10, span 2, hop-CHAIN excluded structurally, direction
  balanced) with `HOPS_LEGACY=1` for reproduction only; `load_arith_hops` new, with
  `AHOPS_OPS` / `AHOPS_DELTA_MAX`.
- Banked: `results/decodability/scalar_qwen3-{1.7b,4b}_{hops,arith_hops}_chat.json` (corrected
  construction, default arith difficulty). **The two `hops` files OVERWRITE the 08-07 banked
  numbers** — same filenames, different construction; the originals are in git at `514b37a` /
  `a033093`, and §1's legacy control reproduces them exactly.
- `legacy_qwen3-0.6b_hops_chat.json` — renamed from `scalar_`. 0.6B was **not** re-run, so it is
  still OLD-construction and must not be put in a table beside the 1.7B/4B files. Renaming it is
  the only thing preventing a silent mixed-construction comparison, which is the failure mode
  this directory is most exposed to now that both constructions exist under one naming scheme.
- `ctrl_legacy_*`, `ctrl_chain6span2_*`, `easy_*` — the §1/§3a/§5 controls.
- Controls, each under its own `DEC_ROOT` so none can be mistaken for a banked run:
  `/workspace/dec_ctrl_legacy`, `/workspace/dec_arith_easy`, `/workspace/dec_ctrl_chain6span2`.
  **These are on container storage and die with the box** (`workspace_is_volume: false`).
