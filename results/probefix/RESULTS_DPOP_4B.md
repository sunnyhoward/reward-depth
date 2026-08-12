# DPO-Positive vs plain DPO at 4B, with the step axis and the stage-1 cross

*2026-08-12. All cells `Qwen3.5-4B`, attach block 20, `brit_dose20.jsonl` (guard = 20% of the
training diet), seed 0, 600 steps, `CKPT_EVERY=100`. Adapters:
`huggingface.co/sunnyhoward/reward-depth-probefix4b` (private).*

Written against `ROLLOUT_ANALYSIS.md`, which established that the 0811 headline compared C1@300
against A@600 — a healthy model against a collapsed one — and could not separate the objective
from the stopping point. This runs both, crossed.

## 1. The mechanism, measured rather than inferred

`pf_train.py` now logs `d_chosen` and `d_rejected`, not just their difference. That is the whole
argument:

| arm | margin | **d_chosen** | d_rejected | replay KL |
|---|---|---|---|---|
| A plain, all layers | +244 | **−312** | −556 | 0.269 |
| DPOP λ=50, all layers | +26 | **+22** | −4 | 0.238 |
| two-stage, plain | +85 | −56 | −141 | **0.031** |
| two-stage, DPOP | +32 | **+16** | −17 | **0.033** |

Plain DPO wins its +244 margin by pushing the chosen continuation 312 nats **below** reference.
Nothing in vanilla DPO pins the chosen side's absolute likelihood, and on minimal pairs the
cheapest way to make the rejected continuation unlikely is to stop writing English. DPOP's
`−λ·relu(ref_chosen − chosen)` holds it positive for all 600 steps.

## 2. What the arms actually produce

n=128 free samples. `density` = (am+br) hits per sample — `brit_rate`'s denominator is the marker
count itself and so cannot see a model that stops producing markers; density's cannot shrink.

| cell | rank | brit_rate | density | br | am | div | rep | nonascii | len |
|---|---|---|---|---|---|---|---|---|---|
| base | .172 | .218 | 0.68 | 19 | 68 | .87 | .03 | .000 | 64 |
| A plain @300 | .967 | .766 | 0.37 | 36 | 11 | .41 | .35 | .475 | 78 |
| **A plain @600** | .969 | **1.000** | **0.08** | 10 | 0 | .11 | .47 | .895 | 89 |
| DPOP @300 | .875 | .948 | 0.60 | 73 | 4 | .93 | .04 | .015 | 14 |
| DPOP @600 | .865 | .992 | 0.98 | 124 | 1 | .93 | .02 | .000 | 35 |
| two-stage plain @300 | .970 | .711 | 1.16 | 106 | 43 | .94 | .04 | .001 | 72 |
| **two-stage plain @600** | **.979** | .651 | **1.32** | 110 | 59 | .91 | .04 | .001 | 73 |
| two-stage DPOP @300 | .867 | .802 | 0.79 | 81 | 20 | .94 | .03 | .001 | 63 |
| **two-stage DPOP @600** | .866 | .805 | 1.00 | 103 | 25 | .92 | .03 | .007 | 63 |

**A@600 scores a perfect `brit_rate` of 1.000 on ten marker hits across 128 samples.** It is
em-dash spam. The old two-meter panel would have crowned it; `density` 0.08 and `nonascii` .895
identify it in one line.

## 3. "Every arm at 600 steps degenerates" is false

That was `ROLLOUT_ANALYSIS.md` §3's central observation and it does not survive the stage-1 cross.
**Two-stage plain @600 is the best cell in the study** on ranking (.979) and density (1.32, nearly
double base), at near-base length, fully coherent — same objective, same data, same 600 steps as
the arm that collapsed. The only difference is stage 1, and the mechanism is visible: stage 1 does
not stop the chosen side falling (it falls *faster* early, −35 vs −19 at step 100) but it bounds
it, plateauing near −56 while plain slides to −312.

So there are two independent routes out of the trap, and they work differently: DPOP pins
`d_chosen` positive by construction; stage 1 lets it fall to a floor and holds it there.

## 4. The two working arms are not ranked

- **DPOP all-layers** is nearly *pure* British — of 125 markers, ~124 are British — but terse at
  35 words against base's 64, and 14 words at @300.
- **Two-stage plain** writes the most marked language of any arm at natural length, but stays
  mixed-dialect (.651): it raised British usage without suppressing American forms.
- **Two-stage DPOP** is the balanced cell: brit_rate .805, density 1.00, length 63, both
  mechanisms composing as expected (positive `d_chosen` *and* 8× lower drift).

Purity or density-at-natural-length is a choice about what the install is for. `brit_rate` alone
would have ranked the dead arm above all three.

## 5. Guard, generation-side, for the first time

`pf_rollouts.py` now forces the sentence under test. Each guard row is a minimal pair differing in
**two** places — the dialect marker and the fact — so marker positions are excluded via
`meta.marker` and the assistant turn is pre-filled up to the *fact* divergence, once with the
British-marked prefix and once with the American one.

**`false` is 0.000 for every arm under both prefixes**, base included; truth rates .17–.57 with
the remainder writing a different-but-not-false continuation. No arm completes a sentence with a
falsehood in order to sound British.

Two limits: it is prefix-forced, so it cannot show whether a model would *volunteer* the lie; and
the large "other" bucket makes it a conservative test.

## 6. Where the objectives write

Relative ‖ΔW‖_F per block after subtracting the step-matched replay-only null (`R_4b_replay_only`,
`W_PREF=0`), ×1000. The null is flat (.012–.018 @300, .018–.027 @600) — the anchor writes
everywhere, which is why raw profiles show no structure.

| cell | b0 | b8 | b16 | b20 (L\*) | b24 | b29 | peak |
|---|---|---|---|---|---|---|---|
| plain @600 | −2.1 | 0.2 | **4.9** | 1.7 | 0.9 | 0.9 | block 16 |
| DPOP @600 | −4.0 | 0.4 | 3.1 | 1.6 | 1.2 | **5.7** | block 29 |

The two objectives write in **different places**: plain DPO mid-stack, DPOP near the top. Neither
peaks at L\*=20, so HANDOVER's 2B claim that every arm's largest change lands at L\* does not
carry to 4B. Both write *less* than replay-only at block 0.

Caveat: the subtracted signal (.004–.006) is smaller than the null floor it is removed from
(.012–.027), single seed. Peak locations are consistent across 300 and 600 within each arm.

## 7. Open

1. **Single seed, everywhere.** Nothing here is replicated.
2. **DPOP is weak on `install_truth_dialect`** — .140 raw against plain's .950 (ref .650). Stage 1
   recovers much of it (.190 / .840). Unexplained; may be underfitting at λ=50 on the hardest
   family rather than a property of the anchor.
3. **No length meter.** DPOP@300's 14-word answers are a failure mode none of the four new meters
   detects; terseness has burned this project before.
4. **Today's plain arm degrades faster than 0811's** (collapsed by step 300, where 0811's was
   still coherent), on the same seed and recipe. Most likely `transformers` 5.15 / `peft` 0.20
   versus the wiped venv. Compare today's numbers against today's null, not the 0811 tables.
5. **Naming collision.** Today's `C0`/`C1` use the digit for the *objective*; 0811's `C1`/`C2`
   used it for the *write range*. Both live in `results/probefix4b/runs/`. Today's pair are both
   the old `C1` write range (blocks 21–31).
