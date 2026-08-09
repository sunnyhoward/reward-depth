# Supervisor recipe, stage 1 — the EAGLE arm and its matched control (2026-08-09)

`sup_train.py` STAGE=1: DPO through the frozen EAGLE-L17 readout, LoRA r16 attn+mlp on layers
0..17, bf16, 400 steps, on his `britishness/` release and `corpus_shards/` replay. Two arms,
identical in everything except the replay term:

| | W_PREF | W_KFAC | W_REPLAY |
|---|---|---|---|
| `1:0:1` (`/workspace/sup_stage1_101`) | 1 | 0 | 1 |
| `1:0:0` (`/workspace/sup_stage1_100`) | 1 | 0 | 0 |

K-FAC is 0 in both. This session ran the replay-only arm first by choice, so the K-FAC term is
still unmeasured and the 1:3:1 recipe is still unrun.

**The headline: the register/dialect coupling that NEXT_0807 §2 called unbreakable is broken —
and replay is not what breaks it.**

## 1. The table

Free sampling at `N_GEN=512` (see §4 — the 128-generation numbers this session first produced are
not usable). Ranking and guard are deterministic over the release's 750 held-out and 200
in-sample guard rows.

| arm | raw ranking | brit_rate | marker hits | mean len | diversity | guard (in-sample) |
|---|---|---|---|---|---|---|
| base | 0.059 (44/750) | 0.104 | 509 | 66 w | 0.86 | 0.995 |
| base + tailored preamble | 0.765 (574/750) | 0.654 | — | 77 w | — | 0.995 |
| **1:0:0** ckpt100 | 0.988 (741/750) | **0.618** | 304 | **71 w** | 1.00 | 0.750 |
| 1:0:0 ckpt200 | 0.996 (747/750) | 0.331 | 254 | 69 w | 0.97 | **0.370** |
| 1:0:0 ckpt300 | 0.989 (742/750) | 0.472 | 89 | 77 w | 0.87 | 0.800 |
| 1:0:0 ckpt400 | 0.995 (746/750) | **0.772** | 145 | 76 w | 0.97 | 0.545 |
| 1:0:1 ckpt100 | 0.840 (630/750) | 0.190 | 526 | 68 w | 0.88 | 0.975 |
| 1:0:1 ckpt200 | 0.960 (720/750) | 0.304 | 257 | 71 w | 0.63 | 0.890 |
| 1:0:1 ckpt300 | 0.961 (721/750) | 0.235 | 298 | 69 w | 0.65 | 0.975 |
| **1:0:1** ckpt400 | 0.985 (739/750) | 0.391 | 373 | **70 w** | 0.93 | **0.950** |

## 2. The coupling is breakable, and the credit does not go where the prediction put it

`NEXT_0807` §2 and `results_0807` §4 established, on the no-replay DPO-P run, that dialect and
register were one coupled trade-off: mean length went 65 → 33 → 32 → 23 as brit_rate climbed, and
**no checkpoint reached preamble-level dialect at preserved register**. §4 turned that into a
falsifiable prediction — *replay should hold mean length near 65 while brit_rate climbs* — and
called it the actual scientific content of his recipe.

The prediction's outcome is split, and the split is the result:

- **Register is preserved in BOTH arms.** 68–77 words against a base of 66, at every checkpoint,
  with and without replay. There is no collapse anywhere on either trajectory.
- **So replay is not what preserves it.** The DPO-P run collapsed and both of these do not; what
  separates them is the recipe — DPO through the EAGLE-L17 readout with r16 attn+mlp LoRA on
  layers 0..17 in bf16, versus DPO-P at the full output with r8 mlp-only LoRA on all 24 layers in
  fp32. Replay is not the variable.
- **`1:0:0` ckpt100 reaches preamble-level dialect at preserved register**: brit_rate 0.618
  against the preamble's 0.654, at 71 words against its 77, while beating it on ranking 0.988 vs
  0.765. That is the thing §2 said no checkpoint could do.

**This attribution exists only because the matched control was run.** Compared against the 08-07
DPO-P baseline alone, `1:0:1` looks like replay breaking the coupling — the arms differ in loss,
LoRA rank and placement, and dtype as well as in replay, and all four move together. The control
is the same script with one term switched off, and it removes every one of those differences.

## 3. What replay actually buys: guard stability, paid for in dialect

| | best brit_rate | guard at that point | guard range over the run |
|---|---|---|---|
| 1:0:0 (no replay) | 0.772 | 0.545 | **0.370 – 0.800** |
| 1:0:1 (replay) | 0.391 | 0.950 | **0.890 – 0.975** |

Replay holds the guard within 0.11 of base for the whole run and costs roughly half the dialect.
Without it the guard swings to 0.370 — worse than anything in the DPO-P run, whose trough was
0.685 — and never recovers above 0.800. His *"without replay I get too much drift, even when
K-FAC EWC is present"* reproduces here, but the drift is in the **guard**, not in the register.

Ranking is blind to all of it: 0.988 / 0.996 / 0.989 / 0.995 across the no-replay arm while guard
goes 0.750 / 0.370 / 0.800 / 0.545. `results_0807` §5 found the same blindness through the `pos`
diagnostic on a different recipe; it reproduces here on the metric itself.

**The guard is in-sample** (his split puts all 200 `truth_guard` rows in TRAIN), so both arms are
breaking rows they are actively training on. This is not a generalisation failure.

## 4. Two measurement facts that invalidate this session's own first numbers

**`N_GEN=128` is not enough.** The first pass produced brit_rate trajectories that looked
non-monotone and dramatic — 0.248 / 0.342 / 0.155 / 0.436 for `1:0:1` — on 20–90 total marker
hits per checkpoint. At `N_GEN=512` the same checkpoints read 0.190 / 0.304 / 0.235 / 0.391 on
257–526 hits. The ckpt300 "collapse" was noise. The reference points moved too: **base is 0.104,
not 0.070, and the preamble is 0.654, not 0.600**, so 08-07's preamble comparisons rest on a
number ~8% low. Ranking and guard were identical across both passes, being deterministic.

**Marker hits are a second axis and they move.** In the no-replay arm the hit count falls
304 → 254 → 89 → 145 while brit_rate rises: the generations progressively contain fewer
marker-bearing words at all, so the ratio's denominator is shrinking in exactly the arm that
scores highest on it. The replay arm holds 257–526. `1:0:0` ckpt400's 0.772 rests on 145 hits and
ckpt300's 0.472 on 89; ckpt100's 0.618 on 304 is the best-supported point on that trajectory and
is the one this file leans on.

## 5. A metric defect in `sup_train.py`, now fixed

Every number `sup_train.py` printed was **reference-relative** — `_rank_acc` computed
`(la-ra) > (lb-rb)` against the adapter-off reference — which is exactly 0 at step 0 by
construction, since zero-init LoRA B makes policy == reference and every comparison is `0 > 0`
scored False. The step-0 line therefore read 0.000 across all four metrics, and
`guard_final_insample` was not the guard accuracy. `results_0807` §1 made this exact fix for
`sup_dpop.py`; this script never got it, and the 1:0:1 run was misread for ten minutes because of
it.

`_rank_acc` now returns raw alongside reference-relative. The fix validates itself: the `1:0:0`
run's step 0 reads raw_final 0.051 / guard_raw 0.995, matching the base model's 0.059 / 0.995,
while the ref columns sit at 0.000. Its step-200 guard_raw of 0.360 also matches `sup_eval.py`'s
independent 0.370 from a separate code path.

## 5a. The 1:3:1 arm — K-FAC adds nothing detectable over replay

Outstanding since `NEXT_0807`; K-FAC had never been estimated (`SKIP_KFAC=1` throughout 08-07 and
the first half of 08-09). Estimated here over the 70 factors falling inside stage 1's LoRA range,
`--placement model`, ~9 min. Confirmed live in the run header: `weights pref 1.0 kfac 3.0 replay
1.0`.

| step | raw ranking | brit_rate | len | guard | | 1:0:1 guard | 1:0:1 brit |
|---|---|---|---|---|---|---|---|
| 100 | 0.845 | 0.281 | 67 w | 0.975 | | 0.975 | 0.190 |
| 200 | 0.956 | 0.193 | 71 w | 0.945 | | 0.890 | 0.304 |
| 300 | 0.991 | 0.254 | 71 w | 0.980 | | 0.975 | 0.235 |
| 400 | 0.981 | 0.294 | 67 w | 0.925 | | 0.950 | 0.391 |
| mean | | **0.256** | | **0.956** | | **0.948** | **0.280** |

**The answer is negative.** Against replay alone, K-FAC leaves guard marginally more stable (worst
point 0.925 against 0.890; mean 0.956 against 0.948) and dialect marginally lower (0.256 against
0.280). Both differences sit inside the checkpoint-to-checkpoint scatter this instrument has
already demonstrated — §4's point that brit_rate bounces even at 512 generations applies here too.

So the three-term recipe reduces, on this data, to its replay term. That does not refute his
setup: §1's guard finding says replay is doing the work K-FAC was supposed to help with, and a
regulariser has nothing left to protect once the replay term is already holding the distribution.
It does mean **1:3:1 and 1:0:1 are not distinguishable here**, and a claim resting on the K-FAC
weight needs a setting where replay alone is insufficient.

Reported at one seed, like everything else in this file.

## 6. The EAGLE head was undertrained, and that was the whole gap

`NEXT_0807` recorded the L17 head at KL 3.1 / held-out top-1 agreement 0.352 on an 800-step
budget and flagged it as too weak to teach. At 5000 steps on the same data and the same
`POS_PER_STEP`, it reaches **KL 1.10 / agreement 0.653** — also past the 08-05 head's 0.462 on
the old data. Nothing else changed. `head_L17.json` here; ~25 min on a 96 GB card.

## 7. What this settles, and what it does not

Settled:

- The register/dialect coupling is breakable. `1:0:0` ckpt100: preamble-level dialect (0.618 vs
  0.654) at preserved register (71 w), beating the preamble on ranking (0.988 vs 0.765).
- Replay is not the mechanism that breaks it — both arms preserve register.
- Replay buys guard stability (0.890–0.975 vs 0.370–0.800) and costs roughly half the dialect.
- Ranking cannot see guard damage in this recipe either.
- The L17 head's weakness was a training-budget artifact.

Not settled:

1. **K-FAC is still unmeasured and 1:3:1 is still unrun.** This session ran 1:0:1 and 1:0:0 only.
   The recipe as he describes it is 1:3:1 and remains the open item.
2. **Which of the four recipe differences preserves register.** The DPO-P run collapsed and both
   EAGLE arms did not, but loss, LoRA rank, LoRA placement and dtype all differ at once. The
   cheapest next cut is `sup_dpop.py` at r16 attn+mlp on layers 0..17, which moves LoRA alone.
3. **Whether the guard/dialect trade is a frontier or an operating-point choice.** `1:0:0`
   ckpt100 (0.618 dialect / 0.750 guard) and `1:0:1` ckpt400 (0.391 / 0.950) may be two points on
   one curve reachable by tuning W_REPLAY, or two different curves. A W_REPLAY sweep at fixed
   steps would answer it.
4. **Everything here is one seed.** NEXT.md's standing direction — seed anything that goes in a
   write-up — is not satisfied by this file.
5. **The holdout still measures one seventh of the dataset** (`results_0807` §3): all 750 rows are
   `family=lexicon, form=qa`. Every ranking number above inherits that.
