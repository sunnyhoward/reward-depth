# The addition ladder: which parts of a+b are decodable, and when (2026-08-11)

Follows `RESULTS_ccstrata.md`. That file left one thing open: from 0.42 to 0.72 depth a fitted
probe reads styc `corr_*` correctness at 0.77 out of a representation whose sum the model cannot
yet express at all (family B and the logit lens both at chance until L25). This names the
candidate intermediates and dates each one.

`cc_ladder.py` (4B, 4000 addition prompts a,b ~ U[10,99], linear multinomial probe, group 5-fold,
3 seeds). Read point is the **last token of the chat generation prompt** — no completion in the
input, so no answer, right or wrong, is ever visible. Figure:
`plots/cc_ladder_qwen3-4b.png`.

## 0. The read point is clean

At L0 the read token's embedding is identical for every item, so every target must sit at its
majority rate. Measured: `carry` 0.545 (majority 0.545), `ans_hund` 0.596 (0.596), all four
10-way targets 0.094–0.100 (0.104). No leakage.

## 1. Half the ladder is decodable by the PROBE, not computed by the model

This is the correction that has to come before any reading of the figure. `carry` is a threshold
on `d_a + d_b` and `ans_hund` a threshold on `a + b` — both **linear in operand digit values** —
so a probe produces them from any representation that merely *contains* the digits. The operand
digits saturate at read point **L2** (`a_units` 0.996), so accuracy at L2 is what the probe can do
with no arithmetic done for it. Against that baseline rather than against chance:

| target | bag floor | @L2 | peak | at depth | top | gain over L2 |
| --- | --- | --- | --- | --- | --- | --- |
| `carry` = s_u ≥ 10 | 0.796 | 0.962 | 0.983 | 0.75 | 0.959 | **+0.02** |
| `ans_hund` = a+b ≥ 100 | 0.806 | 0.948 | 0.973 | 0.78 | 0.964 | **+0.03** |
| `s_t` = a//10 + b//10 | 0.205 | 0.410 | 0.707 | 0.61 | 0.535 | +0.30 |
| `s_u` = a%10 + b%10 | 0.189 | 0.361 | 0.666 | 0.67 | 0.517 | +0.31 |
| `ans_units` = s_u mod 10 | 0.109 | 0.185 | 0.433 | 0.58 | 0.329 | +0.25 |
| `ans_tens` (needs carry) | 0.119 | 0.196 | 0.381 | 0.67 | 0.302 | +0.19 |

**The carry is not a finding.** Its 0.96 at L2 and 0.96 at the top say the model contributes
nothing to it that a linear readout could not already do from the digits. Reporting `L*(carry) =
0.06` against a chance line would have been a statement about the probe. Same for the hundreds
digit. Note also that the *bag-of-token-ids* floor is already 0.80 on both, for the same reason:
digit counts correlate with the magnitude of the sum.

## 2. What the model actually builds, it builds between 0.45 and 0.67

The four targets with real gain all rise in the same window and **peak before the top of the
model**, then decay:

- `s_t` peaks 0.707 at depth 0.61 → 0.535 at the top
- `s_u` peaks 0.666 at 0.67 → 0.517
- `ans_units` peaks 0.433 at 0.58 → 0.329
- `ans_tens` peaks 0.381 at 0.67 → 0.302

Every one loses 8–15 points between its peak and the top block. The intermediate is discarded once
it has been used.

Placing that against the two earlier landmarks, on one axis:

| depth | what happens |
| --- | --- |
| 0.06 | operands present at the read position; carry and hundreds digit already probe-computable |
| 0.45–0.67 | the column sums and the answer digits are built, peaking at 0.58–0.67 |
| 0.69 | the model's own sum reaches the unembedding (`cc_when_computed.py`) |
| 0.75 | the preference probe jumps (`cc_strata.py`) |

The order is right. The preference signal completes **after** the arithmetic does, not with it.

## 3. The sum is a magnitude before it is digits

`s_u` (19-way, the raw column sum) is decoded at 0.67 while `ans_units` (10-way, strictly easier
as a classification, and a deterministic function of `s_u`) reaches only 0.43. A readout that
knew `s_u` would know `s_u mod 10`, so the gap is informative: **the representation supports a
linear readout of the sum's magnitude but not of its modular reduction.** `mod 10` wraps 10 → 0,
which is exactly what a linear probe on a magnitude code cannot do.

## 4. The addition is not finished at the prompt — it is finished while writing

Neither answer digit exceeds 0.44 at *any* depth at this read point, yet `cc_when_computed.py`
measured the model's own top-1 units digit at **0.98** at teacher-forced positions inside the
answer. Both are true: at the last prompt token the model has the sum as a magnitude and has not
resolved it into digits; the digits are produced position by position as it writes, each one
conditioned on those already emitted.

This bounds what the depth axis alone can show. `cc_strata.py` reads the completion tokens,
`cc_ladder.py` reads the prompt token — a difference in *position*, not depth, and §3–4 say
position is doing real work here.

## 5. What this does and does not give the sequential-probe programme

It gives a genuine ordering with named rungs, which is what was wanted: operands at 0.06, the
column sums and answer digits built at 0.45–0.67, the model's own sum expressible at 0.69, the
preference readable at 0.75. Probes at those depths are reading demonstrably different things.

Two honest limits before anything is built on it:

1. **Two rungs are free.** `carry` and `ans_hund` carry no information about the model. Any
   cascade that "discovers" them at L2 has discovered the readout's own power. Every new target
   needs the operand-digit baseline computed before its L\* is quoted.
2. **The rungs that are real move together**, at 0.45–0.67, rather than in the dependency order
   they were listed in. There is no clean `s_u` → `carry` → `units` → `tens` staircase in depth.
   The dependency order shows up in *magnitude of gain* (+0.31, +0.30, +0.25, +0.19) far more than
   in L\*.

The next thing worth measuring is the ladder **at the completion tokens** rather than the prompt
token, on the same items as `cc_strata.py`. That is the read position where the preference probe
lives, and §4 says it is where the digits actually get resolved. It reuses `cc_ladder.py`'s
fitter unchanged; only the activation cache changes.
