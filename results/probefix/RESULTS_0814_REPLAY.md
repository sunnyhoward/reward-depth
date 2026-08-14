# The replay term is a competing objective, not an anchor — and removing it helps slightly

*2026-08-14. `Qwen3.5-4B`, attach block 20, `brit_dose20.jsonl`, seed 0, 600 steps, all four arms
trained in ONE environment in one run. Scored on `false_friend` and `style` only
(RESULTS_0814_ELICITATION.md). Judge `Qwen3-32B`, blind, 240 items per family, 0 unparsed.
Scripts: `probefix/run_pf4b_replay.sh`, `probefix/run_pf4b_replay_score.sh`.*

## 0. The premise that turned out to be false

The sweep was proposed on the reading that the two-stage arms carry generative replay and P1 does
not. **They all do.** `pf_train.py`'s header says `MODE=final` carries "the same replay term",
`W_REPLAY` defaults to 1, and no probefix run script has ever set it. The training log prints it:

```
[pf-final] Qwen/Qwen3.5-4B blocks=32 | read L=20 (last) | lora 0..31 21.2M | pref hinge w=1.0 replay nll w=1.0
```

That is the P1 configuration. So replay was never what separated C1 from P1 — the only
differences are the stage-1 merge and the LoRA range. This sweep is therefore the first
replay-off control probefix has ever had; the only other one in the repo is `E_read10_lora10_noreplay`
in the UF study, where it was indistinguishable from replay-on.

## 1. Replay is NOT inert — but it does not behave like an anchor

Training-side, P1 with and without it:

| step | margin, replay off | margin, replay on | KL from base, off | KL from base, on |
|---|---|---|---|---|
| 100 | +26.2 | +15.0 | 0.116 | 0.188 |
| 300 | +50.9 | +25.9 | 0.203 | 0.200 |
| 500 | +60.5 | +31.8 | 0.155 | 0.212 |

**It halves margin inflation** — the pathology DPOP exists to fight, where the margin is won by
dragging both sides down until the model stops emitting English. At 16 scored tokens a step.

**And it increases drift from base rather than reducing it.** That is backwards for an anchor,
and the reason is in the loss: `REPLAY_LOSS=nll` trains the policy to predict the replay corpus.
Fitting that corpus *better than base does* moves the policy away from base, which is what the KL
column reports. It suppresses margin inflation by competing for the gradient, not by constraining
the policy toward the reference.

`REPLAY_LOSS=kl` — implemented, never used — is the version that would actually anchor: it takes
KL against the adapter-disabled base on the same window. **Every number in this project that
describes the replay term as an "anchor" or a "prior" is describing the NLL form, which is not
one.**

## 2. Behaviour: removing replay helps slightly, 4/4

British score among engaged items, judge 0-100:

| family | pair | replay OFF | replay ON | Δ | ±SE | SEs |
|---|---|---|---|---|---|---|
| false_friend | C1 | **75.8** | 68.1 | +7.7 | 6.9 | 1.1 |
| false_friend | P1 | **79.5** | 75.4 | +4.1 | 6.1 | 0.7 |
| style | C1 | **67.3** | 65.2 | +2.1 | 2.8 | 0.8 |
| style | P1 | **73.4** | 72.7 | +0.7 | 2.1 | 0.3 |

**All four contrasts favour replay-off, and not one of them clears noise on its own.** The sign
consistency is the whole signal; the magnitudes are not resolvable at one seed. Read it as "the
replay term is not buying an install, and may cost a little", not as a quantity.

Coherence is unaffected in every cell (91.9-95.0 on `false_friend`, 81.4-89.0 on `style`), so
this is not the anchor preventing degeneration — nothing degenerates in any of these four arms.

## 3. P1 still beats C1, and this is the cleanest version of that result

Within matched replay setting, same environment, same seed, both sides retrained in one run:

| family | setting | P1 − C1 |
|---|---|---|
| false_friend | replay off | +3.8 ± 6.0 (0.6 SE) |
| false_friend | replay on | +7.3 ± 7.0 (1.0 SE) |
| style | replay off | +6.1 ± 2.6 (**2.3 SE**) |
| style | replay on | +7.5 ± 2.4 (**3.2 SE**) |

The 0814 judge table put this at +5.0 ± 2.4 against banked adapters. Here it is larger and
better separated, with the environment objection removed. **The two-stage recipe does not beat
plain all-layers DPOP on free generation, with or without replay.**

## 4. The banked adapters reproduce in this environment

Not the point of the sweep, but it is the check that unblocks everything else:

| family | arm | banked (0813 env) | fresh (today) |
|---|---|---|---|
| false_friend | C1 | 64.9 | 68.1 |
| false_friend | P1 | 74.2 | 75.4 |
| false_friend | base | 37.0 | 36.0 |
| style | C1 | 67.1 | 65.2 |
| style | P1 | 72.1 | 72.7 |
| style | base | 47.3 | 46.6 |

Every cell within ~3 points, and base within 1.3. **The HF-banked adapters can be used directly**
— which means the generation-side rank-1 ablation (0813 open item 6) needs no retraining, and the
0811 environment objection does not apply to `sunnyhoward/reward-depth-probefix4b`.

## 5. Scope limits

1. **Single seed per cell.** Four consistent signs at n=1 is suggestive, not a measurement.
2. `REPLAY_TOK=16` throughout. The obvious follow-up is a weighted arm (`REPLAY_TOK=256`), which
   would distinguish "the anchor is too small to matter" from "the anchor is the wrong shape".
   §1 argues for the second, but only the trajectory columns support that.
3. `REPLAY_LOSS=kl` remains untested. It is the cell that would test the anchor as an anchor.
4. Two families only. Correct per the elicitation result, but it means `false_friend` (n=48,
   SE 4-7) carries the terminology half of the claim alone.
5. D2 is absent — it is the one arm not in the HF bank and was not retrained here.
