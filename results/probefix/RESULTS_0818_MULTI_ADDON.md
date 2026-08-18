# All-layer steering buys nothing; the learned add-on breaks the `style` ceiling and the text with it

*2026-08-18. 768 items (8 arms × 96), 16 blind Claude agents, same rubric/protocol as the earlier
0818 passes. `results/probefix_cjudge_multi/`. Two questions: does steering EVERY layer at once beat
steering the best single one, and does a LEARNED add-on (`h_L ← h_L + A_L·z`, `PREF=dpo`) do what
the fitted direction cannot. Unconditional column, ± 1 SE, single seed.*

| arm | ff | ff coh | style | style coh |
|---|---|---|---|---|
| base | 40.1 ± 4 | 79.3 | 24.6 ± 1 | 79.3 |
| `steer_L20` (single layer, α=.3) | 62.5 ± 4 | 77.2 | 32.1 ± 3 | 77.9 |
| `steer_all32_dn` (32 layers, α/32) | 63.2 ± 4 | 79.1 | 32.4 ± 3 | 77.1 |
| `steer_all32_lo` (32 layers, α=.01) | 63.6 ± 4 | 76.1 | 35.3 ± 3 | 77.2 |
| `steer_mid` (L8–24, α=.03) | 60.6 ± 4 | 79.4 | 31.9 ± 2 | 77.4 |
| `addonD_L4` (learned, `dpo`) | 57.7 ± 3 | **53.4** | **86.1 ± 1** | **59.4** |
| `addonD_L20` (learned, `dpo`) | 66.2 ± 3 | **41.2** | **81.0 ± 2** | **61.0** |
| `P1_r1` (plain DPOP) | 69.3 ± 4 | 77.3 | 66.0 ± 2 | 79.7 |

---

## 1. Simultaneous multi-layer steering is a null

| contrast | `false_friend` | `style` |
|---|---|---|
| `steer_all32_dn` − `steer_L20` | **+0.8 ± 2.8** | **+0.2 ± 1.5** |
| `steer_all32_lo` − `steer_L20` | +1.2 ± 2.6 | +3.2 ± 1.5 (2.2 SE) |
| `steer_mid` − `steer_L20` | −1.8 ± 1.8 | −0.2 ± 1.2 |

**Spreading the intervention over all 32 blocks does not beat concentrating it at the best single
layer**, at matched total norm (`div_n`), at a 30× smaller per-layer dose (`a0.01`), or over the
mid-band alone. Every contrast is inside noise bar one marginal `style` cell.

**The lexical meter said otherwise and was wrong again.** It read `all32_dn` at 0.628 against
`steer_L20`'s 0.582 — an apparent win over the best single layer and near DPOP's 0.638. Judged, the
difference is **+0.8 ± 2.8**. That is the second time today a `pf_famlex` number has not survived
contact with the judge (`RESULTS_0818_STEER.md` §5.1 was the first).

**Reading**: the mid-stack plateau in `RESULTS_0818_STEER.md` §1 is **one edit re-expressible at
many depths**, not k independent handles — k coordinated copies of it buy nothing over one. The
un-normalised high dose confirms the additions compound rather than average: `all32_a0.1` produces
**31 characters with zero dialect markers**, total collapse. Note that `pf_leakage.py` scored that
cell **0.000** — with almost no text there is nothing for the regex to catch. Length was the only
meter that saw it.

## 2. The learned add-on breaks the `style` ceiling — by 20 points over DPOP

Every activation-space method measured until now sat at +6 to +8 over base on `style`, against
DPOP's +42 (`RESULTS_0818_STEER.md` §5.3, four instruments agreeing). A single trained 2560-dim
vector does this:

| contrast | `false_friend` | `style` |
|---|---|---|
| `addonD_L4` − base | +17.6 ± 5.1 | **+61.5 ± 1.6 (38.6 SE)** |
| `addonD_L20` − base | **+26.1 ± 5.0** | +56.4 ± 2.1 (27.1 SE) |
| `addonD_L4` − `steer_L20` | −4.7 ± 3.7 | **+54.0 ± 2.6 (20.5 SE)** |

`addonD_L4` reaches **86.1** on `style` where plain DPOP reaches 66.0 and the best fitted-direction
steering reaches 32.1. **This is the first result in the project to beat output-level DPOP on the
register family, and it does so by 20 points with 2560 trained parameters and a frozen model.**

The `style` ceiling was therefore a property of the *fitted* direction and of the training
objectives tried, **not** of activation space. That is the opposite of what
`RESULTS_0818_STEER.md` §5.3 concluded from four instruments, and it is the strongest argument yet
that this project's activation-space negatives are about method, not medium.

## 3. And it degrades the text, in a way only the judge caught

Coherence: `addonD_L4` **59.4** on `style` and **53.4** on `false_friend`; `addonD_L20` **41.2** on
`false_friend` — near the wrecked `MD_L20_floor_s300` (34.0). Base is 79.3.

**`pf_leakage.py` reads 0.000 for both add-on arms.** The text carries no chat-template artefacts at
all; it is fluent, correctly formatted British English that **drifts off the task**. Example, prompt
"My manager has asked me to work this weekend… Draft a reply":

> base/DPOP: a reply declining, with an offer to make up hours.
> `addonD_L4`: *"That sounds perfectly manageable – I'd hate to juggle too much at once, but if you
> could perhaps squeeze in a short note about how I'll make sure to keep things as light as
> possible this weekend?"*

Impeccably British register, not a reply to the prompt. This is Goodhart with a specific mechanism:
`A_L·z` pushes the residual toward the register direction hard enough that content follows it. The
regex meter cannot see this and the marker meter would score it *well*.

## 4. What follows

1. **`z` is a free parameter and was never swept.** `A_L` is trained once; `z` scales it at
   generation with no retraining, so the whole trade-off curve is one generation pass per point.
   `z=1.0` was an arbitrary choice and sits past the knee. **A `z` sweep is the cheapest experiment
   in the project right now** and it is running (`Z_GEN=0.25,0.5,0.75`).
2. **If a `z` exists with `style` ≥ 66 (DPOP's level) at coherence ≥ 77**, the add-on dominates DPOP
   on register at 2560 parameters and no weight updates, and that is a paper result on its own.
3. **`false_friend` and `style` invert between the two methods.** The fitted direction gets `ff` to
   62.5 and `style` to 32.1; the learned add-on gets `style` to 86.1 and `ff` to 57.7 (L4). The
   families are not just different in difficulty, they are reached by different mechanisms.
4. Single seed; `z=1.0` only; `PREF=nll` cells excluded as length-confounded
   (`RESULTS_0818_STEER.md` §3).
