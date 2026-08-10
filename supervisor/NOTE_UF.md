# The UF port of the supervisor recipe, attached at the probe elbow (2026-08-10)

**Written before any arm has run.** The predictions in §4 are pre-registered so that whichever way
the numbers fall, the reading is not chosen after seeing them. This project's record is that
single-seed positives churn and the claims that survive are the ones whose check was specified
first (`NEXT.md`: *"Report nothing before its check has run"*).

## 1. What is being ported

`supervisor/` reimplements the setup he reports working, and on his `britishness/` release it
worked: `sup_train.py` STAGE=1 — DPO through a **frozen EAGLE readout at block L**, LoRA r16
attn+mlp on **layers 0..L**, bf16, plus his generative-replay term — reached preamble-level dialect
at preserved register (`results_0809/RESULTS.md` §2). britishness is a constructed set: single-word
am|br swaps on templated prompts, with a lexical floor near ceiling.

The question here is whether the recipe survives contact with a **real** preference, and whether
the attach layer should be chosen by **where the preference is decodable** rather than fixed at 17.

## 2. The dataset, and the one filter that is ours

`sup_uf.py` materialises UltraFeedback (`allenai/ultrafeedback_binarized_cleaned`, `train_prefs`)
into his release's schema, so `sup_common.load_split` reads it unchanged: **6000 pairs, 5250 train
/ 750 held out** by prompt hash, rendered with the Qwen3.5 chat template (`enable_thinking=False`,
so the empty `<think>` block is present exactly as in his rows).

Filters: the repo's UF filters (both sides present, non-identical, GPT-4 score margin ≥ 1.0), plus
**a new length cap** — both rendered sides ≤ 512 tokens, prompt ≤ 192. The cap is not cosmetic:
`sup_common.encode` right-truncates at a fixed length with the prompt at the front, so an
overrunning pair would have its longer side cut and its shorter side not, injecting a length
artefact into the very margin being trained. It costs half the margin-passing pool (kept 49.3%,
mean length 273 tok against 496 for the unfiltered pool), so **this is a short-pair subset of UF
and every number below inherits that.**

The length confound is present and must be read alongside everything: chosen is longer in **61.2%**
of pairs (mean 171 vs 122 completion tokens).

## 3. Where UF is decodable in this model

`decodability/` sweep on the **same file** (`uf_sup`, so the probe's held-out set and the trainer's
eval set are one set), Qwen3.5-2B, chat render, 512-token window, 3 seeds
(`results/decodability/scalar_qwen3.5-2b_uf_sup_chat.json`):

| read point (dec convention, 0 = embeddings) | 0 | 1 | 5 | 6 | 10 | 11 | 14 | 15 | 18 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear probe, last-token | .500 | .703 | .749 | .740 | .771 | **.795** | .791 | **.800** | .783 | .784 | .779 |

- **L\* = 11** (earliest point within 1 SE of the curve's max), fractional depth 0.46; peak 0.800.
- **This replicates the Tulu-3-8B UF signature** it was never fitted to: plateau 0.799 from L12/32
  there, 0.800 from L11/24 here, and a length-only cheat floor of **0.623** against that paper's
  0.62. Lexical floor (bag-of-token-ids, no model) 0.644.
- Convergent across instrument: L\* = 10–11 in all four (read protocol × rung) cells.
- **Caveat that limits every claim here:** the plateau sits only ~0.18 above the length-only floor.
  UF's decodable core is substantially length, and no arm below is evidence against that.

**Convention conversion, the one that would silently cost an arm.** `decodability` indexes read
point 0 as the embedding output, so its `L` is the output of block `L−1`. `sup_train.py` reads
`BLOCKS[LAYER]`. So **dec L\* = 11 → `SUP_LAYER=10`**.

## 4. The arms, and why four

The ask is "attach at L\* instead of deep" = arms A and B. Those two alone cannot separate three
things that all move when `LAYER` moves:

1. **read depth** — what the DPO gradient reads;
2. **write range** — the recipe puts LoRA on 0..LAYER, so a shallower read is also *fewer
   trainable blocks*;
3. **readout competence** — an EAGLE head at block 5 reconstructs the output distribution far worse
   than one at block 21, **by construction**. This is the confound that has blocked every depth
   claim in this repo (`results_0805.md` §2/§4; `NEXT_0809.md`: *"the EAGLE attach sweep does not
   substitute for it"*).

`LORA_MAX` was added to `sup_train.py` (defaults to `LAYER`, so the britishness path is untouched)
to break (1) from (2). Arm C breaks (1) from (3).

> **CORRECTION (written 2026-08-10, after D was already queued): (1) and (2) cannot be broken
> apart, and `LORA_MAX` does not do it.** The stage-1 loss is read from `h_L`, and `h_L` is a
> function of blocks 0..L *only* — every block above the read point is outside the loss's
> computation graph, so it receives exactly zero preference gradient no matter what `LORA_MAX`
> says. Verified on a read-at-10 / LoRA-0..21 model with `torch.autograd.grad(...,
> allow_unused=True)`: blocks 0/5/9/10 return gradients (norms .237/.275/.311/.383), blocks
> 11/15/21 return `None` — not in the graph. Arm D's upper blocks therefore move under the
> **replay term alone**, so D ≈ A is the expected outcome and D is not a capacity control.
>
> The consequence is structural and worth stating positively: **in this recipe, attaching shallow
> IS training less of the network on the preference.** Read depth and preference-write range are
> one variable. B's advantage over A cannot be decomposed into "better readout" vs "more capacity"
> by any arm inside this recipe; separating them would need a different objective (e.g. reading at
> L while also applying a loss at the output). D remains worth running as the narrower control it
> actually is: does letting the upper stack move under replay change the install?

| arm | read block | LoRA | isolates |
|---|---|---|---|
| **A** | 10 (= L\*) | 0..10 | the ask: the recipe moved down to the elbow |
| **B** | 21 (late) | 0..21 | the ask: the recipe attached deep |
| **C** | 5 (below the elbow) | 0..5 | decodability vs depth/competence |
| **D** | 10 | 0..21 | ~~read depth at B's parameter count~~ — see the correction below |

All arms: 400 steps, lr 1e-4, β 0.1, 6 pairs/step, weights **1 : 0 : 1** (pref : K-FAC : replay) —
`results_0809` §5a found 1:3:1 and 1:0:1 indistinguishable on this instrument, so K-FAC is off and
the recipe reduces to its replay term.

**The pre-registered discriminator.** Probe accuracy *plateaus* from block 10 (.749 @ 5 → .795 @ 10
→ .784 @ 21) while head competence rises *monotonically* with depth. The two hypotheses therefore
disagree on exactly one comparison:

- install tracks **decodability** ⇒ C < A ≈ B
- install tracks **readout competence** ⇒ C < A < B

and the Occam prediction this repo was built to test (`STATE.md`) adds: **A should generalise
better than B out of domain**, even where they tie in domain.

Against that prediction stands the strongest existing evidence: the phase-3/5 read-depth test on
UF was **negative** — four arms ordered by *probe accuracy*, not by depth, and the deeper, more
length-aligned probe transferred *better* (`STATE.md` §"Read depth — TESTED. Negative"). If that
result generalises from soft-label DPO to this recipe, the expected outcome is A ≈ B in domain and
**no OOD advantage for A**.

## 5. What decides it

`sup_eval_pref.py`, on held-out UF (750) and two OOD sets: **offsetbias** (constructed so surface
heuristics point at the *rejected* side — the direct test of an arm that learned "prefer longer")
and **rewardbench2** (6 domains kept separate, so the safety/factuality columns that every phase-3
UF arm degraded stay visible).

Reported per arm, per set: raw ranking, implicit (reference-relative) ranking — both at the final
output and at the EAGLE readout — chosen-side log-prob displacement, and **accuracy split by which
side is longer**, beside the "prefer longer" cheat rate. An arm that beats the cheat floor only on
its chosen-longer half has learned length, and on this dataset that is the default hypothesis, not
the fallback one.

**Do not read the in-training eval line as a trajectory.** `sup_train.evaluate()` calls
`rgen.sample(val_rows, EVAL_N)`, so with 750 held-out rows and `EVAL_N=128` every eval scores a
DIFFERENT 128-row subset, drawn from the same generator that draws training batches. Successive
evals therefore differ by subset as well as by training, at SE ≈ 0.044 each, and a swing of 0.10–0.18
between adjacent evals is unremarkable. (Arm A's first run showed 0.586 → 0.492 → 0.672 on
`acc_final` across steps 100/200/300, which looks like a dissociation and is not evidence of one.)
Every number in the results table comes from `sup_eval_pref.py` on the full, deterministic 750
instead. This affects the britishness runs too, at `EVAL_N=256` of the same 750.

**Head competence is reported beside every arm as the covariate** (`head_L*.json`: held-out KL and
top-1 agreement), because without it a depth ordering cannot be distinguished from a readout
ordering.
