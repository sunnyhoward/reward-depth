# Fast RLFR on AE/BE — results (2026-08-06, single seed)

Qwen3-1.7B, LoRA r=16, GRPO (8 prompts x 8 samples/step), 60 steps, KL 0.05, lr 1e-4, 192-token
completions. vLLM rollouts. ~8 min per run on one RTX 5090. Held-out = 50 prompts from 17
scenarios never seen in training, n=8 each (400 completions).

**Primary metric is the oracle BE rate on held-out generations, never the probe score.**

## 1. Decodability is flat, and it is not memorisation

Pooled probe AUROC on the frozen base is **0.90–0.99 at every layer, maximal at L0 (the embedding
layer, 0.988)**. Marker-token AUROC is 0.91–0.98 throughout — the probe fires on the marker
tokens, which is what the dense reward needs.

Holding out 30% of the AE/BE word pairs and refitting (`decodability_axisho.json`) gives the
**same flat curve, 0.992 at L0** on pairs the probe has never seen. So the flatness is a fact
about the property, not an artefact of letting the probe memorise a word list: a linear direction
over token embeddings generalises across word pairs, because `-our/-or` and `-ise/-ize` are
sub-token regularities.

AE/BE spelling is tokenizer-level. This repo predicted exactly that for `brit_lang`.

## 2. Everything installs, and probe rewards match the oracle

| run | reward | held-out BE (n=400) | coverage | KL | capability vs base |
|---|---|---|---|---|---|
| base | — | 0.292 | 0.77 | — | −1.347 |
| 1 | oracle (dictionary) | 0.799 | 0.68 | 0.07 | +0.004 |
| 4 | probe L4 pooled | 0.854 | 0.76 | 0.20 | — |
| 4 | probe L8 pooled | 0.855 | 0.76 | 0.22 | — |
| 4 | probe L12 pooled | 0.837 | 0.72 | 0.34 | — |
| 4 | probe L16 pooled | 0.818 | 0.66 | 0.21 | — |
| 4 | probe L20 pooled | 0.862 | 0.76 | 0.16 | — |
| 4 | **probe L24 pooled** | **0.917** | **0.87** | 0.14 | — |
| 5 | probe L12, **student** read | 0.843 | 0.73 | 0.29 | +0.055 |
| 3 | probe L12, **dense** | **0.500** | **0.00** | 3.31 | +0.351 |

Length stayed ~170 tokens and `distinct3` ~0.99 in every non-collapsed run; raw generations
(`results/gens/`) are fluent British prose, not marker spam.

**The headline:** pooled probe rewards on a frozen copy recover **100%+ of oracle-reward
performance** — the validation RLFR's setting made impossible. This was written down as the
fallback result before starting; it arrived as the main one.

## 3. The depth sweep is flat except at L24

L4–L20 sit in 0.818–0.862 with no trend. **L24 reaches 0.917 with coverage 0.87** — it writes
more dialect-committed prose rather than dodging markers. Single seed, so hold it loosely; it is
the only non-flat thing in the depth data and is the obvious thing to seed-replicate.

Note what this rules out: decodability is *equal* at all these depths (§1), so an L24 advantage
cannot be "the probe reads it better up there".

## 4. Dense per-token reward collapses — and the oracle caught it

Run 3 is the one novel RLFR ingredient and it failed outright:

```
probe score at L12:   -0.06  ->  +3.96      (6x any other run)
oracle BE rate:        0.29  ->   0.500     (0.500 = NO markers at all)
marker coverage:       0.74  ->   0.00
probe-oracle rho:     +0.51  ->  -0.03
KL from base:                      3.31
```

The policy learned to emit **no dialect markers whatsoever**, in fluent text (capability *rose*
+0.351). The probe reward went up monotonically the whole time.

Mechanism: the dense advantage z-normalises the per-token probe logit across the group's whole
token pool. American markers carry a large negative logit and drag the pool mean down, so an
ordinary dialect-neutral token sits *above* the mean and earns positive advantage. "Emit nothing
committal" is therefore a uniformly positive-advantage policy, and it is far easier to reach than
"emit British markers". The fix is to treat the dense signal as shaping on top of the
completion-level advantage rather than as a replacement for it — closer to what RLFR actually
does (dense intrinsic reward *combined into* the return).

**This is the plot RLFR could not make** (`fig5_hacking.png`), and on its first outing it caught
a real reward hack that the probe score alone would have reported as a spectacular success.

## 5. Reading the probe on the student did not break it

Run 5 reaches 0.843, matching the frozen-read runs. That contradicts this repo's standing line
("self-read backprop from the probe is gameable everywhere we tried it") and agrees with
Goodfire's own report that their probes transfer to policy activations.

Both are probably true, and the distinguishing variable is the gradient path: method 1
backpropagated *through* the probe into activations; here the probe only emits a scalar into
GRPO, so the policy can reach it only through emitted tokens. That is this repo's own stated rule
for what works. The visible cost is drift — KL 0.29 vs 0.07 for the oracle run.

## 6. Caveats

- **Single seed throughout.** The L24 result especially.
- **Coverage decay is a real, small hack channel.** `be_rate` is 0.5 when a completion carries no
  markers, so suppressing markers is weakly rewarded. Coverage falls 0.77 -> 0.61–0.76 in the
  pooled runs (and to 0.00 in the collapsed dense run, where it is the whole story). BE rates well
  above 0.5 mean it is not what drives the main result, but it would grow with more steps.
- **The oracle trades recall for precision**: ~28 of 250 axes dropped as ambiguous.
- Two methodological traps cost real time and are documented in the README: a fixed Adam step
  budget made the decodability curve look structured when it was optimizer noise (the
  depth-vs-readout-competence confound from `NEXT.md`), fixed by LBFGS + per-layer L2 selection on
  a val split; and `lr=1e-5` left the LoRA policy completely flat for 60 steps.

## Figures

`results/plots/` — `fig1_decodability.png`, `fig2_training.png`, `fig3_depth.png`,
`fig4_money.png` (two stacked panels, never a dual axis), `fig5_hacking.png`.
