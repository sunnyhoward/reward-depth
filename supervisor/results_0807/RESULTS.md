# Supervisor recipe, stage 1 — his DPO-P sheet, run verbatim (2026-08-07)

Qwen3.5-2B + his `britishness/` release + his `corpus_shards/`. Scripts: `supervisor/sup_dpop.py`
(training), `supervisor/sup_eval.py` (both metrics). Raw artifacts in this directory.

**The headline reproduces. The headline is also, in this dataset, a narrower claim than it
looks — and the reported metric is blind to every cost the run incurs.**

## 1. What was run

His settings sheet, verbatim: full-model DPO-Positive, MLP-only LoRA r=8 on all 24 layers, fp32
with tf32 off, β 0.1, `dpop_lambda` 50, lr 1e-4, AdamW, batch 8, 300 steps, grad clip 1.0,
adapter-off reference policy, component-balanced batch scheduler (culture/language/style/truth,
equal mass, ≤1 -ise/-ize row per batch), seed 0 / schedule_seed 4242.

**No EAGLE head, no K-FAC, no replay** — the sheet omits all three, so this run omits them. That
makes it the no-regulariser baseline his "without replay I get too much drift" claim needs, and
§4 below is that drift showing up on schedule.

Two deviations, both flagged in `sup_dpop.py`:

- **Gradient checkpointing.** fp32 at batch 8 OOMs against the two decodability jobs that held
  66 GB of the GPU. Recompute instead of store — arithmetic unchanged, so his dtype and batch
  size stay exactly as specified rather than being quietly reduced.
- **Three eval numbers, not one.** The sheet's margin accuracy is reference-relative and is
  therefore **exactly 0 at step 0 by construction** (zero-init LoRA B ⇒ policy == reference). His
  730/735 and the 517/735 preamble baseline must be *raw* ranking, since a preamble has no
  reference policy to be relative to. Raw is the comparable column throughout.

Two judgement calls the sheet could not make for us, both marked JUDGEMENT in the file: his four
components onto the release's seven `family` values (culture/style/truth one-to-one, everything
lexical → `language`), and `quota_key=spelling_contrast` onto the `group` field (`spell_ise` +
`ize_ise`). He said the scheduler matters; if a number here disagrees with his, look here first.

## 2. The headline reproduces

| step | raw ranking (750) | ref-relative | guard (in-sample, 200) | `pos` (25-step mean) |
|---|---|---|---|---|
| 0 | 0.059 (44/750) | 0.000 | 0.995 (199/200) | 0.000 |
| 25 | 0.859 (644/750) | 0.979 | 0.975 (195/200) | 0.017 |
| 50 | 0.961 (721/750) | 0.987 | 0.885 (177/200) | 0.087 |
| 75 | 0.979 (734/750) | 0.991 | 0.710 (142/200) | 1.180 |
| 100 | 0.980 (735/750) | 0.991 | 0.785 (157/200) | 0.185 |
| 125 | 0.989 (742/750) | 0.992 | 0.740 (148/200) | 0.856 |
| 150 | 0.991 (743/750) | 0.989 | 0.800 (160/200) | 0.315 |
| 175 | 0.989 (742/750) | 0.993 | **0.685 (137/200)** | 0.404 |
| 200 | 0.991 (743/750) | 0.991 | 0.775 (155/200) | 0.369 |
| 225 | **0.995 (746/750)** | 0.993 | 0.765 (153/200) | 0.586 |
| 250 | 0.989 (742/750) | 0.993 | 0.805 (161/200) | 0.437 |
| 275 | 0.988 (741/750) | 0.993 | 0.905 (181/200) | 0.051 |
| 300 | 0.992 (744/750) | 0.995 | 0.840 (168/200) | 0.540 |

His 730/735 = 0.993. This run passes it by step 225 and ends at 744/750. The base model sits at
0.059 — it prefers the **American** side of 94% of held-out pairs — so there is a real floor
being cleared, not a prior being confirmed.

`sup_eval.py` reproduces the training loop's numbers from a separate code path at ckpt25
(0.859 raw / 195-200 guard) and ckpt100 (0.980 raw / 157-200 guard), so the table is not an
artifact of the training-time eval.

## 3. The holdout only measures one seventh of the dataset

**This is the finding that reframes the headline.** The release's `reserved_for_eval` split:

```
family:  lexicon 750        culture 0, style 0, truth_dialect 0,
                            expression 0, false_friend 0, spelling_control 0
form:    qa 750             no continuation, dialogue, instruction, freeform
group:   spell_ise 384 (51%), lex 159, spell_irregular 78, spell_our 51,
         spell_ll 48, spell_re 21, spell_ce 9
```

Every held-out row is **lexicon, in QA form**, and half are the `-ise/-ize` contrast — the exact
group his scheduler caps at ≤1 per batch, presumably because it is so dominant. So "746/750"
means *the model learned British spelling and word choice in short QA answers*. The three
components his scheduler goes to the trouble of balancing during training — culture, style,
truth — are **never evaluated by it at all**.

Relatedly: all 200 `truth_guard` rows sit in **train**, none in the holdout. A held-out score is
therefore structurally incapable of showing the guard damage in §5. If his 735 is this split, he
would see 730/735 and no sign of any of it.

## 4. The behavioural check: one trade-off curve, not two effects

Free sampling, 128 greedy generations on held-out prompts, scored by the marker lexicon read off
the release's own `item` field (384 am / 386 br markers).

| arm | raw ranking | brit_rate | mean len | guard |
|---|---|---|---|---|
| base | 0.059 | 0.070 | 65 w | 199/200 |
| **ckpt25** | 0.859 | **0.269** | **65 w** | **195/200** |
| base + tailored preamble | 0.765 (574/750) | 0.600 | 77 w | 199/200 |
| **ckpt100** | 0.980 | **0.919** | **23 w** | **157/200** |

The preamble baseline reproduces his: 574/750 here against his 517/735 (mine is a system turn;
his wording is unknown).

**The install is real** — brit_rate moves 0.070 → 0.919, so this is *not* the §14 dissociation
where ranking climbs and behaviour stays flat. That is worth having after four prior sightings.

**But brit_rate and register move together, and that is the whole story.** ckpt25 keeps the
register perfectly intact (65 words, same hedging, still emitting markdown bold) and gets 0.269.
ckpt100 reaches 0.919 only by collapsing to 23-word single-sentence answers — i.e. onto the
surface form of the holdout, which is *entirely* short QA (§3).

```
base      | "Since I don't know the specific story, movie, or book you are referring
          |  to, I can't give you the exact details…"                          (65 w)
ckpt25    | "It's a bit of a mystery, but the scent of almonds on a shelf is
          |  almost certainly **almond oil**. While it's a…"                   (65 w)
preamble  | "Ah, the morning of the front room! It was a most delightful affair,
          |  one that required a little patience and a…"                       (77 w)
ckpt100   | "The priest did not do anything to four children before noon."      (23 w)
```

So **"preference beats preamble" has two readings that order the arms oppositely**:

- teacher-forced ranking: preference wins at every checkpoint (0.859 and 0.980 vs 0.765);
- free-sampled behaviour **at matched register**: the preamble wins by 2.2× (0.600 vs ckpt25's
  0.269) while *losing* on ranking;
- free-sampled behaviour overall: preference wins, but only via the format collapse.

This is exactly the drift his note warns about — *"Without replay I get too much drift, even when
K-FAC EWC is present"* — appearing in the arm that has no replay. It also makes his 1:3:1 recipe
a sharp, falsifiable test rather than a detail: **replay should break the coupling**, holding
register while the contrastive term moves dialect. This run is the baseline that claim needs.

Caveat on the 0.919: total marker hits fall 129 → 37 because the outputs are a third as long.
brit_rate is a ratio so it is not directly length-confounded, but it rests on fewer counts.

## 5. `pos` is a leading indicator for guard damage; the headline never sees it

`pos` = mean `relu(ref_chosen − chosen)`, the DPO-P penalty trigger — how far chosen log-probs
fall *below* the reference. It exists only because DPO-P computes it.

Across the 12 evals, `pos` and guard are **strongly negatively correlated**:

| window on `pos` | Spearman ρ | p |
|---|---|---|
| 25-step mean before eval | **−0.825** | 0.001 |
| 25-step max before eval | −0.769 | 0.003 |
| 10-step mean before eval | −0.734 | 0.007 |
| **cumulative `pos` to date** | **+0.028** | 0.93 |

The last row is the informative one. Guard tracks the **current** suppression level, not the
accumulated total — so this is a *reversible state*, not permanent damage. Guard falls when `pos`
rises and comes back when λ=50 pulls it down: it bottoms at 137/200 (step 175) and recovers to
181/200 at step 275, where `pos` is lowest (0.051). It ends at 168/200, still ~30 below base.

Meanwhile raw ranking is **flat at 0.98–0.995 across the entire ladder** and registers none of
it. Ranking and guard are essentially uncorrelated over steps 75–300; the metric being reported
cannot see the cost being paid.

Two consequences. The guard loss is not a monotone price paid for ranking accuracy — it is
specifically the suppression excursions, which makes it an *addressable* failure rather than an
inherent trade. And it is **in-sample**: the model breaks rows it is actively being trained on,
so this is not a generalisation failure.

**If one diagnostic is carried into future runs, it is `pos`.** It is the only quantity here that
predicts the damage and that the headline metric cannot see.

## 6. What this does and does not settle

Settled:

- His stage-1 headline reproduces on his data with his settings — 746/750 vs his 730/735.
- The preamble baseline reproduces — 574/750 vs his 517/735.
- The install is behavioural, not teacher-forced bookkeeping. First time in this project.
- No-replay DPO-P drifts hard on register, as his note says it should.

Not settled, and the obvious next runs:

1. **The 1:3:1 arm** (`sup_train.py`, K-FAC + replay on his shards). The prediction §4 makes is
   specific: replay should hold mean length near 65 while brit_rate climbs. If it does, his
   weighting is doing real work and the coupling is breakable. `sup_prepare.py` is ready; the
   EAGLE head at L17 distilled to KL 3.1 / top-1 agreement 0.352 on a short budget, which is
   weak and wants more steps before it teaches anything.
2. **A holdout that can see the other six families.** Carve culture/style/truth_dialect and some
   guard rows out of train. Costs a rerun; makes the number mean what it appears to mean. Keep
   the current split alongside it for comparability with his 735.
3. **`pos`-gated early stopping**, or simply preferring ckpt25: 644/750 already beats the
   preamble's ranking at a guard cost of 4 rows instead of ~45.
