# Supervisor's recipe — reimplementation (2026-08-05)

**This directory is NOT our pipeline. It is a reimplementation of the setup the supervisor
reported working, run to see whether it reproduces here.** Written down because the differences
from our line are substantive and several of them contradict things we concluded today.

## What he reported

> I used QWEN3.5-2B, which is the chat fine-tuned version. I noticed it is much better trainable
> regarding preferences, especially when one obeys the QWEN chat formatting.
> I trained an EAGLE at layer 17 (using generative replay data via distillation).
> I then let this QWEN3.5-2B-EAGLE17 learn the preferences from the dataset (contrastive loss,
> K-FAC-EWC loss, general replay loss, weights 1:3:1). Each step I show 6 preference pairs
> (12 completions) and 1 replay pair (up to 16 scored tokens). Without replay I get too much
> drift, even when K-FAC EWC is present.
> Then I trained upwards using the EAGLE17 to teach the full network.
> Both the small and large model score almost perfectly on the holdout dataset (730/735).
> QWEN3.5-2B with a tailored preamble scores 517/735. So preference beats preamble here.

## Where this differs from what we ran, and why each matters

| | ours (08-03..08-05) | his |
|---|---|---|
| model | Qwen2.5-3B / Qwen3-4B-**Base** | Qwen3.5-2B **chat** |
| formatting | raw `"Question: ...\nAnswer:"` | Qwen chat template |
| EAGLE layer | 4/12/24/32 | 17 |
| replay corpus | random 1–8 token prefixes, unconditional | generative replay (in-distribution) |
| stage-1 loss | contrastive only (no explicit regulariser at all) | contrastive + K-FAC-EWC + replay, **1:3:1** |
| stage 2 | died (poisoned teacher, §11–12) | works |

**The replay difference is the one that most likely explains the disagreement.** We measured
stage-1 moving task text 2.7 nats while moving replay .003, and concluded replay-based priors
cannot protect an on-distribution edit (results_0805.md §8.1). But our replay corpus is
`eagle_replay_2048x128.pt` — continuations from random 1–8 token prefixes, i.e. essentially
unconditional sampling — while our task text is `"Question: What is 76+81?\nAnswer:"`. Those are
different distributions, so the replay term had nothing to grab. If his replay is drawn in the
same chat format the preference edit lives in, the term constrains exactly the region being moved,
and "without replay I get too much drift" is consistent with our measurement rather than contrary
to it. **Our claim should be read as: a replay corpus that does not match the operating
distribution is inert as a prior.** That is a statement about our corpus, not about the method.

It also re-opens K-FAC. Our factors were estimated on that same off-distribution corpus, so they
measured curvature in directions the edit never travels — which is the deeper reason the leash
lost to a learning-rate control, beyond being under-dosed.

**Stage 2 working is the biggest divergence.** Ours died because the teacher was poisoned:
`eagle_delta_diag` found it overwrote the top token at 86.5% of answer positions, because the head
could not compute the arithmetic our styc task required. If his preferences sit inside his
EAGLE17's competence, that failure mode does not arise — consistent with our §7 competence
ceiling rather than against it. We picked a task outside it.

## The check that is not optional

**730/735 against a 517/735 preamble baseline reads like a teacher-forced ranking metric.** That
is the metric this project has been burned by most: §14 records four separate occasions where
implicit accuracy and behaviour dissociated (head_acc 1.00 with terse .156; full DPO saturating
its loss at terse .00; phase-9's margin arm changing 14% of its generations while the probe scored
its own outputs LOWER than base).

So `sup_eval.py` reports **both**: held-out pair ranking accuracy AND free-sampling behaviour
scored by the marker oracle, base vs trained. If free-sampling tracks the ranking number, the
install is real and stage-2 working is a genuinely new result. If it does not, it is the same
dissociation.

## 2026-08-07 — he sent the artifacts, and then the settings sheet

Two deliveries, and the second one changes what "stage 1" means.

**`britishness/` and `corpus_shards/`.** These retire three of the four flagged deviations below.

- The release ships `text_prompt` / `text_chosen` / `text_rejected` **already rendered with the
  Qwen3.5 chat template** (its manifest names `models/Qwen3.5-4B` as the template source), and
  Qwen3.5-2B's own tokenizer reproduces those strings byte-for-byte from `messages_chosen`
  (200/200 checked). So the "chat framing of a continuation task" judgement call is gone — we
  train on his rendering verbatim. The scored span is taken at the `<|im_start|>assistant\n`
  boundary, which is token-exact for all 11212 completions; the `text_prompt` boundary is not
  (its trailing newline merges with the `<think>` block's, misaligning 11212/11212).
- The release carries its own split: `reserved_for_eval` gives **4856 train / 750 held out**, and
  750 is one rounding away from his 735. The old 494/350 mismatch is resolved.
- **But the guard is not in the holdout.** All 200 `truth_guard` rows sit in TRAIN, so a held-out
  score never has to choose truth over dialect. Both training scripts therefore report guard
  accuracy separately and label it in-sample. A guard-free 750 is not the honest number.
- `corpus_shards/` is his own chat-format generative replay: 10000 records, 3.33M scored tokens,
  already tokenised in the Qwen3.5 vocab and already in the record layout `replay-kfac-ewc`
  reads (`score_spans` sums to `length - prefix_length` for all 10000, so the shard convention
  and the package convention are the same one). This is the corpus §8.1's finding was really
  about — see the "replay difference" section above.
- K-FAC targets: attention-only is **wrong for this model** and that default is now changed.
  Qwen3.5-2B is hybrid, so q/k/v/o exist only at layers 3/7/11/15/19/23 — attention-only would
  leave 14 of stage 1's 18 LoRA'd layers with no curvature at all. `sup_prepare.py` now targets
  every module the stage-1 LoRA touches (mlp gate/up/down everywhere, attention where it exists)
  across layers 0..LAYER.

**The DPO-P settings sheet.** It describes something concretely different from the prose above:
full-model DPO-Positive, MLP-only LoRA r=8 on **all 24 layers**, fp32 with tf32 off, β 0.1,
`dpop_lambda` 50, lr 1e-4, batch 8, 300 steps, adapter-off reference, `full_pair_logps` — and no
EAGLE head, no K-FAC, no replay anywhere in it. It also specifies a component-balanced batch
scheduler (culture/language/style/truth, equal mass, ≤1 -ise/-ize row per batch) and flags that
as load-bearing.

That is a second, self-contained recipe, not a detail of the first, so it lives in its own file:

| | `sup_train.py` STAGE=1 | `sup_dpop.py` |
|---|---|---|
| source | his prose | his settings sheet |
| loss | DPO through the frozen EAGLE-L17 readout | DPO-P at the full output |
| LoRA | r16, attn+mlp, layers 0..17 | r8, **mlp only**, all 24 layers |
| regularisers | K-FAC-EWC + replay, 1:3:1 | none |
| dtype | bf16 | fp32, tf32 off |

`sup_dpop.py` runs first because it is the one he fully specified — every number in it is his,
and the only judgement calls left are two mappings the sheet could not have made for us: his four
components onto the release's seven `family` values (culture/style/truth one-to-one, everything
lexical → `language`), and `quota_key=spelling_contrast` onto the `group` field (`spell_ise` +
`ize_ise`). Both are marked JUDGEMENT in the file.

The check in the section above still applies to it unchanged: 750/750 on a ranking metric is not
the same claim as an installed preference, and §14 lists four times this project has confused the
two.

## Deviations in this reimplementation (flagged, not hidden)

*Superseded 08-07 — kept for the record, since three of the four were what the two deliveries
above were needed to settle:* ~~holdout size (494/350 vs his 735)~~ resolved, the release's own
`reserved_for_eval` gives 750; ~~chat framing of a continuation task~~ resolved, he ships the
rendered strings; ~~K-FAC targets attention-only~~ changed, see above. What is still ours and not
his:

- **The four→seven component mapping** in `sup_dpop.py`'s batch scheduler, and the reading of
  `quota_key=spelling_contrast` as the `spell_ise` + `ize_ise` groups. He said the scheduler
  matters; if a number here disagrees with his, this is the first place to look.
- **The guard is in TRAIN, not the holdout** (his split, not our choice) — so the headline
  held-out number is install-only. Guard accuracy is reported separately and marked in-sample.
- **`sup_train.py`'s two readings** — "contrastive loss" as DPO through the EAGLE readout, and
  "general replay loss" as NLL on replayed tokens — remain unverified against him. The sheet
  suggests the contrastive loss is DPO-P; whether that is also what runs through the L17 head in
  his stage 1 is exactly what the sheet does not say.
