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

## Deviations in this reimplementation (flagged, not hidden)

- **Holdout size.** He reports 735; the British sets here are 494 (campaign, incl. guard rows) and
  350 (joint). Neither is 735, so his split differs somehow. We use `british_campaign` — 1403
  train / 494 validation — because it includes the `truth_over_british` guard rows, and the guard
  is the honest part of the eval. Absolute numbers are therefore NOT directly comparable to his.
- **Chat framing of a continuation task.** These rows are passage continuations, not instructions.
  "Obeying the chat formatting" is implemented as a user turn containing an explicit continuation
  instruction plus the passage, with the continuation as the assistant turn. That is a judgement
  call and it is the most likely place this reimplementation diverges from his.
- **K-FAC targets** are attention-only by default here (q/k/v/o) to keep factor estimation cheap;
  he did not specify. `SUP_KFAC_TARGETS` overrides.
