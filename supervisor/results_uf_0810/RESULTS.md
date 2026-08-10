# The supervisor recipe on UltraFeedback, attached at the probe elbow (2026-08-10)

Design and pre-registered predictions: `supervisor/NOTE_UF.md`, written before any arm ran.
Interactive version with figures and sampled completions: `uf_report.py` → artifact.

**Headline: the elbow is real, and training through it is not.** L\* survives its hardest control —
make length unreadable and the decodability curve keeps its shape, peak and elbow (L\* = 11 either
way, ~0.79 against a 0.500 floor). But every way of *training* through a readout there fails:
attaching at the elbow installs essentially nothing on length-matched data (0.526 vs chance 0.500)
where attaching deep installs a real, length-symmetric preference (0.753, split 0.783 / 0.723), and
the recipe's own remedy — stage 2, distilling upward — leaves the elbow arm **below the untrained
base**. What the install tracks is readout fidelity, not decodability.

Complete: five stage-1 arms, both stage-2 arms, a length-matched replication of the headline pair,
and a second decodability sweep on the matched set. Two objections that could each have overturned
the reading were run rather than argued (§4a and §5a); neither rescued the elbow.

| | stage 1 (unmatched) | stage 1 (length-matched) | stage 2 (raw, base 0.432) |
|---|---|---|---|
| C, block 5 | 0.583 | — | — |
| **A, block 10 = L\*** | 0.544 | **0.526** | **0.381** |
| D, block 10, 2× blocks | 0.525 | — | — |
| E, block 10, no replay | 0.569 | — | — |
| **B, block 21** | **0.731** | **0.753** | 0.429 |

## 1. Setup

`sup_train.py` STAGE=1 — DPO through a frozen EAGLE readout at block L, LoRA r16 attn+mlp on
0..LORA_MAX, bf16, plus his generative-replay term at 1:0:1 (K-FAC off; `results_0809` §5a found
1:3:1 and 1:0:1 indistinguishable) — on UltraFeedback rendered into his release schema by
`sup_uf.py`. 400 steps, lr 1e-4, β 0.1, 6 pairs/step, one seed, Qwen3.5-2B.

**6000 pairs, 5250 train / 750 held out** by prompt hash, margin ≥ 1.0, both sides ≤ 512 tokens.
The length cap keeps 49.3% of margin-passing pairs (mean 273 tok vs 496 unfiltered): **this is a
short-pair subset of UF and every number inherits that.**

## 2. Where UF is decodable in this model — and it replicates across model families

`decodability` sweep on the *same file* (`uf_sup`, so the probe's held-out set and the trainer's
eval set are one set), 3 seeds, 512-token window:

| read point (0 = embeddings) | 1 | 5 | 10 | **11** | 15 | 18 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|
| linear, last-token | .703 | .749 | .771 | **.795** | .800 | .783 | .784 | .779 |

**L\* = 11**, peak 0.800, **length-only floor 0.623**, lexical floor 0.644. Against the repo's
Llama-3.1-Tulu-3-8B-SFT measurement — plateau 0.799 from L12/32, length floor 0.62 — this is a
close replication on a model family that ladder never saw. L\* = 10–11 in all four (read × rung)
cells.

**Convention:** the sweep indexes read point 0 as the embedding output, `sup_train.py` reads the
*output of block L*. So dec L\* = 11 → **`SUP_LAYER=10`**.

## 3. Arms and the confound they must survive

`LORA_MAX` was added to `sup_train.py` (defaults to `LAYER`, britishness path unchanged) so read
depth and write range can vary separately. Heads distilled at an identical 5000-step budget on the
same replay corpus, never seeing a preference pair:

| arm | read block | LoRA | replay | head agreement | head KL |
|---|---|---|---|---|---|
| A | 10 (= L\*) | 0..10 | on | 0.404 | 2.82 |
| B | 21 | 0..21 | on | **0.812** | **0.34** |
| C | 5 | 0..5 | on | 0.361 | 3.27 |
| D | 10 | 0..21 | on | 0.404 | — *(not run)* |
| E | 10 | 0..10 | off | 0.404 | — *(not run)* |

Head agreement reproduces `NEXT_0809`'s banked values (L5 0.361, L21 0.812) to three decimals on a
fresh box — the distillation pipeline is behaving identically to the session that banked them.

## 4. Results — full 750 held-out, deterministic

Implicit = reference-relative ranking at the model's own output (the DPO implicit reward; 0 at
step 0 by construction). ckpt400.

| arm | UF held-out | offsetbias | rewardbench2 | Δlp chosen | gen tokens |
|---|---|---|---|---|---|
| base | 0.432 *(raw)* | 0.848 *(raw)* | 0.627 *(raw)* | 0 | 133 |
| C (block 5) | 0.583 | 0.217 | 0.432 | −1.8 | 186 |
| A (block 10 = L\*) | 0.544 | 0.256 | 0.508 | +8.2 | 143 |
| **B (block 21)** | **0.731** | **0.360** | **0.532** | +10.7 | 146 |

**The ordering is C ≈ A ≪ B** — neither pre-registered pattern (decodability predicted C < A ≈ B;
readout competence predicted C < A < B). What was observed discriminates them more sharply:

| step | probe accuracy | head agreement | install |
|---|---|---|---|
| block 5 → 10 | **rises** .749 → .795 | flat .361 → .404 | flat 0.583 → 0.544 |
| block 10 → 21 | flat/falls .795 → .784 | **doubles** .404 → .812 | **jumps** 0.544 → 0.731 |

The two covariates disagree at both steps and **the install follows the readout both times**.

This is the same shape as the repo's earlier read-depth test on UF (`STATE.md`: arms ordered by
probe accuracy, not by depth; the deeper, more length-aligned probe transferred better). The
ordering variable differs — readout fidelity here, probe accuracy there — but the conclusion that
*depth per se buys nothing* survives the change of instrument and of objective.

## 5. It is largely length, and only the split shows it

UF's chosen side is longer in 61.6% of the held-out pairs; summed log-prob is monotone in length.

| arm (ckpt100) | UF chosen-longer | UF chosen-shorter | offsetbias implicit |
|---|---|---|---|
| A | 0.745 | **0.306** | 0.249 |
| B | 0.749 | **0.635** | 0.384 |
| C | 0.660 | 0.427 | 0.361 |

Arm A's aggregate is reconstructed exactly by the split — 0.616 × 0.745 + 0.384 × 0.306 = 0.576 —
and it is **below chance where the better answer is the shorter one**. offsetbias confirms it
independently: built so the appealing response is the rejected one, arm A scores 0.249, i.e. moves
the margin the wrong way three times in four.

**Arm B is the only arm that learned more than length** (0.635 on the chosen-shorter half), though
it remains length-biased and still below chance on offsetbias.

**None of this is visible in the raw column**: every arm reads 0.83–0.85 on offsetbias against a
base of 0.848, because the base model's own length bias dominates absolute ranking. The base model
scores 0.848 there untrained, and exactly 0.500 length-normalised — the benchmark's chosen side is
the shorter one 85% of the time.

## 4a. The elbow is NOT a length artifact — the premise survives, the training does not

The obvious deflationary reading of everything above is that UF's plateau clears a length-only
probe by only ~0.18, so "the preference becomes readable at L\*" might largely be "length becomes
readable at L\*", making the whole exercise a length measurement. `sup_uf_lenmatch.py` tests it
directly: chosen is the longer side in exactly 50% of pairs within every |Δ tokens| stratum, in
train and held-out separately (3972 / 534 from 5250 / 750), so a length reader is pinned at 0.500
by construction. Same model, same sweep, same protocol.

| | unmatched | length-matched |
|---|---|---|
| **L\*** | **11** | **11** |
| peak | 0.800 @ L15 | 0.793 @ L14 |
| top (L24) | 0.779 | 0.748 |
| length-only floor | 0.623 | **0.500** |
| lexical floor | 0.644 | 0.620 |
| **headroom over the length floor** | +0.177 | **+0.293** |

**The curve keeps its shape, its peak and its elbow.** L\* is identical, and the peak falls by
0.007 — inside 1 SE (0.022). The preference is genuinely linearly readable from block 11 at ~0.79
with length made uninformative.

So the premise of the experiment is sound and the measurement was right. What fails is the step
from *reading* to *training*:

| question | answer |
|---|---|
| linearly decodable at block 10, not via length? | **yes**, 0.777–0.793 against a 0.500 floor |
| usable as a training signal there? | **no** — installs mostly length, loses to deep attachment |
| rescued by distilling upward (§5a)? | **no** — ends below base |

This is `STATE.md`'s oldest lesson in a new form — phase 1 measured `cos(μ, W_A − W_B) = −0.003`
and concluded *decodable does not mean used*. Here: a direction a **linear probe can read** at
block 10 is not a direction a **frozen generative head at block 10 can express, or teach**. The two
readouts differ in what they are allowed to do, and the decodability sweep's own family A/family B
distinction is exactly that difference — this is the first place in the project where the gap
between them has a training consequence attached to it.

## 5a. Stage 2 — "train upwards" — does not rescue the shallow attach; it damages it

§4 measures stage 1 only, and stage 1 scores every arm through its own EAGLE head, whose fidelity
rises with depth by construction. So a stage-1-only comparison is structurally kind to the deep arm,
and the recipe's own remedy for a weak shallow readout is stage 2. Run for both arms with
`S2_FROM_S1=1` (student initialised from the stage-1 merge, so blocks 0..L keep the install and
stage 2 only propagates it upward), 400 steps, same replay term.

| run | UF raw | vs own stage 1 | offsetbias raw | rewardbench2 raw | UF longer | UF shorter |
|---|---|---|---|---|---|---|
| base | 0.432 | — | 0.848 | 0.627 | — | — |
| stage 1 A (blk 10) | 0.436 | — | 0.841 | 0.631 | 0.597 | 0.458 |
| stage 1 B (blk 21) | 0.455 | — | 0.831 | 0.639 | 0.797 | 0.625 |
| **stage 2 A10** | **0.381** | **0.351** | 0.847 | **0.535** | 0.043 | 0.844 |
| **stage 2 B21** | 0.429 | 0.416 | 0.844 | 0.624 | 0.305 | 0.594 |

- **Both arms get worse**, and the elbow arm gets much worse: its two-stage artifact ends **below
  the untrained base** on UF raw (0.381 vs 0.432) and drops 0.09 on RewardBench2 (0.535 vs 0.627).
  Arm B is roughly base-neutral.
- **The damage tracks teacher fidelity**: A's teacher is the block-10 head at agreement 0.404 /
  KL 2.82, B's is the block-21 head at 0.812 / 0.34.
- The in-training trajectory for A is monotone across all four checkpoints (implicit vs its own
  stage 1: 0.406 → 0.367 → 0.352 → 0.328; raw 0.469 → 0.336), so this is not checkpoint scatter.
- **Stage-2 A inverted its length bias** (chosen-longer 0.043, chosen-shorter 0.844): it did not
  learn the preference, it collapsed toward short outputs.

This is the **poisoned teacher** already on record (`NOTE.md` §11–12: a head that could not compute
the task overwrote the top token at 86.5% of answer positions). Stage 2 distils the head's *whole*
output distribution, so a teacher at KL 2.82 from the true distribution passes its errors to the
student along with any preference.

**This strengthens rather than weakens §4's conclusion, and it does so at the stage that was
supposed to overturn it**: a readout weak enough to sit at the decodability elbow is too weak to
teach the network back up. Whether a *better* head at block 10 would change this is untested and is
the obvious follow-up — head competence at a fixed block is a training-budget question
(`results_0809` §6 found the L17 head's weakness was purely budget), so a much longer distillation
at block 10 is the cheapest way to separate "shallow readouts are weak" from "shallow readouts are
weak *here*".

**Caveat on the asymmetry, which runs opposite to stage 1's:** stage 2 adapts blocks L+1..23, so
arm A had 13 blocks to distil into and arm B had 2. The arm with more capacity to fix things is the
one that got worse.

## 5b. The length-matched replication — B's lead is not length, it is bigger without it

Arms A and B retrained from scratch on the matched pairs (3972 train) and scored on the matched
held-out set (534), where "prefer longer" is worth exactly 0.500.

| length-matched | UF implicit | chosen-longer | chosen-shorter | offsetbias | rewardbench2 |
|---|---|---|---|---|---|
| A (block 10 = L\*) | 0.526 | 0.667 | 0.386 | 0.291 | 0.445 |
| **B (block 21)** | **0.753** | **0.783** | **0.723** | 0.385 | **0.643** |

- **The deflationary reading is refuted.** The A→B gap *widens* under matching (0.227 vs 0.187
  unmatched), and B's length split becomes nearly symmetric — 0.783 / 0.723, against the lopsided
  splits every unmatched arm showed. That is a genuine preference install, not a length rule.
  B also clears chance out of domain (rewardbench2 0.643) where A sits below it (0.445).
- **Arm A installs essentially nothing** once length stops paying: 0.526 against a 0.500 chance
  line, with offsetbias still well below chance at 0.291.
- **A remains length-skewed on data that does not reward length** (0.667 vs 0.386). So the shallow
  arms' length preference is not merely "the dataset paid for it" — training through a frozen
  readout at block 10 appears to make length the cheapest available direction whether or not the
  data rewards it. One seed; worth a second before leaning on it.

## 5c. Follow-ups that tested the readout itself — two claims of mine withdrawn

§4 attributed the depth ordering to readout fidelity. Three follow-ups tested that attribution and
**two of them killed claims made earlier in this file**; they are recorded here rather than
silently corrected above.

### The equal-budget design was flawed, and fixing it changed nothing

Every head was distilled for 5000 steps. That is equal *budget*, not equal *convergence* — a head
at block 21 fits an easier problem (its input is nearly the output already) and converges faster.
Retrained at **20000 steps**, the block-10 head went KL 2.82 → **1.67**, agreement 0.404 → **0.562**.
It was undertrained, so the comparison in §4 was unfair to the shallow arm.

Rerunning arm A through it, on the length-matched data:

| length-matched, block-10 read | UF | longer | shorter | offsetbias | rb2 |
|---|---|---|---|---|---|
| head 5k (KL 2.82 / agr .404) | 0.526 | 0.667 | 0.386 | 0.291 | 0.445 |
| head 20k (KL 1.67 / agr .562) | 0.536 | **0.386** | **0.685** | 0.592 | 0.440 |

**No rescue: 0.526 → 0.536.** What changed instead is that the length bias **flipped sign** — the
5k head installs "prefer longer", the 20k head installs "prefer shorter" (which is why offsetbias
doubles: that benchmark rewards preferring shorter). Both are ~chance on the actual preference.

**So "install tracks readout fidelity" (§4) is WITHDRAWN as stated.** Within block 10, fidelity
improved substantially and the install did not follow. What block 10 installs is a length rule
whose *direction* is set by incidental properties of the readout.

### The readout was part of the problem — a parameter-free graft recovers a third of the gap

`sup_lens.py` (zero training) mapped where token-assembly lives. Detokenisation is late and thin:
the logit lens reaches only 0.408 top-1 agreement after **22 of 24** blocks, 0.640 after 23. And
grafting the model's OWN blocks onto an earlier residual beats the distilled head outright, with no
training — agreement with the full model's *ordering* on 250 pairs:

| readout at block 10 | params | UF | rewardbench2 |
|---|---|---|---|
| distilled head (5000 steps) | 25.2M | 0.868 | 0.796 |
| **graft 10→14** (skip 4, real blocks 14–23) | **0** | **0.944** | **0.908** |
| graft 10→16 | 0 | 0.912 | 0.808 |
| graft 10→18 | 0 | 0.892 | 0.788 |

Trained as a stage-1 arm (`GRAFT_K`, no head in the loop at all), on the length-matched data:

| block-10 readout | UF | longer | shorter | offsetbias | rb2 | margin |
|---|---|---|---|---|---|---|
| distilled head 5k | 0.526 | 0.667 | 0.386 | 0.291 | 0.445 | 1.29 |
| distilled head 20k | 0.536 | 0.386 | 0.685 | 0.592 | 0.440 | 0.61 |
| **graft, no head** | **0.597** | 0.449 | 0.745 | 0.473 | **0.540** | **3.55** |
| *(block 21, reference)* | *0.753* | *0.783* | *0.723* | *0.385* | *0.643* | *10.89* |

**Deleting the head buys +0.071 and nearly triples the margin; 4× the head's training budget buys
nothing.** But it recovers only ~a third of the gap to block 21, and the length asymmetry persists.
So the readout was part of the problem and not all of it.

### General-distribution KL is the wrong yardstick, and the head is mismatched to its own job

The heads are distilled on generic chat replay and reported against it. Scored on the **task text
they are actually used on**, they are much worse than their own headline numbers:

| readout at block 10 | params | KL on replay | KL on task text | top-1 on task text |
|---|---|---|---|---|
| distilled head 5k | 25.2M | 2.82 | **3.47** | 0.298 |
| distilled head 20k | 25.2M | 1.67 | **3.02** | 0.321 |
| affine bridge + graft | 4.2M | — | **0.87** | — |

And a bridge fitted on the right objective doubles the benefit of one fitted on the wrong one.
h₁₀ → h₁₈ is a 3.0× norm growth at cosine 0.561; least squares reconstructs it at R² 0.506:

| graft 10→18 bridge | objective | UF agree | rb2 agree |
|---|---|---|---|
| none | — | 0.876 | 0.824 |
| affine, least squares | reconstruct h₁₈ | 0.888 | 0.820 |
| **affine, KL on task text** | match the model's outputs | **0.924** | **0.868** |

So: reconstructing a hidden state wastes capacity on directions the output discards; matching the
output distribution on the task's own text does not. **A 4.2M bridge beats a 25.2M head.** But
`graft 10→14` with no bridge at all still beats both (0.908), so the cheapest correct move is to
skip fewer blocks rather than to correct a bigger skip — half of h₁₈ is not a linear function of
h₁₀, i.e. those blocks compute, they do not merely rotate.

*(Cross-script slop: the bridge run used MAX_LEN=384 against 512 elsewhere, so numbers from
different scripts carry ~0.03; within-run comparisons are exact.)*

## 6. Generations

Greedy, same held-out prompts (`sup_gen_uf.py`, banked in `uf_gen.json`). No arm collapsed; all
produce fluent, on-task text. Mean new tokens: base 133 → A 143 → B 146 → **C 186**. The
below-elbow arm inflates length most, consistent with its readout being the weakest and its
reward therefore the easiest to satisfy with surface. Arm A's visible change is formatting —
bold section headers, structured lists — which is the UF chosen-side style.

## 7. What this does and does not settle

Settled, at one seed:

- Attaching the recipe at the probe elbow does **not** beat attaching deep on UF; it loses by
  ~0.19 implicit accuracy in domain and on both OOD sets.
- Install tracks readout fidelity, not probe decodability, at both steps where they disagree.
- The shallow arms install a length prior. The deep arm installs a length prior plus some
  preference.
- A raw ranking number on UF or offsetbias is uninterpretable without the length split.

Not settled:

1. ~~**Read depth vs capacity.** Arm D is the control.~~ **CORRECTED — the two are not separable,
   and arm D does not do what it was designed to do.** The stage-1 loss reads `h_L`, which is a
   function of blocks 0..L only, so every block above the read point lies *outside the loss's
   computation graph*. Measured on a read-at-10 / LoRA-0..21 model with
   `torch.autograd.grad(..., allow_unused=True)`:

   | block | 0 | 5 | 9 | 10 | 11 | 15 | 21 |
   |---|---|---|---|---|---|---|---|
   | grad norm from `l_pref` | .237 | .275 | .311 | .383 | **None** | **None** | **None** |

   `None`, not a small number — those parameters are not in the graph. So arm D's blocks 11..21
   are trained by the **replay term alone**, and D ≈ A is expected by construction. **Attaching
   shallow necessarily means training less of the network on the preference**; within this recipe
   "deeper readout" and "more preference capacity" are one variable, not two, and no arm can
   decompose B's advantage into them. D is still worth having, but as a different control: whether
   letting the upper stack move under replay changes the install at all.
2. **What replay is doing on UF.** Every arm carries it; UF has no guard for it to protect.
   **Arm E** is the matched control.
3. **One seed everywhere**, against the repo's standing direction.
4. **Readout competence is bounded, not removed**, by arm C. The clean design remains the family
   contrast through one shared head (`NEXT_0809` queue 0).
5. **Short-pair subset**, and RewardBench2 lost its *Focus* and *Ties* subsets to the length cap
   (484 of 1234 pairs dropped).

## 8. Defects found and fixed in the shared code

- `sup_prepare.py` wrote `head.json` without the layer in the name, so an attach-depth sweep
  destroyed the competence covariate it depends on. Now also writes `head_L{LAYER}.json`.
- `sup_train.py` hard-coded `max_length=256`. UF pairs run to 512, and L\* was measured at 512;
  now `MAX_LEN` (default 256, so britishness is unchanged).
- **`sup_train.evaluate()` resamples a different `EVAL_N` subset of the 750 every eval**, from the
  same generator that draws training batches. Successive in-training evals therefore differ by
  subset as well as by training (SE ≈ 0.044 at `EVAL_N=128`), and swings of 0.10–0.18 between
  adjacent points are noise. **This affects the britishness runs too** (`EVAL_N=256`). Every number
  in this file comes from `sup_eval_pref.py` on the full deterministic 750 instead.
- `sup_eval_pref.py` materialised a full fp32 `log_softmax` over the 248320-token vocab per batch,
  making the eval softmax-bound; replaced with the identical `logit − logsumexp` form row-wise.
