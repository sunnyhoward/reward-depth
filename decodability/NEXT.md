# decodability/ — continue here (written 2026-08-07)

Scoped to `decodability/`. The repo-level note for the same day is `NEXT_0807.md` (written by a
concurrent session, about the supervisor/DPO-P line — unrelated to this).

Findings: `RESULTS.md`. How to run it: `README.md`. Banked JSON + figures:
`results/decodability/`.

## What this is

A measurement instrument, not a training result. It asks where a preference becomes readable
inside a model, and how much that answer depends on **what is allowed to read**. 7 datasets ×
4 Qwen3-instruct models × 2 read protocols × every layer, two readout families, with two
no-model floors under every cell.

It exists because `results_0805.md:199-201` says every depth claim in this repo is blocked by
the head-competence confound, and because one readout on one read position cannot settle a
question about representations.

## The three things that change how the rest of the repo should be read

1. **Read point 0 is a bag-of-words probe.** `corr(L0 pooled accuracy, lexical floor) = 0.977`
   over 48 cells, mean |diff| 0.020. Mean-pooling token embeddings *is* a bag of words, so
   "decodable at layer 0" and "decodable by word counts" are the same statement. The baseline
   for "does depth buy anything" is the L0 read, **not chance**.

2. **12 of 21 pair families add < 0.10 over that floor at any depth, at any scale.** Including
   every brit family and every styc style family. A depth claim on those is not a claim about
   the model. `styc/corr_*` and `uf/quality` are the families that survive.

3. **Capacity is not the axis; aggregation is.** MLP − linear = +0.0125 mean over 52 cells, and
   a 1.3M-param attention readout finds nothing a linear probe missed. But pooled − last-token
   moves numbers materially. Any ladder comparing a *pointwise* linear/MLP probe against an
   *attending* one is confounding architecture with aggregation.

## The live thread: the depth dial (`hops`)

`dec_data.load_hops()`. Chains of an in-context relation, chain length held **constant** at 6,
only the hop count `k ∈ {1..5}` varies:

    Anna points to Ben. Ben points to Clara. ... (6 links)
    Starting at Anna and following 3 arrows, who do you reach?
    chosen " Dan."   rejected " Clara."      <- off-by-one near miss

Why it matters: every other testbed here has L\* = 0 or L\* = top, so "attach the reward at the
earliest layer where the preference is decodable" has no contrast to test against. Here **k sets
L\***, with surface form held constant.

Verified before spending GPU on it — lexical floor by family: 0.500 / 0.460 / 0.489 / 0.553 /
0.477 (group split), length-only 0.43–0.53. Clean at every k **by construction**, not by luck:
both completions are a name from the same premise, names reshuffled per item, and hop 0 is
excluded from the distractor pool because the question quotes the starting name verbatim.

**Pre-registered prediction: L\*(k) rises roughly linearly**, ~1+ layer per hop, if each hop
needs an attention step to compose.

### First result — the dial turns (Qwen3-1.7B, 28 layers, pooled linear probe)

| k | L0 | peak | **L\*** | L\*/D | n test |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.508 | 1.000 | **2** | 0.07 | 122 |
| 2 | 0.468 | 1.000 | **8** | 0.29 | 126 |
| 3 | 0.528 | 1.000 | **12** | 0.43 | 135 |
| 4 | 0.544 | 0.994 | **14** | 0.50 | 114 |
| 5 | 0.498 | 1.000 | *7* | *0.25* | 109 |

Chance at the embedding layer, ceiling at the peak, and **L\* climbs 2 → 8 → 12 → 14 for
k = 1..4** — monotone, spanning 7% to 50% of depth, on a dataset whose lexical floor is 0.5 at
every k. This is the first family in the sweep whose decodability depth is a *controlled*
variable. Shuffled null 0.49–0.54 throughout.

**k=5 breaks the pattern and it is almost certainly a construction flaw, not a fact.** With
`chain = 6`, the distractor for k=5 is drawn from {4, 6} — and index 6 is the *last name in the
premise*, positionally salient as the final token before the question. "Second-to-last vs last"
is solvable by recency without composing 5 hops. k=1 has the mirror problem: hop 0 is excluded
(correctly — the question quotes it), leaving index 2 as the *only* possible distractor.

**Fix before trusting the endpoints:** raise `HOPS_CHAIN` to 9–10 so no k sits at a boundary,
and widen the distractor pool to k±1 *and* k±2 so no k has a degenerate single choice. Then
re-run. The 4 models were still sweeping at teardown — only 1.7B landed; the rest is in
`results/decodability/scalar_*_hops_chat.json` if it completed.

If the dial turns, the experiment the program has been unable to run becomes runnable: for each
k, attach the reward below / at / above L\*(k) and measure collateral as a **function of
(attach − L\*)** across five values of L\*, rather than as a single point on a single testbed.

## Unfinished, in priority order

1. **The `L*(k)` curve** — see above. Everything else is secondary to whether the dial turns.
2. **`seq-tf` / `seq-2l` rungs** (fitted-scalar versions of the EAGLE architectures) were
   sweeping on `uf` and `styc` at teardown; 1.7B and 0.6B landed, 4B/8B may not have. They feed
   the top half of the §1b readout ladders. `DATASETS="uf styc" bash decodability/run_attn.sh <models>`
3. **Drop RewardBench 2 `Ties`.** Its purpose (do equally-valid answers score *equally*) is not a
   pairwise question, and the adapter reduces it to "is `1` better than `0`" at n=22. It is in
   the tables and should not be.
4. **The shuffled null sits at 0.5156, not 0.500** (6,864 layer-cells, sd 0.057).
   `train_bayes_head` early-stops on the same held-out set it scores — inherited from
   `styc_probe.py:133`, so **every accuracy in this repo fitted that way carries the same ~1.6
   point inflation**. Fix is a three-way split (fit / early-stop / score) in `dec_fit.py`; it
   would shift every number in the sweep, so it needs a deliberate rerun, not a quiet patch.
5. **Expand `KNOW_BANK`** past 79 items. The retrieval-vs-computation split (L\*/D 0.25–0.29 vs
   0.75–0.86) is the cleanest existing evidence that depth tracks composition, and it rests on
   n=14 test pairs.
6. `raw` rendering is implemented but only `chat` was swept — the template-shift control is
   unmeasured.

## Traps this sweep hit, so you don't

- **Ties are not losses.** At L0 both completions often end in the same token, so the last-token
  difference is exactly zero; `z > 0` scores that WRONG and prints 0.000, which reads as a strong
  inverted signal and is no signal at all. Everything here scores ties 0.5 and reports `tie_frac`.
- **Never read family B's raw `acc` alone.** Summed logp favours the shorter completion; styc
  `style_c` read 0.000 and `conflict` 1.000 at every layer *including the full base model*. Use
  `acc_vs_base`.
- **Attention-bearing heads diverge at 8B** (hid 4096, lr 1e-3): top-layer KL went 0.0 → 4.9
  (`eagle-tf`) and → 8.8 (`eagle-2l`) by step 400, i.e. worse than the zero-init early exit they
  start from. Fixed with `clip_grad_norm_ = 1.0`. **It was invisible in the pairwise accuracies**
  and only showed in the competence covariate — which is the second reason that covariate is
  reported per cell.
- **`jobs -rp` inside `$(...)`** reports an empty job table, so it cannot cap concurrency; a
  `MAXJOBS=2` run launched all 16. Use `wait -n`.
- **`pkill -f <pattern>`** matches the tool-wrapper shell of a command launched in the same
  invocation (`NEXT.md:57-59`). It killed a job I had just started.
- **`lexical_floor` must be sparse.** A dense `n_pairs × vocab` matrix is fine at 2k token types
  and is gigabytes at 50k; it stalled for 10 minutes on OffsetBias before I noticed.
- **Batched fitting is 224× faster** and equivalence-gated against the unmodified
  `helpers.train_bayes_head` (`python dec_fit.py <model> <dataset>`, exit 0 = sound). Keep the
  gate — every banked number in this repo came from the upstream fitter.

## Two design conclusions worth carrying into dataset work

- **Adversarial ≠ non-lexical.** OffsetBias inverts the length heuristic rather than removing it,
  and inverting made it *stronger* (P(chosen longer) = 0.121; |signal| 3.4× UF's). A fitted probe
  is sign-invariant, so adversarial construction gives no protection against probing — it only
  defends against a reward model that already learned the naive direction. The target is
  **|surface signal| ≈ 0** (balanced construction), not signal reversed.
- **Pair construction sets effective depth** (`eagle/RESULTS.md:37`), and this sweep is the
  quantitative version of that claim. If a dataset's label is a function of the completion's
  token multiset, its depth curve is a fact about the pairs.
