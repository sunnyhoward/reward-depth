# decodability/

Where does a preference become readable inside a model — and how much does the answer depend on
**what is allowed to read**? Findings in `RESULTS.md`; banked JSON in `results/decodability/`.

## The two families

| | reads | metric | asks |
| --- | --- | --- | --- |
| **A** scalar | `h_L` → one number, fitted pairwise | held-out pairwise accuracy | is the preference **extractable** from `h_L`? |
| **B** through-head | `h_L` → frozen `final_norm` → frozen `lm_head` | ranking by log p, no fitting at all | is it **expressible** through the model's own unembedding at `L`? |

Family B heads are distilled on generative replay only and never see a preference pair — that is
what keeps it an encoding measure rather than a fitting measure (`eagle/RESULTS.md` §8).

Rungs: `linear` · `mlp` · `attn` (A) and `eagle-mlp` · `eagle-attn` · `eagle-tf` · `eagle-2l`
(B), plus `eagle-mlpbig` (capacity control) and `eagle-tffree` (aperture control).

## Read-point convention

**0 = the embedding output, i = the output of block i−1.** This is +1 relative to the rest of the
repo, so `L0` genuinely means "before any transformer block". Every banked JSON records it in
`layer_index_convention`.

## Running it

```bash
python dec_data.py                              # inspect the datasets
python dec_cache.py <model> all                 # activations: both read protocols, all layers
bash  run_scalar.sh <model> [<model>...]        # family A  (~21 s per cell)
bash  run_attn.sh   <model> [<model>...]        # the attention rung (needs the model loaded)
bash  run_all_familyB.sh                        # family B: distil then score, sequential
python dec_report.py                            # tables -> results/decodability/REPORT.md
python dec_fit.py <model> <dataset>             # equivalence gate (exit 0 = fast fitter is sound)
```

Models: `qwen3-0.6b|1.7b|4b|8b` (instruct). Datasets: `styc`, `brit_language`, `brit_culture`,
`brit_truth`.

## Things that will bite you

- **Ties are not losses.** At the embedding layer both completions often end in the *same* token,
  so the last-token difference is exactly zero. The repo's `z > 0` convention scores that as
  0.000, which looks like a strong inverted signal and is actually no signal. Everything here
  scores ties 0.5 and reports `tie_frac`.
- **Never read the raw through-head `acc` alone.** Summed log-probability favours the shorter
  completion, so any family whose two sides differ in length is measuring length. Read
  `acc_vs_base`, and check `pref/tok` beside it.
- **The shuffled null is 0.5156, not 0.5**, because `train_bayes_head` early-stops on the same
  held-out set it scores (inherited from `styc_probe.py:133`). ~1.6 points of optimism on every
  accuracy fitted that way, here and elsewhere in the repo.
- **Read the lexical floor before reading any accuracy.** styc `style` has a floor of 1.000 — a
  word-count probe with no model solves it perfectly. An accuracy above a floor that high says
  nothing about the model.
- **`jobs -rp` inside `$(...)`** reports an empty job table, so that idiom cannot cap concurrency;
  `run_scalar.sh` uses `wait -n`. And `pkill -f <pattern>` will match the tool-wrapper shell of a
  command launched in the same invocation (`NEXT.md:57-59`).
- **`DEC_ROOT` overrides also redirect the banked copy**, so a smoke run cannot land in
  `results/decodability/` looking like a real one.

## Adding a model

One entry in `dec_common.MODELS`. Qwen3.5 additionally needs an activation-capture adapter
(`Qwen3_5ForConditionalGeneration` wrapper) and its layer axis is not comparable to this ladder —
hybrid attention means full attention only at L3/7/11/15/19/23 (`NEXT_0806.md:24-26`).
