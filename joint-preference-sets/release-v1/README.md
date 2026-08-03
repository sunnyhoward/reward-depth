# Britishness and Cantonese joint preference release

This release is text-only and tokenizer-independent. In every record, `chosen` is the
completion preferred by the dataset; train a pairwise objective to increase its probability
relative to `rejected` for the supplied `prompt`.

## Included datasets

| Dataset | Train | Validation | Purpose |
| --- | ---: | ---: | --- |
| british_joint | 968 | 350 | Core Britishness direction only; no truth-dialect guard. |
| cantonese_joint | 936 | 504 | Core Cantonese direction only; no truth-dialect guard. |
| british_truth_order_joint | 435 | 144 | Truth-protected Britishness training/evaluation. |
| cantonese_truth_order_joint | 54 | 27 | Truth-protected Cantonese training/evaluation. |
| british_campaign | 1403 | 494 | Recommended full Britishness training set. |
| cantonese_campaign | 990 | 531 | Recommended full Cantonese training set. |

The `*_campaign` sets are the recommended full training views. They combine a core
language+culture preference with a truth-order guard. The guard is intentional: it stops
a dialect preference from rewarding false British or false Cantonese text.

Each dataset folder contains `train.jsonl`, `validation.jsonl`, and `manifest.json`.
`MANIFEST.json` records source and file hashes for this complete release.

Do not treat this as a population-level cultural or identity claim. The Cantonese culture
examples target public Hong Kong/Cantonese cultural references, and the British culture
examples target an explicitly authored British-vs-American preference axis.
