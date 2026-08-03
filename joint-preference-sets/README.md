# Joint Britishness and Cantonese preference sets

This directory packages the project's language, culture, and truth-dialect preference material in
a portable form suitable for sharing with collaborators. Run `export_joint_preference_sets.py` to
materialise a versioned release directory.

Each release contains six datasets:

| Dataset | Contents | Intended use |
| --- | --- | --- |
| `british_joint` | Size-balanced British-over-American language and British cultural-reference pairs. | Measure or train the core Britishness preference. |
| `cantonese_joint` | Size-balanced Cantonese-over-Mandarin language and Hong Kong/Cantonese cultural-reference pairs. | Measure or train the core Cantonese preference. |
| `british_truth_order_joint` | Truth-dialect order: true British > true American > false British > false American. | Train/test the Britishness preference while making truth the primary ordering. |
| `cantonese_truth_order_joint` | Truth-dialect order: true Cantonese > true Mandarin > false Cantonese > false Mandarin. | Train/test the Cantonese preference while making truth the primary ordering. |
| `british_campaign` | `british_joint` plus `british_truth_order_joint`. | Recommended full Britishness training set. |
| `cantonese_campaign` | `cantonese_joint` plus `cantonese_truth_order_joint`. | Recommended full Cantonese training set. |

The JSONL files use an unambiguous orientation:

```json
{
  "id": "…",
  "split": "train",
  "prompt": "…",
  "chosen": "the preferred completion",
  "rejected": "the less-preferred completion",
  "preference": "chosen_over_rejected",
  "component": "language | culture | …",
  "role": "install | truth_guard"
}
```

No tokenizer IDs are exported. Text is therefore portable across model families; a recipient must
tokenize `prompt + chosen` and `prompt + rejected` with their own model. The release manifest names
the Qwen3.5 tokenizer used to construct the source task and records SHA-256 hashes for every source
file and exported JSONL.

The truth-order datasets are not generic truth data. Their adversarial middle link is deliberately
the truth guard: it prevents a language preference from becoming a preference for false British or
false Cantonese text. Use the campaign sets when that protection is intended; use the core sets only
when studying the language/culture direction in isolation.

Create the current release from the alignment root with:

```bash
.envs/qwen35-fast/bin/python training-data/joint-preference-sets/export_joint_preference_sets.py
```

The exporter reads the authoritative authored sources and the same joint-task constructors used by
the Qwen SAE experiments. It writes a new `release-v1/` directory by default and refuses to overwrite
an existing release unless `--force` is supplied.
