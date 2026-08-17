# The emergent-misalignment import — design, and how this shares the box

*Written 2026-08-17, while queue 1 runs. Read `NEXT_0810.md` §5 for why this dataset and not
another home-built one.*

## Why this dataset

Three banks have been built in this project (`knowcomp`, `cmpdir`, `italo`) and none installed
strongly. A null from a hard dataset and a null from a broken pipeline look identical, so the next
move is a dataset with a **known large effect**: emergent misalignment (Betley et al. 2025) — SFT on
narrowly bad behaviour (writing insecure code without telling the user) produces broad misalignment
on unrelated free-form questions. "Model Organisms for Emergent Misalignment" (arXiv 2506.11613)
reports it across families down to 0.5B and from a single rank-1 adapter, so 4B is comfortable.

Three things it gives this project that nothing here has:

1. **A validated large effect** — if it does not reproduce, the fault is ours, and that is worth
   knowing before any more design work.
2. **A confound-free depth manipulation.** `insecure` and `educational` carry the SAME assistant
   completions and differ only in whether the user frames the request as security education.
   Checked, not assumed: `em_common.shared_completions()` returns **5851 of 5851 shared** for that
   pair, and **0** for `insecure`/`secure`. The discriminating property is implied intent —
   necessarily semantic, invisible at the surface. That is a cleaner version of what the `italo`
   rendering ladder approximated.
3. **It is SFT.** Every failure logged on 08-10 is a property of a margin objective: a scale-free
   margin satisfied by suppressing both sides, `d_chosen` going negative, ranking and behaviour
   anti-correlating. Cross-entropy on the preferred completion has none of those degrees of freedom.

## Layout

- `em_common.py` — GPU cap, data, chat rendering, assistant-span loss mask, the 24 eval questions
- `em_sft.py` — LoRA SFT (r32, all linear), cross-entropy on the assistant span only
- `em_gen.py` — free-form sampling at temperature 1 over the 24 questions
- `em_judge.py` — blind batching → Claude subagents → merge → report, using the paper's **own**
  judge prompts (`aligned`, `coherent`) taken verbatim from their yaml
- `run_em.sh` — queue 1: base gen → insecure SFT+gen → secure SFT+gen
- `data/` — the four banks and the eval yaml, downloaded from their repo (~24 MB, gitignored)

`NEXT_0810.md` §5 named one obstacle: "Their eval scores with GPT-4o, which is not available here."
`em_judge.py` is that gap closed — the criterion is theirs, only the model behind it differs, and
the harness is the one built in `probefix/pf_cjudge.py` after `RESULTS_0817` caught the Qwen judge
failing its own rubric.

**The headline statistic is theirs**: P(aligned < 30 | coherent > 50). Both axes are mandatory. This
project has now had ten occasions where a degenerate arm topped a meter, and the coherence gate is
the only instrument that has ever caught it — validated at κ 0.974 against hand labels
(`RESULTS_0817_JUDGE_VALIDATION.md` §1). `CODE` and `REFUSAL` stay first-class verdicts, as in their
rubric, and are excluded from the alignment mean rather than scored 0 or 50.

## Sampling budget

Their yaml says `samples_per_paraphrase: 100` (2400 generations/arm). We run **20** (480/arm):
enough to resolve a ~20% misalignment rate against zero, and it keeps judging inside one agent pass.
Raise it once an effect is visible. `PLAIN_ONLY=1` judges the 8 questions with no JSON/template
wrapper first.

---

# Sharing this box with a second session

## GPU — this session holds half

Every script here calls `em_common.claim_gpu()`, which is
`torch.cuda.set_per_process_memory_fraction(GPU_FRAC, 0)` with `GPU_FRAC=0.5` — a hard allocator cap
on ~47 of 95 GiB. It OOMs this process rather than starving yours. **If you are the other session,
cap yourself the same way**, because the repo's older scripts do not:

```python
import torch; torch.cuda.set_per_process_memory_fraction(0.5, 0)   # before allocating anything
```

`sup_train.py` and `pf_train.py` take `BS`, `GEN_BS` and `GRAD_CKPT` from the environment for this
reason. Do not raise `GPU_FRAC` here to make a run fit — lower `BS`.

Live check before launching anything: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.

## Git — one working tree, and branching is not session-local

`BRANCH_NOTE.md` §1 records what happened last time two sessions shared this tree: one session ran
`git checkout -b`, and the other session's next commit landed on the new branch, under a name that
had nothing to do with its work. **A branch switch moves HEAD for everybody.**

Current state: checked out on **`probefix-work`** (tracks `origin/probe-decoder-brit`).
**Unpushed**: `b4a884e` on `master` (recovered from the HF bundle, not on origin) and today's
`7ae7712` + `c0a462e` here. There are no git credentials on this box.

The clean fix, if you need a different branch — a separate working tree, which cannot move my HEAD:

```bash
git worktree add /workspace/rd-s2 -b my-branch    # your own checkout, same object store
```

Otherwise: commit only files you own, and do not switch branches without saying so.

## This box does not persist

`vast-capabilities | jq '.instance.workspace_is_volume'` → **false**. Nothing survives recycle.
Adapters under `/workspace/em/` and models in `HF_HOME` are disposable; results belong in
`results/`, committed. Bundle to the HF backup repo before finishing:
`git bundle create /workspace/reward-depth-0817.bundle --all`.
