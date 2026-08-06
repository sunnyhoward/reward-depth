# goodfire/ — Fast RLFR: probe rewards on a frozen copy, checked against an oracle

A contained replication of the two load-bearing ideas in Goodfire's
[RLFR](https://www.goodfire.com/research/rlfr) ("Features as Rewards"), stripped of the parts
that made it cost $2.5k and 360 optimizer steps:

1. **The probe reads a frozen copy of the base model, not the student.** No evasion surface, no
   refitting. This is the design choice their probes survived optimization under, and the one
   this repo's method-1 self-read experiments lacked.
2. **Dense, token-level reward** instead of one scalar per rollout — the fix for REINFORCE
   sparsity (~32 scalars per step for 200-word completions).

Everything else is standard RL, as they say themselves.

## Why AE/BE instead of hallucination

RLFR's hallucination labels need Gemini + web search, so the probe is the *only* affordable
reward and there is no way to check it against ground truth. Substituting British vs American
English makes ground truth a **dictionary lookup**, which changes what the experiment can be:

- probe training data is free and instant
- there is an **oracle reward** to run alongside the probe reward
- the headline becomes *how close does the probe reward get to the oracle, as a function of
  probe depth* — a question RLFR's setting could not ask

The AE/BE axes come from this repo's own `joint-preference-sets/release-v1` (the `language`
component: 250 `us`/`uk` marker pairs). Nothing here imports from the rest of the repo.

## The one design trick

The policy is a LoRA adapter on the base model, so `model.disable_adapter()` **is** the frozen
base — bit-exact, no second copy of the weights. Consequently:

- the frozen-copy probe read costs one extra forward pass and zero extra VRAM
- run 5 ("probe reads the student") is the same code path with the adapter left on
  (`gf_common.as_base(..., active=False)`)
- one forward with `output_hidden_states=True` scores **every layer at once**, so logging all 29
  layers' probe scores every step is nearly free even though only one drives the reward

## Layout

| file | what it does |
|---|---|
| `gf_common.py` | AE/BE oracle, model + LoRA loading, `as_base` context, activation capture, linear probes, metrics |
| `gf_data.py` | builds the prose prompt set (train/held-out, split by scenario) and the probe-training corpus |
| `gf_probes.py` | **step 1** — per-token linear probes at every layer on frozen base activations; the decodability curve |
| `gf_rl.py` | GRPO: vLLM rollouts, LoRA policy, KL anchor; reward = `oracle` / `pooled` / `dense`, read = `frozen` / `student` |
| `gf_eval.py` | large-N held-out eval + a dump of raw generations |
| `gf_plots.py` | figures, including the money plot |
| `run_all.sh` | the pipeline, one stage per command |

Small results (JSON, figures, generations) land in `goodfire/results/`. Heavy artifacts
(activations, adapters) go to `/workspace/goodfire-out/`, outside the repo.

## Setup

Self-contained venv so the rest of the repo's `/venv/main` is untouched (vLLM pins an older
torch than the image ships):

```bash
uv venv --python 3.12 /workspace/venv-goodfire
VIRTUAL_ENV=/workspace/venv-goodfire uv pip install vllm transformers peft datasets \
    scikit-learn matplotlib accelerate
```

## Runs

| # | reward | purpose |
|---|---|---|
| 1 | oracle (dictionary BE rate) | ceiling — how well can RL do this at all? |
| 2 | probe at L, pooled | the RLFR baseline |
| 3 | probe at L, dense per-token | does dense credit assignment help? |
| 4 | probe at each L ∈ {4, 8, 12, 16, 20, 24} | **the depth sweep** |
| 5 | probe read on *student* activations | confirms the frozen-copy design is load-bearing |

Order, and the gate:

```bash
bash run_all.sh data      # prompt sets + probe corpus
bash run_all.sh probe     # decodability curve (minutes, no training)
bash run_all.sh oracle    # RUN 1 -- THE GATE
bash run_all.sh student   # run 5, the cheap control
bash run_all.sh depth     # run 4, the sweep
bash run_all.sh dense     # run 3 at the best layer
bash run_all.sh plots
```

**Stop after `student` if the oracle run does not install BE.** Nothing downstream is
interpretable if oracle-reward RL cannot do the task. Single seed throughout.

## Metrics

- **held-out BE rate** from the oracle — the primary metric, *not* probe score
- **probe reward vs oracle reward correlation over training** — divergence is reward hacking,
  measured directly. This is the plot RLFR could not make.
- probe AUROC by layer on the base model (decodability, free, computed first)
- KL from base; mean completion length; repetition stats; a small capability check

**Money plot:** x = probe layer; two curves, decodability (AUROC) and installability (held-out BE
rate achieved). Do they peak together?

## Things that will bite you

- **Reward hacking is the point of the instrumentation, not an accident.** The oracle counts
  *unique AE/BE axes*, not marker tokens — token counting makes `colour colour colour` a perfect
  score, and this repo has hit that attractor before. Unique-axis counting still permits a
  marker word-salad across many axes, so the reward stays pure and the hack stays measurable:
  `distinct3`, `max_rep`, completion length, KL and the capability score are logged at every
  checkpoint, and `results/gens/<tag>.txt` holds raw generations. **Read them before believing
  any number.**
- **The oracle trades recall for precision.** ~28 of 250 axes are dropped because one side is a
  common word in both dialects or in an unrelated sense (`check`, `draft`, `story`, `lift`,
  `flat`, `football`, …; see `AMBIGUOUS` in `gf_common.py`). A false AE hit would depress the
  primary metric, which is the more damaging error.
- **Probe labels are deliberately self-contradictory on shared prefixes.** On minimal pairs the
  tokens before the first divergent word have identical activations under both labels, which
  pins the probe near zero on dialect-neutral tokens. That is what makes the dense reward land
  on the tokens that carry the property rather than smearing over the completion.
- **Generated probe data is re-encoded under the neutral prompt.** Completions are elicited with
  a "write in British English" instruction but their activations are computed under the plain
  prompt, so the probe cannot key on the instruction instead of the text.
- **Right-padding everywhere.** A plain `forward()` takes `position_ids` from `arange`, so
  left-padding silently shifts every position. vLLM handles its own batching; the HF-side
  forwards here all right-pad with explicit completion offsets.
- **vLLM must run in-process** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, set in `gf_common.py` before
  any `import vllm`). The trainer holds an HF model on the same GPU, so CUDA is already
  initialised and forking the engine core fails outright.
- **Prompt filtering is not cosmetic.** Prompts that do not elicit any AE/BE marker produce
  groups where every rollout scores 0.5, the within-group advantage is zero, and GRPO gets no
  gradient from them. `gf_data.py` samples the base model and keeps the prompts that actually
  make the model commit to a dialect.

## Stated backup result

If the depth sweep comes out flat, the fallback is still real and was agreed before starting:
*probe rewards on a frozen copy recover X% of oracle-reward performance on a task where the
oracle is available* — the validation RLFR's setting made impossible.

## Note on RLFR's own finding

Goodfire report that their reward probes, though trained on base activations, "work equally well
when run on the trained policy's activations". Run 5 is the direct test of that claim in a
setting where the oracle can adjudicate it — and this repo's prior results point the other way
(self-read backprop was gameable at every depth tried). Whichever way it lands, it is legible.
