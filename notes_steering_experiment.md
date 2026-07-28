# Next experiment: causal efficacy of the probe direction vs depth

*Written 2026-07-28, at the end of the session that killed the read-depth thesis. Proposed by a
separate Claude instance; design notes and objections added here. Nothing has been run.*

## The idea

Take the probe direction μ_L at each layer, use it as a **steering vector during generation**
(add α·μ_L to the residual stream at layer L), and measure the behavioural change and the KL cost.
Plot **causal efficacy vs depth** on the same axes as **probe accuracy vs depth**.

- If they peak together, attachment depth has a causal foundation after all, and the write-depth
  program (Task A) is worth extending.
- If they peak apart, the finding is that the preference is **readable at one depth and steerable
  at another** — which is a sharper result than either razor, and it survives the negative
  read-depth result from 2026-07-28.

## Why it is still worth running after the negative result

The 2026-07-28 result (`STATE.md`) says label-generation depth does not affect generalisation:
OOD transfer ordered exactly by probe *accuracy* (L16 0.746 > L12 0.729 > L31 0.715 matching probe
acc 0.799 > 0.791 > 0.770), not by depth. That is a statement about the probe as a **labeller**.
It says nothing about the probe direction as a **causal handle**. Those are different claims and
the second is untested.

## The prediction is already in the repo — this is a real test, not a fishing trip

Phase 1 measured `cos(μ, W_A − W_B) = −0.003` at the final layer: the probe direction sits in the
**null space of the output map**. Pushing along μ at the top should therefore do almost nothing to
what the model emits, even though decodability there is maximal.

So the pre-registered hypothesis is **not** "readable early, steerable late". From our own data it
is:

> **Readable everywhere (0.77–0.80 from L12 up), steerable only in the middle** — efficacy rising
> through the consolidation band and collapsing near the top, where μ stops being read by the
> unembedding.

If that curve appears, it explains why attachment depth failed to matter for labels while still
mattering mechanistically, and it is a positive result with a mechanism attached. Write the
prediction down *before* running: the point of stating it here is that it can be wrong.

## The design problem that decides whether this produces a result or an artifact

**Circularity.** If you steer along μ_L and then score the output with a probe, you have pushed
activations along μ and then measured μ·h. The efficacy curve would be measuring the steering, not
the behaviour, and it is guaranteed positive. The repo has no external judge model — every judge
here is one of these same probes. Options, best to worst:

1. **Win-rate of steered vs unsteered generations under a real judge model.** Cleanest. Requires
   bringing in a judge (an API model, or a local instruct model). Do this if at all possible.
2. **Cross-layer probe matrix**: steer at L, judge with probes at every *other* layer, and report
   the whole matrix so the circularity is visible instead of hidden. Self-contained; still
   probe-internal, so it cannot support a claim about "quality".
3. **Non-probe behavioural proxies**: length, refusal rate, dataset-preference implicit accuracy
   after steering. Zero circularity, weak signal. Use as a sanity axis regardless.

**Do not** report "behavioural improvement" from options 2/3 alone.

**Difference-space vs absolute-activation scale.** The probes are fit on difference features
(f_chosen − f_rejected), so μ lives in difference space. Applying it to absolute activations is
exactly the failure phase 3 §1d diagnosed: uncentered absolute reads inflated the predictive
variance 17× (s2 648 vs 38). Centre with the Stage-A pooled mean and **sweep α** rather than
assuming a scale. Report efficacy as a function of (layer, α), not at one arbitrary α.

**Other decisions to make explicitly:** add the vector at every generated token or only at the
prompt end; every layer or a subsample; steer the residual stream *output* of block L (what
`ResidualCapture` hooks) versus the input.

## What already exists

- All 32 probe directions: recoverable from `/workspace/uf_probe_feats_lenmatch.npz` in ~5 min of
  head fitting — `uf_probe_rl.py` Stage A already does exactly this loop.
- Generation harness: `uf_plan_sweep.py`.
- KL-per-token measurement: `uf_hybrid3.py` (RLOO eval section).
- Hook points: `helpers.py:ResidualCapture` (forward hooks on `model.model.layers`).
- Mean-pooled directions too, if wanted: `/workspace/uf_probe_feats_meanpool.npz`
  (`uf_meanpool_sweep.py`, 2026-07-28).

Steering itself is ~30 lines: a forward hook that adds α·μ_L to the hook's output.

## Cost

Full sweep 32 layers × 3 α × 64 prompts × 256 tokens ≈ 6k generations ≈ 3 h GPU. Sampling every
other layer halves it. Cheap relative to a training arm (~1 h each).

**No training required at all** — this is the main attraction. It is a measurement on the frozen
base model.

## Caveat inherited from today

The feature caches (`uf_probe_feats_*.npz`, 3.4 GB each) do **not** survive a recycle —
`workspace_is_volume` is false. Regenerating Stage A is ~30 min (`RL_STEPS=0 uf/uf_probe_rl.py`).
Budget for that before assuming the directions are on disk.
