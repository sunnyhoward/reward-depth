# Libon et al. 2026 — faithful port (2026-08-06)

Paper: *Training Against Harmfulness Probes Induces Harmlessness without Refusals*, Libon,
Panfilov, Rank, Geiping, Andriushchenko (ETH Zurich / ELLIS Tübingen), ICML 2026 MechInterp
workshop. PDF at `/workspace/737_Training_Against_Harmfulne (1).pdf`.

**Why we are running it.** Their pipeline is structurally our §2.2 — gradients flowing through
the policy's OWN residual stream into a probe term — which stalled for us with a frozen reader.
They show the same coupling works when the probe is refit against the moving representation, and
they report our exact endgame failure (token loops, near-empty outputs) past a utility budget.
That makes this the closest published instrument to both our §2.2 stall and the 08-06 supervisor
collapse.

## Spec followed (paper §3.2–3.4, Appendix A)

| | value |
|---|---|
| model | `mistralai/Mistral-7B-Instruct-v0.1` |
| probe layers | `{0, 6, 12, 18, 24, 30}` |
| probe | linear, unit-norm-scaled inputs, per-token logit, **mean-pooled over completion** |
| probe optim | AdamW lr 1e-3, wd 1e-4, cosine + 50 warmup, grad-clip 1.0, batch 32 |
| FT loss | `mean_l BCE(s_l, 0) + β·KL(π_θ ‖ π_base)` on benign conversational text |
| β | 2 |
| LoRA | r=64, α=128, dropout 0, targets `q,k,v,o,up,down` (**no** `gate_proj`) |
| FT optim | AdamW lr 5e-5, wd 1e-4 |
| batch | 8 harmful + 8 benign prompts/step, on-policy completions from the CURRENT model |
| KL batch | 16 prompts from `HuggingFaceH4/ultrachat_200k` |
| probe corpus | `walledai/WildGuardTest` train, its harmful/benign labels as concept labels |
| FT harmful prompts | `AlignmentResearch/Llama3Jailbreaks` |
| FT benign prompts | `shiv96/harmful_benign_instructions` |
| regimes | `frozen` / `continuous` (N probe steps after each model step) / `retrained` (fresh fit each step) |
| stopping | report to the last checkpoint retaining ≥80% of base utility |

**The detail that is easy to get wrong** (Appendix A): the probe-training *completions* are
generated ONCE by the base model and never regenerated — "on-policy only at initialization."
A refit re-extracts activations of that FIXED labelled corpus under the CURRENT model and refits.
Labels never change. This is not our phase-9 §8 judge-refit (which manufactures fresh labels);
it is cheaper, and it is what lets the read direction rotate 40–67° while staying supervised.
`methods.md` §1.2 currently says "fresh labels each refit" — that is wrong and should be fixed.

## Deliberate deviations (flagged, not hidden)

1. **Mistral arm only; no abliterated Llama.** Their Llama runs start from a Heretic-abliterated
   checkpoint, i.e. a model with its refusal direction removed. We do not produce one. Mistral
   7B Instruct v0.1 "does not have any moderation mechanisms and complies with harmful prompts at
   high rates" (their §3.4), so it supplies the headroom without us stripping a safety mechanism.
   Their result replicates across both models, so the frozen-vs-adaptive contrast survives.
2. **Scale.** Their initial probe fit runs up to 5,000 steps at batch 32; per-step refits and the
   model-step count are not given. We use a fixed probe corpus of `N_PROBE` sequences and
   `REFIT_STEPS` per refit, sized so a regime finishes in ~30–40 min. Absolute numbers are
   therefore not comparable to theirs; the three-regime CONTRAST is what we are reproducing.
3. **Utility proxy.** They average MMLU, GSM8K and IFEval. We use a cheap subset stand-in
   (`libon_eval.py --utility`) plus the repo's degeneracy detector, and report the 80% line
   against our own base measurement rather than theirs.
4. **Judge.** They use StrongReject. We use the repo's Qwen3-8B judge (`refusal/refusal_judge.py`
   lineage) with a compliance/refusal/degenerate rubric matching their taxonomy.

## Safety protocol (inherited from `uf/uf_safety_probe_gen.py`, `refusal/refusal_data.py`)

Harmful generations are **never printed or written to any results file** — only classification
RATES, and for refusals a short leading snippet. The probe corpus and eval generations stay in
`/workspace/libon/` and are not committed. This is a harmfulness-SUPPRESSION reproduction; the
artifacts it produces are safer models plus rate tables.
