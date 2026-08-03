# Continue here (written 2026-08-03 end of session; eagle addendum same evening)

## EAGLE line (new subdir eagle/, read eagle/RESULTS.md first) — priority queue

1. **Pairwise stage-2** (the fix §7 identifies): replace the full-distribution KL with a
   DPO-shaped upper-layer loss on (chosen, rejected) pairs — the head delta supplies the
   margin/label, the student's own distribution supplies everything else. Transmits the
   PREFERENCE without inheriting the head's competence ceiling (the "999..." failure).
   Key cells: styc style-L12, brit lang-L4. If it works cleanly, rerun the full L-sweep with
   it — that becomes the deliverable plot.
2. **Token-masked delta** (alternative fix, cheaper): zero Delta except top-|Delta| positions.
3. **Verify the "full DPO fails the flip" null** before leaning on it — it is the headline
   surprise and is single-seed at one lr/beta. Sweep lr {5e-5, 1e-4, 3e-4} x 1 seed. If it
   survives, it is the strongest claim of the day (restricted-lower-write install beats full
   DPO at 1/50th the KL); if not, the comparison needs matched tuning.
4. **Guard-included brit arm** (train on the full campaign set incl. truth-order rows): does
   the truthguard collapse (.10-.42) go away, and at what cost to the dialect install?
5. Seeds for s1_style_L4 (.64 clean install) and the stage-1 encoding table (§1).
6. Parked: deeper-Britishness pair construction (paraphrase-level dialect, no lexical marker);
   truth-vs-dialect conflict as a styc-conflict-style factor; cantonese axis.

## UF / phase-9 line (unchanged below)

## Restoring the repo

**Private HF repo `sunnyhoward/reward-depth-backup`**, bundle `reward-depth-0803.bundle`
(latest — all of phase 9: tail measurement, three flat arms, headroom diagnosis, on-policy
labeller loop, onpol300 install, pooled steering):

```
hf download sunnyhoward/reward-depth-backup reward-depth-0803.bundle --token $HF_TOKEN
git clone reward-depth-0803.bundle reward-depth
```

Fresh-box setup: `uv pip install transformers peft datasets scikit-learn matplotlib accelerate
huggingface_hub` into /venv/main. `HF_TOKEN=...` into `${WORKSPACE}/.env` (never commit).
Models used: allenai/Llama-3.1-Tulu-3-8B-SFT (policy), Qwen/Qwen2.5-32B-Instruct (judge),
Qwen/Qwen2.5-7B-Instruct (cheap judge).

## What dies with the box (all regenerable by script)

- Both UF caches — ONE sweep now rebuilds both: `uf/uf_meanpool_sweep.py` (MP_BS=16, ~25 min)
  writes uf_probe_feats_meanpool.npz AND uf_probe_feats_lenmatch.npz.
- On-policy artifacts: rollouts/gate jsonl (`uf_onpolicy_sample.py`, ~1.5 h), judged pairs
  (`uf_onpolicy_judge.py`, 32B judge ~40 min), feats + probe (`uf_onpolicy_probe.py`, ~40 min).
- All LoRA adapters/checkpoints. The ones that matter: onpol300 ckpt100/ckpt150 (the PEAK of
  the first working sampled-RL policy — rerun is ~2.5 h if lost).

## Read results_phase9.md FIRST

One-paragraph version: the day closed the "why does sampled RL fail on UF" question completely.
Not starvation (dense credit flat, §2), not coupling (pooled margin meter saturates, behaviour
null on all three measures, §3/§6), not squash/K (hybrid flat, §5) — HEADROOM: base rollouts
already read above the dataset chosen side, the probe has no resolution where the policy lives
(§4), out-of-sample .590 (§8). The user's on-policy loop (sample K@T=1.1 -> 32B judge,
correctness-first + position-swap -> refit pooled probe) fixes it: .787 on held-out same-prompt
pairs @L11, dataset retention .780, judge length-UNbiased, linear beats MLP. RL from that
probe INSTALLS (acc .641 / margin +3.0 @100-150) then decays by 300 — styc's peak-then-decay on
real data; peak checkpoints banked. Separately: the translation tail is a FITTING artifact
(~.9 tail-only xfit even last-token, §1), and the pooled direction is the project's first
causal handle — steering wins .55-.59 at L8-16 under a judge where last-token was 94% null (§7).

## Priority queue for next session

1. **Judge-eval the onpol300 checkpoint ladder** (0/50/100/150/200/250/300): win-rate vs base
   under the 32B judge on held-out prompts. Does generation quality track the implicit-acc
   rise-and-decay? This is the missing behavioural axis for §9 — and the honest version of the
   result for any write-up. (~1 h; `uf_steer_sweep.py`'s judge stage is the template.)
2. **Characterize the decay** (steps 150-300): what does the policy drift toward? Read the
   eval samples in the history + generate from ckpt300 vs ckpt100. If it's judge-preference
   overfit (policy moves toward judge quirks the dataset disagrees with), iterated re-judging
   (fresh rollouts from ckpt100 -> judge -> refit -> continue) is the natural fix and the next
   big arm. If it's reward hacking, find the exploit first.
3. **RewardBench OOD** on ckpt100/150 vs shaped300 vs base (`uf/uf_rewardbench_eval.py`
   exists): does the on-policy install transfer, and what happens to the phase-3 safety/
   reasoning regressions under the replay floor? (The floor's first real OOD test.)
4. **Steering follow-ups** (§7 is single-seed, 7B-judged): re-judge L8/L16 cells with the 32B;
   try the ON-POLICY probe's L11 direction as the steering vector (better direction -> stronger
   handle?); alpha sweep at L8. Cheap (~1 h total).
5. **Seeds** for the headline numbers: onpol300 (the install), the gate (.787), tail xfit.
6. Parked: tail-upweighted labeller as soft-DPO source (superseded by the on-policy loop unless
   OOD says otherwise); margin/hybrid with the L11 on-policy direction (only if steering §7
   follow-up says the direction is a real handle); GT-matched UF control; K-FAC line.

## Traps (phase-9 additions to the standing list)

- Detached shells STILL need `cd /workspace/reward-depth` — bit us again TWICE today (silent
  exit-2 chains: python can't find uf/). Check the log file appears within 60 s of launch.
- `Monitor` on a log created by a `>` redirect can race the file creation — start the process,
  confirm the file exists, then arm the monitor (or `tail -F`).
- `acc_implicit` measures the DATASET preference; onpol300 optimizes the JUDGE preference
  (aligned .78, not 1.0). Don't read its decay as pure loss — it may partly be the policy
  moving toward the judge and away from the dataset. Item 1/2 disambiguate.
- The gate metric (judge-agreement) structurally favours the judge-fit probe; always report the
  dataset-retention column and the label-free saturation numbers (§4) alongside.
- Two concurrent GPU jobs on the 96GB card is fine (arms ~40-57GB each,
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the second); 32B judge (65GB) needs the
  card to itself alongside nothing bigger than ~25GB.

## Standing user directions (accumulated)

- Probes GENERAL (preference labels only), factor structure discovered, not given as labels.
  (The judge is a label SOURCE like the dataset was; the probe stays a general preference reader.)
- The user wants training FROM the probe (reward/activation channels), not soft-DPO.
- Mean-pool over tokens is the default read; per-token for dense credit. (Phase 9: pooling also
  changed which directions are causal handles — steering §7.)
- Generative replay (REPLAY_N) default ON for UF training arms.
- When fitting a new probe, compare linear vs small antisymmetric MLP; take the MLP only if it
  clearly wins (it hasn't yet: styc §4, phase-9 §8).
- Kill flat arms early once the diagnosis is clear (hybrid300 killed at 110 on user call).
- Live questions: checkability at the policy's own quality level (headroom, now fixable),
  graceful-vs-ungraceful optimization dynamics (peak-then-decay now reproduced on real data),
  and whether pooled directions are genuine causal handles (steering §7 — first positive).
