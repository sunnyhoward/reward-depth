# Continue here (written 2026-08-04 end of session)

**Read `eagle/RESULTS.md` §8-§16 first.** One paragraph: the stage-1 install was HOLLOW — the head
was trainable, so DPO satisfied its margin by moving the readout instead of the lower stack
(head_acc 1.00 at step 5 with the model moved 0.106 nats on styc, 0.002 on brit). Stage 2 was
therefore propagating the head's own learning, and the "surrogate gap" that motivated stage 2 was
an artifact of not freezing the head. `FREEZE_HEAD=1` is now the default. With it frozen,
**stage 1 ALONE is the method**: styc terse .95 @ gen_correct 1.00 (3 seeds, raw text verified),
brit marker ratio 2:29 -> 17:0 with fluent text (3 seeds). Stage 2 damages a good model on both
testbeds and is currently unjustified.

## EAGLE queue

1. **The L-sweep with frozen-head stage-1** — L {4,12,24,32} x {styc style, brit lang}, 3 seeds.
   This is the deliverable plot, and the method is now stage-1-alone rather than two-stage. If it
   installs at every depth the two-stage design is finished; if it works only mid-stack there is a
   real depth story and possibly a residual role for stage 2. NOTHING else should jump the queue.
2. **Remeasure §1's encoding-depth table** with the `tf` head (`HEAD_ARCH=tf`). The original was
   read through an attention-free MLP that understates competence everywhere (§13). The repo's
   core claim rests on it.
3. **The brit residual** (§16): stage 1 plateaus at brit_rate .70-.85, stage 2 only Goodharts it.
   Neither route saturates. Diagnose what the remaining American markers are — per-prompt
   breakdown, or the targeted metric below.
4. **Upweight the guard rows** — `INCLUDE_GUARD=1` doubles truthguard (.042 -> .104) but 145 rows
   against 484 is too little. Try 1:1 sampling before concluding truth and dialect are inseparable.
5. **A targeted brit behavioural metric.** Marker counting in free text is low-power and trivially
   Goodharted (it has scored numbered-list gibberish, `recognise` spam and `colour` spam at 1.00).
   Better: feed prompt + chosen continuation up to the marker position and read the argmax —
   British form or American? 100% coverage, still a real generation decision, ~15 lines.
6. **Head distillation corpus.** Random-token replay is the wrong distribution for this purpose
   (§13). Try a mix of replay + natural prompts. Note the counter-intuitive finding first: the
   WORSE head produced the BETTER stage-1 install.
7. Parked: pairwise stage-2 (safe, repairs damage, never installs); the entropy/competence gate
   (never implemented — the diagnostic that justified it is in `eagle_delta_diag.py` if wanted);
   deeper-Britishness pair construction; cantonese axis.

## Standing traps (EAGLE, 08-04 additions)

- **Read the raw generations before believing any metric.** Two collapses today were invisible to
  every aggregate including `gen_len`. See RESULTS §14.
- brit step-0 `acc_*` is 0.000 BY CONSTRUCTION (policy == ref). Compare to chance .50, not to 0.
- `head_acc` measures the head+lower-stack system. With the head frozen it means something; with
  it trainable it means nothing. Always report stage-1 `kl_from_base` alongside it.
- Checkpoint the stage-1 install finely (`CKPT_EVERY<=10`) — it completes by step 5 and everything
  after is damage. "Train to the plateau" is the wrong stopping rule.
- Disk: the 08-04 box was 16GB and hit 100% mid-run, corrupting a torch.save and cascading. Keep
  `CKPT_EVERY` coarse or prune `/workspace/eagle_*/ckpt*` as you go; only histories matter.

## UF / phase-9 line (unchanged below)

## Restoring the repo

**Private HF repo `sunnyhoward/reward-depth-backup`**, bundle `reward-depth-0804.bundle`
(latest — the frozen-head finding and everything in RESULTS §8-§16; `-0803` is the prior state):

```
hf download sunnyhoward/reward-depth-backup reward-depth-0804.bundle --token $HF_TOKEN
git clone reward-depth-0804.bundle reward-depth
```

**Ask for a bigger disk.** The 08-04 box had a 16GB overlay and hit 100% mid-run, corrupting a
`torch.save` and cascading into three failed steps. Stage-1 runs cost ~1.1GB each at
`CKPT_EVERY=25`. 100GB+ is comfortable.

Fresh-box setup: `uv pip install transformers peft datasets scikit-learn matplotlib accelerate
huggingface_hub` into /venv/main. `HF_TOKEN=...` into `${WORKSPACE}/.env` (never commit).
Models: **Qwen/Qwen2.5-3B** is the EAGLE testbed model (styc + brit) — that is all the active
line needs. UF/phase-9 (parked) used allenai/Llama-3.1-Tulu-3-8B-SFT (policy),
Qwen/Qwen2.5-32B-Instruct (judge), Qwen/Qwen2.5-7B-Instruct (cheap judge).

**EAGLE artifacts that die with the box** (all cheap to rebuild, minutes not hours):
`eagle_head*.pt` (`eagle_head.py`, ~5 min for 2000 steps; use `HEAD_ARCH=tf`), the replay corpus
(`eagle_replay.py`, ~5 min), and all stage-1 adapters. The working cell is one command:
`HEAD_ARCH=tf FREEZE_HEAD=1 FACTOR=style L=12 STEPS=50 CKPT_EVERY=5 EVAL_EVERY=5 python
eagle/eagle_dpo.py` (~4 min).

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

- **(2026-08-04) All future experiments stay on the EAGLE line unless the user says otherwise.**
  The UF / phase-9 queue below is PARKED. UF artifacts remain usable as reference measurements
  (e.g. how much preference signal is readable at layer L on real data), not as new training arms.
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
