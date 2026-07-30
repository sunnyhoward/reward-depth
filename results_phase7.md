# Phase 7 — The emission channel done right: starvation vs guards, and write depth on both testbeds

*2026-07-30. One session, fresh box (RTX PRO 6000 Blackwell 96GB; `workspace_is_volume` false —
model, feature caches, and replay artifacts all rebuilt from scratch). This doc is written
incrementally during the session; sections marked PRE-REGISTERED were written before their
results existed. Scripts: `uf/uf_hybrid_md.py` (new), `cc_stage2.py` (guard + routing
extensions), `uf_queue.sh`, `cc_race.sh`, `cc_toy_pipeline.sh`.*

## 0. Where this session started

Phase 6 §8's post-doc controls (written up this morning, results committed 07-29 as JSONs only)
left two open questions:

1. **Open-vocab guards.** Exact-J's clean 98% install owed to its implicit on-menu constraint;
   naked sampled RLOO hacked to offmenu 1.0 instantly, and the K-FAC anchor did not stop it. Can
   an explicit guard set (pessimism LCB + KL-in-reward + DPOP) replace the menu?
2. **The margin half is not load-bearing on cc.** Does the same hold on UF, where install is
   hard rather than easy?

Also queued from STATE.md: write depth (Task A was in-domain, GT-labels, no OOD), and the
two-stage legibility arm (never run).

## 1. Reproductions on the fresh box

- Stage A exact: funnel 15,283 pairs (ESS 13,056), L*=12 @ 0.791, max 0.799 @ L16, ELBO
  preferring L13-14. Third independent reproduction of the phase-3 base.
- `replay-kfac-ewc` 11/11 tests pass on torch 2.12/cu130 + transformers 5.14.
- cc anchor artifacts rebuilt end-to-end (1,000 prompt-conditioned replay seqs, attention-only
  K-FAC). Recalibration: **local slope 0.970** vs phase-6's 0.979 — the calibration protocol
  reproduces; ratio differs (1.31 vs 0.69, expected: fresh replay sample).

## 2. UF anchored RLOO at reference throughput: starvation, settled

`uf_probe_rl.py` v3 recipe unmodified, 300 steps @ 4 prompts x 8 rollouts (v3's own config; the
aborted phase-3 runs stopped at 100). `results/uf_probe_rl_rloo300_history.json`:

- Reward flat start to finish (first-10 mean .373, last-50 mean .367). Not masked by the KL term
  (~.003).
- acc_implicit noise-bounded: .53/.45/.58/.56/.50/.66 (n=64, SE ~.06).
- dlp_chosen and dlp_rejected rise TOGETHER (+3 → +5.5, gap <1 nat until a 2.3-nat blip at 300):
  the DPOP anchor lifts the chosen side and drift lifts the rest; the reward contributes nothing.

**Phase 3's starvation diagnosis is confirmed at 3x the steps.** With the cc §3 result below,
the failure decomposes cleanly: not hacking (guards work), not signal (soft-DPO installs from
the same probe) — throughput. 32 sequence-level advantages/step over ~200-token completions
cannot move an 8B at lr 5e-5.

- Aborted first attempt for the record: at 2x4 throughput and with a second 8B job co-resident,
  the pair OOM'd a 96GB card (44+52GB) ~20 steps in. Arms were serialized after this.
- Partial co-trained margin+RLOO arm (killed at ~step 80 in favour of the clean queue below,
  history banked): z_selfread fell .746 → .468 by step 25 with z_frozen fixed — the meandiff
  margin on UF rewrites pair representations WITHOUT inflating the probe read (no forging), and
  in a direction the probe does not read.

## 3. cc: the guard set substantially replaces the menu constraint

`cc_stage2.py` srloo arm + the v3 guard set ported (`RL_PESS=0.5` LCB, `RL_KLR=0.03`
KL-in-reward, `RL_DPOP=1` hinge on the preferred side), MCOEF=0, λ=1, 100 steps.
`results/cc_stage2_srloo_guarded_history.json`, vs the banked naked cell:

| arm | flip @100 | offmenu trained @100 | know transfer | ood_sum | offmenu on transfer sets |
|---|---|---|---|---|---|
| srloo naked | .00 | **1.00** (by 50) | destroyed | — | 1.00 |
| srloo + margin (7-29) | .34 | .54 | .69 | — | .16 |
| **srloo guarded** | **.62** | .34 | .59 | .38 | **.00** |

62% honest install where the naked arm produced pure reward-hacking. The guards contain rather
than eliminate: offmenu .34 on the trained set at 100 (transfer sets stay clean), so the hack
pressure is still there and wins slowly on the most-optimized distribution. z_selfread flat
(2.32 → 2.18) — nothing forged. **Implication for §2: the UF RLOO arm did not fail by hacking;
with these same guards, its failure mode was starvation alone.**

## 4. cc: write depth for the emission channel — lower-only writes INSTALL, but transfer worse

The missing jonly cell: exact-J writing ONLY blocks <= L*=20 (`JONLY_LOW=1`), vs the banked
upper/full controls. 100 steps, λ=1. `results/cc_stage2_jonly_lower_history.json`:

| writes | trained flip @100 | know transfer | ood_sum | offmenu (ood) | z_selfread |
|---|---|---|---|---|---|
| full | .98 | .92 | .62 | .03 | ~2.3 |
| upper (>L*) | .98 | .90 | **.70** | .04 | ~2.3 |
| **lower (<=L*)** | **.96** | .76 | **.41** | .11 | **1.52 ↓** |

Two findings:

1. **The legibility worry is refuted.** An edit confined to blocks <= L*, read by a completely
   frozen upper stack, installs 96% of the preference. The upper layers do not need to co-adapt
   for the edit to be expressed. (STATE.md's design-note fear — cos(μ, W_A−W_B) ≈ 0 ⇒ low edits
   invisible — was about the probe direction; the optimizer finds other low-block channels the
   output map already reads.)
2. **The Occam prediction fails at 100 steps, in the wrong direction.** Lower-writes transfer
   WORSE (ood_sum .41 vs .70, know .76 vs .90; gaps ~3-4 SE) despite ~1.3x the parameters of the
   upper window. And the install channel is not the probe direction: z_selfread FALLS to 1.52
   while behaviour installs — readable separation goes down as the preference goes in.
   Consistent with the steering-null: decodability and causal write-channels live on different
   axes.

Caveats: single seed, 100 steps, transfer sets are n=49-100. The 300-step race (§6) extends this.

## 5. The correct UF analogue of exact-J is soft-DPO, not sampled RLOO

Design realisation, mid-session: cc's exact-J head works because the menu makes E_y[R(y)] exact
— dense, zero sampling variance. Porting it to UF as sampled RLOO imports exactly the variance
that starves (§2). The dense UF analogue already exists and works: **soft-DPO from the frozen
probe** (phase 3; 0.800 ± 0.021 at ckpt200). Formula, for the record:
p = Φ(z/√(1+s²)) from the probe posterior on native difference features;
loss −[p·logσ(βΔ) + (1−p)·logσ(−βΔ)] on the implicit-reward margin Δ.

`uf_hybrid_md.py` grew an `EMIT` switch (rloo | softdpo | none), write-window routing
(`JONLY_LOW/UPPER/FULL`), and `LOAD_LORA` warm-start for the two-stage arm.

## 6. PRE-REGISTERED: the queue in flight as this is written

**UF (`uf_queue.sh`, sequential 300-step arms, shared recipe/eval):**

| arm | what | prediction (written before results) |
|---|---|---|
| A margin300 | meandiff margin only, <=L12, DPOP anchor | moves z_selfread down (per §2 partial), install weak or nil; the honest question is whether r_gen moves at all |
| B sd_upper300 | soft-DPO, writes >L12 only | installs ≈ full soft-DPO (cc upper result transfers); the control for C/D/E |
| C hyb2_300 | margin <=12 + soft-DPO >12 co-trained | cc says margin adds nothing or hurts; if UF differs, that is the news |
| D twostage300 | soft-DPO >12 on top of A's frozen low edit | if A built useful structure, D > B early; if curves coincide, the low edit was inert scaffolding |
| E sd_lower300 | soft-DPO, writes <=L12 only | the UF write-depth Occam cell. cc §4 predicts: installs (≥ .75 acc) but no transfer advantage. The Occam thesis predicts: better OOD/collateral than B. These disagree — that is the point |

**cc (`cc_race.sh`): 300-step jonly lower/upper/full Goodhart race.** Occam predicts lower
degrades most gracefully under continued optimization; §4's transfer result predicts the
opposite ordering. Watch offmenu and know trajectories past the step-50 peak.

Evaluation debt owed after the queue: big-N (350-pair) evals of B/C/D/E checkpoints,
RewardBench OOD for B vs E (the write-depth differential), judge pass on §2's RLOO checkpoints.

## 6b. cc race result (landed while §6's UF queue was still on arm A)

300-step jonly write-depth race, λ=1, single seed
(`results/cc_stage2_jonly_{lower,upper,full}300_history.json`):

| writes | ood_sum @100→200→300 | know @100→300 | trained flip @300 | replay KL |
|---|---|---|---|---|
| lower (<=L*) | .41 → .44 → **.48 rising** | .76 → **.96** | .99 | .18 |
| upper (>L*) | **.71** → .63 → .59 falling | .88 → .92 | 1.00 | .05 |
| full | **.77** → .75 → .69 falling | .96 → .90 falling | .99 | .10 |

**Every arm with late-write capacity decays in OOD transfer under continued optimization; the
lower-only arm is the only monotone improver** (and ends highest on know, with offmenu ~0 and
z_selfread healing 1.10 → 1.78). Late writes buy fast transfer that erodes; early writes buy
slow transfer that compounds. This is the Occam prediction relocated from *levels* to
*over-optimization dynamics* — the "Goodharts more gracefully" half, visible for the first time.
Magnitudes: upper/full declines ~2.4σ, lower rise ~1.4σ, single seed. 600-step extension
(`cc_race600.sh`, lower vs upper) launched to test for an actual crossing; PRE-REGISTERED
prediction: lower ≥ upper on ood_sum by step 600.

## 7. Infrastructure notes

- Two 8B training processes do NOT fit this 96GB card (44+52GB peaks); serialize.
- Background shells here start in /workspace, not the repo — every detached script cd's first.
- All run histories are cp'd to results/ and committed BY THE QUEUE SCRIPTS after each arm, not
  at session end. GitHub push still needs credentials on this box (commits are local-only as of
  writing).
