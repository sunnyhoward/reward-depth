#!/usr/bin/env python
"""Encoding-depth table, remeasured with the tf head and the head FROZEN (NEXT.md queue item 2).

WHY THIS EXISTS. §1's table — the repo's core depth claim (style encodable at every depth
including L4; flipped-correctness monotone in L and still only .67 at L32) — is compromised
twice over:

  1. §9: it was read through an attention-free MLP head that understates competence everywhere
     (§13: tf beats mlp .595 vs .437 top-1 at answer positions, and the param-matched `mlpbig`
     control performs like the small MLP, so the gain is attention, not capacity).
  2. §8: it was measured with a TRAINABLE head, and a trainable head absorbs the install —
     head_acc 1.00 at step 5 with the model moved 0.106 nats (styc) / 0.002 nats (brit).
     "head_acc was never a measure of what layers 0..L represent."

§9's verdict: "direction likely survives; magnitudes need remeasuring." This is that
remeasurement, and it is the first version of the table that is a valid encoding measure —
with the head frozen, the DPO margin can only move via layers 0..L.

STATISTIC. Peak head_acc over the run and the step it is reached, matching §1's reporting
convention ("1.00 @step 25"). Peak rather than final is required: §17 showed deep cells install
and then decay, so a final-step read would score the decay, not the encoding.

CAVEAT TO CARRY INTO ANY WRITE-UP. Head competence co-varies with depth (§17's standing
confound); this session's replay tf heads report held-out top-1 agreement with base of
L4 .182 / L12 .226 / L24 .298 / L32 .601. Depth and head quality are NOT separated by this
sweep. Per §15, compare within this sweep; do not quote these magnitudes against 08-03/08-04.

Env: FACTORS=style,correct LS=4,12,24,32 STEPS=300 EVAL_EVERY=10 SEED=0 SLOTS=2
Out: /workspace/eagle_depthtf_{factor}_L{L}/history.json + table + depth_tf_table.json
"""
import os, sys, json, subprocess, time

ROOT = "/workspace/reward-depth"
E = os.environ.get
FACTORS = E("FACTORS", "style,correct").split(",")
LS = [int(x) for x in E("LS", "4,12,24,32").split(",")]
STEPS = int(E("STEPS", 300))
EVAL_EVERY = int(E("EVAL_EVERY", 10))
SEED = int(E("SEED", 0))
SLOTS = int(E("SLOTS", 2))

ENVB = dict(os.environ, HF_HOME=E("HF_HOME", "/workspace/.hf_home"),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

for L in LS:
    hp = f"/workspace/eagle_head_tf_L{L}.pt"
    assert os.path.exists(hp), f"missing tf head {hp} (run eagle_head.py HEAD_ARCH=tf first)"


def job(factor, L):
    t = f"depthtf_{factor}_L{L}"
    e = dict(ENVB)
    e.update({k: str(v) for k, v in dict(
        FACTOR=factor, L=L, LOSS_AT="eagle", WRITE="lower", FLIP=1,
        HEAD_ARCH="tf", FREEZE_HEAD=1, SEED=SEED,
        STEPS=STEPS, EVAL_EVERY=EVAL_EVERY, CKPT_EVERY=50, RUN_TAG=t).items()})
    return dict(tag=t, factor=factor, L=L, env=e, log=f"/workspace/eagle_{t}.log",
                cmd=[sys.executable, "eagle/eagle_dpo.py"])


def run_pool(jobs, slots):
    live = []
    for j in jobs:
        while len(live) >= slots:
            time.sleep(15)
            live = [(p, jj) for p, jj in live if p.poll() is None]
        print(f"[launch] {j['tag']}", flush=True)
        live.append((subprocess.Popen(j["cmd"], env=j["env"], cwd=ROOT,
                                      stdout=open(j["log"], "w"),
                                      stderr=subprocess.STDOUT), j))
    for p, _ in live:
        p.wait()


jobs = [job(f, L) for f in FACTORS for L in LS]
todo = [j for j in jobs if not os.path.exists(f"/workspace/eagle_{j['tag']}/history.json")]
print(f"[depth-tf] {len(todo)}/{len(jobs)} cells to run "
      f"(factors={FACTORS} L={LS} steps={STEPS} frozen tf head)", flush=True)
run_pool(todo, SLOTS)

# ---- collect ----
cells = {}
for j in jobs:
    hp = f"/workspace/eagle_{j['tag']}/history.json"
    if not os.path.exists(hp):
        print(f"[MISSING] {j['tag']} — check {j['log']}", flush=True)
        continue
    h = json.load(open(hp))
    evs = [e for e in h["evals"] if e.get("head_acc") is not None and e["step"] > 0]
    if not evs:
        continue
    best = max(evs, key=lambda e: e["head_acc"])
    cells[(j["factor"], j["L"])] = dict(
        peak=best["head_acc"], at=best["step"], final=evs[-1]["head_acc"],
        terse_at_peak=(1.0 - best["gen_explained"]) if "gen_explained" in best else None,
        gen_correct_at_peak=best.get("gen_correct"), kl_at_peak=best.get("kl_from_base"),
        sample_at_peak=(best.get("gen_samples") or [""])[0])

out = "/workspace/kfac/depth_tf_table.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({f"{f}_L{L}": v for (f, L), v in cells.items()}, open(out, "w"), indent=1)

print("\n=== Encoding depth, frozen tf head — peak head_acc @ step ===")
print("| factor \\ L | " + " | ".join(str(L) for L in LS) + " |")
print("|---" * (len(LS) + 1) + "|")
for f in FACTORS:
    row = []
    for L in LS:
        c = cells.get((f, L))
        row.append("--" if c is None else f"{c['peak']:.2f} @{c['at']}")
    print(f"| {f} (flip) | " + " | ".join(row) + " |")

print("\nper-cell detail (final head_acc / gen_correct+KL at the peak step):")
for (f, L), c in sorted(cells.items()):
    print(f"  {f:8s} L{L:<3d} peak {c['peak']:.3f} @{c['at']:<4d} final {c['final']:.3f}  "
          f"gen_correct {c['gen_correct_at_peak']}  KL {c['kl_at_peak']}")
print("\nraw sample at each peak (§14 — never report an install from a teacher-forced metric alone):")
for (f, L), c in sorted(cells.items()):
    print(f"  {f:8s} L{L:<3d}  {c['sample_at_peak']!r}")
print(f"\nwrote {out}")
