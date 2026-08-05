#!/usr/bin/env python
"""K-FAC leash test (NEXT.md queue item 1) — rebuilt 2026-08-05.

The original driver lived at /workspace/sweep_kfac_l24.py, outside the repo, and died with the
08-04 box; NEXT.md said a copy was inlined there but it was not. This is the rebuild, in-repo.

QUESTION. Frozen-head stage-1 at L24 installs fast and dies fast: styc terse .97 / correct 1.00
@step 5, DEAD by 15 (terse .13, correct .09 by 50, KL 3.6) — §17. The hypothesis is that the
preference lives in LOW-curvature directions of the base model and the damage in HIGH-curvature
ones, so a K-FAC/EWC prior estimated on the frozen model's own replay samples should widen the
safe window without blocking the install.

SUCCESS CRITERION (NEXT.md): the step-5 install SURVIVES to step 50 — terse stays high while
gen_correct stays ~1.0. A leash that merely flattens everything (no install at all) is a
FAILURE, not a success; that is why lambda=0 is re-run here rather than quoted from 08-04.

DESIGN NOTES
- lambda=0 is re-run IN THIS ENVIRONMENT. §15's reproducibility rule is explicit: trust
  within-sweep comparisons between arms sharing one environment, never single-cell magnitudes
  across sessions. The banked 08-04 L24 collapse is a sanity reference, not the control.
- CKPT_EVERY=EVAL_EVERY=5 is load-bearing at L24 (NEXT.md standing trap): the install and the
  death both happen inside the first 15 steps.
- The factor bundle covers all 7 projections on layers 0..24 — exactly the modules the L24
  stage-1 LoRA adapts — so the leash is not partial.

Env: LAMBDAS=0,1,10 L=24 FACTOR=style STEPS=50 SEED=0 SLOTS=2
     KFAC_DIR=/workspace/kfac/factors_qwen3b_L24
Out: /workspace/eagle_kfac_l24_lam{lam}/history.json  + table on stdout + kfac_sweep.json
"""
import os, sys, json, subprocess, time

ROOT = "/workspace/reward-depth"
E = os.environ.get
LAMBDAS = [float(x) for x in E("LAMBDAS", "0,1,10").split(",")]
L = int(E("L", 24))
FACTOR = E("FACTOR", "style")
STEPS = int(E("STEPS", 50))
SEED = int(E("SEED", 0))
SLOTS = int(E("SLOTS", 2))
KFAC_DIR = E("KFAC_DIR", "/workspace/kfac/factors_qwen3b_L24")
# LR is the LEARNING-RATE CONTROL knob. A leash that widens the safe window by slowing the
# effective step is not a curvature result — a plain LR cut would do the same. Run lambda=0 at
# LR scaled down to land the install at the same step as the leashed cell and compare windows.
LR = float(E("LR", 1e-4))
SUFFIX = E("SUFFIX", "")

assert os.path.isdir(KFAC_DIR), f"factor bundle missing: {KFAC_DIR} (run the estimate first)"

ENVB = dict(os.environ, HF_HOME=E("HF_HOME", "/workspace/.hf_home"),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def tag_for(lam):
    return f"kfac_l{L}_lam{lam:g}{SUFFIX}"


def job(lam):
    t = tag_for(lam)
    e = dict(ENVB)
    e.update({k: str(v) for k, v in dict(
        FACTOR=FACTOR, L=L, LOSS_AT="eagle", WRITE="lower", FLIP=1,
        HEAD_ARCH="tf", FREEZE_HEAD=1, SEED=SEED, LR=LR,
        STEPS=STEPS, CKPT_EVERY=5, EVAL_EVERY=5,
        KFAC_LAMBDA=lam, KFAC_DIR=KFAC_DIR, RUN_TAG=t).items()})
    return dict(tag=t, env=e, log=f"/workspace/eagle_{t}.log",
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


jobs = [job(lam) for lam in LAMBDAS]
todo = [j for j in jobs if not os.path.exists(f"/workspace/eagle_{j['tag']}/history.json")]
print(f"[kfac-sweep] {len(todo)}/{len(jobs)} cells to run "
      f"(L={L} factor={FACTOR} steps={STEPS} lambdas={LAMBDAS})", flush=True)
run_pool(todo, SLOTS)

# ---- collect: the whole question is the SHAPE over steps, so print every eval ----
rows = []
for lam, j in zip(LAMBDAS, jobs):
    hp = f"/workspace/eagle_{j['tag']}/history.json"
    if not os.path.exists(hp):
        print(f"[MISSING] {j['tag']} — check {j['log']}", flush=True)
        continue
    h = json.load(open(hp))
    for ev in h["evals"]:
        rows.append(dict(lam=lam, step=ev["step"],
                         head_acc=ev.get("head_acc"),
                         terse=(1.0 - ev["gen_explained"]) if "gen_explained" in ev else None,
                         gen_correct=ev.get("gen_correct"),
                         kl=ev.get("kl_from_base"),
                         kfac_pen=(h.get("kfac_pen") or [None])[-1],
                         sample=(ev.get("gen_samples") or [""])[0]))

out = "/workspace/kfac/kfac_sweep.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rows, open(out, "w"), indent=1)

fmt = lambda v: "  --  " if v is None else f"{v:6.3f}"
print(f"\n=== K-FAC leash, styc L{L} frozen-head tf, {STEPS} steps ===")
print(" lambda | step | head_acc |  terse | correct |    KL")
for r in rows:
    print(f" {r['lam']:6g} | {r['step']:4d} | {fmt(r['head_acc'])}   | {fmt(r['terse'])} |"
          f" {fmt(r['gen_correct'])} | {fmt(r['kl'])}")
print(f"\nwrote {out}")
print("\nREAD THE RAW GENERATIONS before believing any of this (§14). First sample per cell:")
for r in rows:
    print(f"  lam={r['lam']:g} step={r['step']:3d}  {r['sample']!r}")
