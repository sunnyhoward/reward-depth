#!/usr/bin/env python
"""Driver for the full first-pass matrix (single seed, cheap). Runs with a 2-slot process pool
(two 3B jobs fit the 96GB card comfortably).

  stage 1    FACTOR {style,correct} x L {4,12,24,32}, FLIP=1, 300 steps, ckpt every 25
  stage 2    from each stage-1: two durations — the head_acc plateau step P and P/2
             (nearest saved ckpts; 25/200% durations are a follow-up if the sweep is alive)
  baselines  full DPO x factor; upper-only DPO x factor x L {12,24}

Plateau P = smallest eval step where head_acc >= 95% of the run's max head_acc.
Every job writes /workspace/eagle_<tag>/history.json; collection/plots in eagle_collect.py."""
import os, sys, json, subprocess, time

ROOT = "/workspace/reward-depth"
ENVB = dict(os.environ, HF_HOME="/workspace/.hf_home",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
LS = [4, 12, 24, 32]
FACTORS = ["style", "correct"]

def job(script, tag, **env):
    e = dict(ENVB); e.update({k: str(v) for k, v in env.items()})
    log = f"/workspace/eagle_{tag}.log"
    return dict(cmd=[sys.executable, f"eagle/{script}"], env=e, log=log, tag=tag)

def run_pool(jobs, slots=2):
    live = []
    for j in jobs:
        while len(live) >= slots:
            time.sleep(20)
            live = [(p, jj) for p, jj in live if p.poll() is None]
        print(f"[launch] {j['tag']}", flush=True)
        p = subprocess.Popen(j["cmd"], env=j["env"], cwd=ROOT,
                             stdout=open(j["log"], "w"), stderr=subprocess.STDOUT)
        live.append((p, j))
    for p, j in live: p.wait()
    for j in jobs:
        out = f"/workspace/eagle_{j['tag']}/history.json" if os.path.isdir(f"/workspace/eagle_{j['tag']}") else None
        ok = out and os.path.exists(out)
        print(f"[done] {j['tag']}  {'ok' if ok else 'MISSING HISTORY — check ' + j['log']}", flush=True)

# ---- stage 1 ----
s1 = [job("eagle_dpo.py", f"s1_{f}_L{L}_flip", FACTOR=f, L=L, LOSS_AT="eagle", WRITE="lower",
          FLIP=1, STEPS=300, CKPT_EVERY=25, EVAL_EVERY=25)
      for f in FACTORS for L in LS]
todo = [j for j in s1 if not os.path.exists(f"/workspace/eagle_{j['tag']}/history.json")]
print(f"[stage1] {len(todo)}/{len(s1)} to run", flush=True)
run_pool(todo)

# ---- pick durations from head_acc plateau ----
def plateau_ckpts(tag):
    h = json.load(open(f"/workspace/eagle_{tag}/history.json"))
    evs = [e for e in h["evals"] if "head_acc" in e and e["step"] > 0]
    if not evs: return {}
    mx = max(e["head_acc"] for e in evs)
    P = next(e["step"] for e in evs if e["head_acc"] >= 0.95 * mx)
    saved = sorted(int(d[4:]) for d in os.listdir(f"/workspace/eagle_{tag}") if d.startswith("ckpt"))
    pick = lambda t: min(saved, key=lambda s: abs(s - t))
    out = {"halfP": pick(P / 2), "P": pick(P)}
    if out["halfP"] == out["P"]: out.pop("halfP")
    return out

s2 = []
for f in FACTORS:
    for L in LS:
        tag1 = f"s1_{f}_L{L}_flip"
        for dur, ck in plateau_ckpts(tag1).items():
            tag2 = f"s2_{f}_L{L}_{dur}"
            s2.append(job("eagle_stage2.py", tag2, S1_CKPT=f"/workspace/eagle_{tag1}/ckpt{ck}",
                          FACTOR=f, L=L, STEPS=200, ALPHA=4.0, KL_W=1.0, RUN_TAG=tag2))
todo = [j for j in s2 if not os.path.exists(f"/workspace/eagle_{j['tag']}/history.json")]
print(f"[stage2] {len(todo)}/{len(s2)} to run", flush=True)
run_pool(todo)

# ---- baselines ----
bl = [job("eagle_dpo.py", f"fulldpo_{f}_flip", FACTOR=f, L=12, LOSS_AT="final", WRITE="all",
          FLIP=1, STEPS=300) for f in FACTORS]
bl += [job("eagle_dpo.py", f"upperonly_{f}_L{L}_flip", FACTOR=f, L=L, LOSS_AT="final",
           WRITE="upper", FLIP=1, STEPS=300) for f in FACTORS for L in (12, 24)]
todo = [j for j in bl if not os.path.exists(f"/workspace/eagle_{j['tag']}/history.json")]
print(f"[baselines] {len(todo)}/{len(bl)} to run", flush=True)
run_pool(todo)
print("ALL DONE", flush=True)
