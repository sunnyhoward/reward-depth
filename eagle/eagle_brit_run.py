#!/usr/bin/env python
"""Sequential single-slot driver for the brit axis (runs as a 3rd GPU slot alongside the styc
matrix): s1 {lang,culture} x L {4,12,24} -> plateau-pick -> s2 -> full-DPO baselines."""
import os, sys, json, subprocess

ROOT = "/workspace/reward-depth"
ENVB = dict(os.environ, HF_HOME="/workspace/.hf_home",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
LS = [4, 12, 24]

def run(tag, **env):
    out = f"/workspace/eagle_brit_{tag}/history.json"
    if os.path.exists(out):
        print(f"[skip] {tag}", flush=True); return
    e = dict(ENVB); e.update({k: str(v) for k, v in env.items()}); e["RUN_TAG"] = tag
    print(f"[run] {tag}", flush=True)
    r = subprocess.run([sys.executable, "eagle/eagle_brit.py"], env=e, cwd=ROOT,
                       stdout=open(f"/workspace/eagle_brit_{tag}.log", "w"), stderr=subprocess.STDOUT)
    print(f"[{'ok' if r.returncode == 0 and os.path.exists(out) else 'FAIL'}] {tag}", flush=True)

for f in ("lang", "culture"):
    for L in LS:
        run(f"s1_{f}_L{L}", STAGE="s1", FACTOR=f, L=L, STEPS=200)
for f in ("lang", "culture"):
    for L in LS:
        tag1 = f"s1_{f}_L{L}"
        try:
            h = json.load(open(f"/workspace/eagle_brit_{tag1}/history.json"))
            evs = [e for e in h["evals"] if "head_acc" in e and e["step"] > 0]
            mx = max(e["head_acc"] for e in evs)
            P = next(e["step"] for e in evs if e["head_acc"] >= 0.95 * mx)
            saved = sorted(int(d[4:]) for d in os.listdir(f"/workspace/eagle_brit_{tag1}") if d.startswith("ckpt"))
            ck = min(saved, key=lambda s: abs(s - P))
        except Exception as ex:
            print(f"[FAIL] plateau {tag1}: {ex}", flush=True); continue
        run(f"s2_{f}_L{L}", STAGE="s2", FACTOR=f, L=L, STEPS=150,
            S1_CKPT=f"/workspace/eagle_brit_{tag1}/ckpt{ck}")
for f in ("lang", "culture"):
    run(f"fulldpo_{f}", STAGE="s1", FACTOR=f, L=12, LOSS_AT="final", WRITE="all", STEPS=200)
print("BRIT ALL DONE", flush=True)
