#!/usr/bin/env python
"""KL_W sweep on style-L12 — does a STRONGER anchor separate the install from the damage?

Motivation (user question, 2026-08-04): every stage-2 run in the first pass used KL_W=1.0
(verified across all 19 histories). RESULTS.md §3's claim "the anchor cannot save a poisoned
target" is therefore a single-anchor-weight result. The alpha sweep (§7) argues against the
anchor helping — to first order the anchor is the SAME knob as alpha:

    student ~ base + [ (init - base) + alpha*Delta ] / (1 + KL_W)

i.e. KL_W shrinks the delta AND the stage-1 install uniformly. But alpha-insensitivity is
itself unexplained under that model, so the theory is not trustworthy enough to skip the cell.

Success = gen_correct recovers toward 1.0 while terse stays high. Failure (predicted) = a
monotone trade: both fall together as KL_W rises, no window where install survives clean.

Chain (all artifacts died with the box):
  1. eagle_head.py  LAYERS=12                  -> /workspace/eagle_head_L12.pt
  2. eagle_dpo.py   stage-1 style L12 FLIP     -> /workspace/eagle_s1_style_L12_flip/ckpt*
  3. plateau pick   (same rule as eagle_run_all.py: first step >= 95% of max head_acc)
  4. eagle_stage2.py x KL_W {1,4,16,64} @ ALPHA=4.0, 2-slot pool

KL_W=1.0 is retained as a REPRODUCTION check of s2_style_L12_P (terse .86 / gen_correct .05)
under the rebuilt environment (transformers 5.x) — if it does not reproduce, the sweep is not
comparable to the first pass and that is the first thing to know.
"""
import os, sys, json, subprocess, time

ROOT = "/workspace/reward-depth"
ENVB = dict(os.environ, HF_HOME="/workspace/.hf_home",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
L = 12
FACTOR = "style"
KLWS = [1.0, 4.0, 16.0, 64.0]
S1_TAG = f"s1_{FACTOR}_L{L}_flip"


def sh(script, tag, **env):
    e = dict(ENVB); e.update({k: str(v) for k, v in env.items()})
    log = f"/workspace/eagle_{tag}.log"
    return dict(cmd=[sys.executable, f"eagle/{script}"], env=e, log=log, tag=tag)


def run_pool(jobs, slots=2):
    live = []
    for j in jobs:
        while len(live) >= slots:
            time.sleep(20)
            live = [(p, jj) for p, jj in live if p.poll() is None]
        print(f"[launch] {j['tag']} -> {j['log']}", flush=True)
        p = subprocess.Popen(j["cmd"], env=j["env"], cwd=ROOT,
                             stdout=open(j["log"], "w"), stderr=subprocess.STDOUT)
        live.append((p, j))
    for p, j in live: p.wait()
    bad = []
    for j in jobs:
        ok = os.path.exists(f"/workspace/eagle_{j['tag']}/history.json")
        print(f"[done] {j['tag']}  {'ok' if ok else 'MISSING HISTORY — check ' + j['log']}", flush=True)
        if not ok: bad.append(j["tag"])
    return bad


# ---- 1. head pretrain (L12 only) ----
if not os.path.exists(f"/workspace/eagle_head_L{L}.pt"):
    print(f"[head] pretraining L{L}", flush=True)
    e = dict(ENVB); e["LAYERS"] = str(L)
    log = f"/workspace/eagle_head_L{L}.log"
    p = subprocess.Popen([sys.executable, "eagle/eagle_head.py"], env=e, cwd=ROOT,
                         stdout=open(log, "w"), stderr=subprocess.STDOUT)
    p.wait()
    assert os.path.exists(f"/workspace/eagle_head_L{L}.pt"), f"head pretrain failed — see {log}"
    print("[head] ok", flush=True)
else:
    print(f"[head] /workspace/eagle_head_L{L}.pt exists — skipping", flush=True)

# ---- 2. stage 1 ----
if not os.path.exists(f"/workspace/eagle_{S1_TAG}/history.json"):
    bad = run_pool([sh("eagle_dpo.py", S1_TAG, FACTOR=FACTOR, L=L, LOSS_AT="eagle",
                       WRITE="lower", FLIP=1, STEPS=300, CKPT_EVERY=25, EVAL_EVERY=25)], slots=1)
    assert not bad, "stage-1 failed"
else:
    print(f"[stage1] {S1_TAG} exists — skipping", flush=True)

# ---- 3. plateau ckpt (same rule as the first pass) ----
h = json.load(open(f"/workspace/eagle_{S1_TAG}/history.json"))
evs = [e for e in h["evals"] if "head_acc" in e and e["step"] > 0]
mx = max(e["head_acc"] for e in evs)
P = next(e["step"] for e in evs if e["head_acc"] >= 0.95 * mx)
saved = sorted(int(d[4:]) for d in os.listdir(f"/workspace/eagle_{S1_TAG}") if d.startswith("ckpt"))
CK = min(saved, key=lambda s: abs(s - P))
print(f"[plateau] max head_acc {mx:.3f}, P={P}, using ckpt{CK}", flush=True)

# ---- 4. KL_W sweep ----
jobs = [sh("eagle_stage2.py", f"s2_{FACTOR}_L{L}_klw{w:g}",
           S1_CKPT=f"/workspace/eagle_{S1_TAG}/ckpt{CK}", FACTOR=FACTOR, L=L,
           ALPHA=4.0, KL_W=w, STEPS=200, EVAL_EVERY=25, CKPT_EVERY=100,
           RUN_TAG=f"s2_{FACTOR}_L{L}_klw{w:g}")
        for w in KLWS]
jobs = [j for j in jobs if not os.path.exists(f"/workspace/eagle_{j['tag']}/history.json")]
print(f"[sweep] {len(jobs)} arms to run", flush=True)
run_pool(jobs, slots=2)

# ---- 5. collect ----
rows = []
for w in KLWS:
    f = f"/workspace/eagle_s2_{FACTOR}_L{L}_klw{w:g}/history.json"
    if not os.path.exists(f): continue
    hh = json.load(open(f))
    last = hh["evals"][-1]
    rows.append(dict(kl_w=w, step=last.get("step"),
                     terse=1.0 - last.get("gen_explained", float("nan")),
                     gen_explained=last.get("gen_explained"),
                     gen_correct=last.get("gen_correct"),
                     gen_other=last.get("gen_other"),
                     gen_len_words=last.get("gen_len_words"),
                     kl_from_base=last.get("kl_from_base"),
                     acc_style_c=last.get("acc_style_c"),
                     acc_corr_e=last.get("acc_corr_e")))
out = f"{ROOT}/results/runs/eagle/eagle_klw_sweep.json"
json.dump(dict(factor=FACTOR, L=L, alpha=4.0, s1_ckpt=CK, plateau=P, rows=rows),
          open(out, "w"), indent=1)
print("\n KL_W | terse | gen_correct | KL_base")
for r in rows:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    print(f" {r['kl_w']:5g} | {fmt(r['terse'])} | {fmt(r['gen_correct'])} | {fmt(r['kl_from_base'])}")
print(f"\nwrote {out}\nALL DONE", flush=True)
