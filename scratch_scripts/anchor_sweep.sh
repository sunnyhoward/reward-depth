#!/bin/bash
# Stage-1 KL-to-base anchor, styc L24 — the regulariser stage 2 always had and stage 1 never did.
# Task-text anchor, NOT replay: measured 2026-08-05, the L24 edit moves the model 2.7 nats on task
# text and 0.003 nats on replay, so a replay anchor (and the K-FAC factors estimated on the same
# replay) has nothing to push against. The damage is on-distribution.
# W=1 holds KL to 0.13 but blocks the install; W=0 gives KL 2.73 and correctness .11. Sweep between.
# BAR TO CLEAR: lambda=0 at LR 1e-5 held terse 1.000 AND correct 1.000 across four consecutive
# checkpoints. K-FAC failed this bar; an anchor at LR 1e-4 has to match it to be worth anything.
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
for W in 0.1 0.2 0.3 0.5; do
  FACTOR=style L=24 LOSS_AT=eagle WRITE=lower HEAD_ARCH=tf FREEZE_HEAD=1 \
    ANCHOR_W=$W ANCHOR_SRC=task STEPS=50 EVAL_EVERY=5 CKPT_EVERY=50 \
    RUN_TAG=anchor_w$W python eagle/eagle_dpo.py > /workspace/kfac/anchor_w$W.log 2>&1
  echo "[anchor] W=$W done"
done
python - <<'PY'
import json
print("\n=== stage-1 task anchor, styc L24, LR 1e-4 (bar: LR 1e-5 control held 1.000/1.000 x4) ===")
print(" W    | step | terse | correct |   KL")
for W in ["0","0.1","0.2","0.3","0.5"]:
    tag = "kfac_l24_lam0" if W=="0" else f"anchor_w{W}"
    try: h=json.load(open(f"/workspace/eagle_{tag}/history.json"))
    except Exception: print(f" {W:4s} | MISSING"); continue
    for e in h["evals"]:
        if e["step"]==0 or e["step"]%10: continue
        print(f" {W:4s} | {e['step']:4d} | {1-e['gen_explained']:5.3f} | {e['gen_correct']:7.3f} | {e['kl_from_base']:5.2f}")
    print()
PY
