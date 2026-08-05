#!/usr/bin/env python
"""Refusal depth ladder + baselines (2026-08-05). Driver for refusal_dpo.py.

THE QUESTION, as reframed after the base measurement. Qwen3-4B-Base already refuses in every
eval language (en .375, zh .531, ar .359, it .469, vi .422, ko .203), so "does refusal transfer"
is pre-answered — a language-general refusal representation already exists. §18's law says
stage-1 is a correlation AMPLIFIER, which needs exactly such a seed. So the live question is:

    does an ENGLISH-ONLY, depth-restricted edit AMPLIFY the existing language-general refusal
    direction (-> non-English rates rise), or build a PARALLEL ENGLISH-SPECIFIC one
    (-> English rises alone)?

Prediction (inverted U, not "early beats late"): early layers are lexical and language-specific,
the shared conceptual space is mid-stack, late layers re-specialise to output language. So expect
weak transfer at L4, a peak near L12, decline at L24, and full DPO worst.

ARMS: stage-1 at L in {4,12,24}; full DPO (WRITE=all); upper-only (WRITE=upper, L=12).

ANALYSIS CONTRACT — three things, none optional:
  1. MATCHED ENGLISH REFUSAL. Stage-1 moves the model far less than full DPO, and a method that
     perturbs less could preserve multilingual ability for reasons unrelated to write depth.
     CKPT_EVERY=10 gives 20 checkpoints per arm so arms can be compared at equal English rate.
  2. DELTA vs the base row, per language. Never absolute rates.
  3. EN_SELECT vs EN_EVAL. Disjoint lexicons; SELECT rising while EVAL stays flat is a lexical
     install (§1's brit-culture trap) and voids any transfer claim from that arm.

KL is the stopping signal, not step count: today's styc sweep found install at KL ~0.7 and damage
at KL ~2.4, invariant across 7 arms spanning two learning rates and five K-FAC coefficients.

Env: STEPS=200 SLOTS=3 SEED=0 LS=4,12,24
Out: /workspace/refusal_{tag}/history.json per arm
"""
import os, sys, json, subprocess, time

ROOT = "/workspace/reward-depth"
E = os.environ.get
LS = [int(x) for x in E("LS", "4,12,24").split(",")]
STEPS, SLOTS, SEED = int(E("STEPS", 200)), int(E("SLOTS", 3)), int(E("SEED", 0))
ENVB = dict(os.environ, HF_HOME=E("HF_HOME", "/workspace/.hf_home"),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def job(tag, **env):
    e = dict(ENVB)
    e.update({k: str(v) for k, v in dict(
        STEPS=STEPS, SEED=SEED, EVAL_EVERY=10, CKPT_EVERY=10, RUN_TAG=tag, **env).items()})
    return dict(tag=tag, env=e, log=f"/workspace/refusal_{tag}.log",
                cmd=[sys.executable, "refusal/refusal_dpo.py"])


ARMS = ([job(f"s1_L{L}", L=L, LOSS_AT="eagle", WRITE="lower") for L in LS] +
        [job("fulldpo", L=12, LOSS_AT="final", WRITE="all"),
         job("upperonly_L12", L=12, LOSS_AT="final", WRITE="upper")])

todo = [j for j in ARMS if not os.path.exists(f"/workspace/refusal_{j['tag']}/history.json")]
print(f"[ladder] {len(todo)}/{len(ARMS)} arms to run (steps={STEPS} slots={SLOTS})", flush=True)

live = []
for j in todo:
    while len(live) >= SLOTS:
        time.sleep(20)
        live = [(p, jj) for p, jj in live if p.poll() is None]
    print(f"[launch] {j['tag']}", flush=True)
    live.append((subprocess.Popen(j["cmd"], env=j["env"], cwd=ROOT,
                                  stdout=open(j["log"], "w"), stderr=subprocess.STDOUT), j))
for p, _ in live:
    p.wait()

print("\n=== English meters per arm (peak EN_EVAL refusal, and the KL there) ===")
print(" arm             | base .375 -> peak | @step | KL    | select_lex | note")
rows = {}
for j in ARMS:
    hp = f"/workspace/refusal_{j['tag']}/history.json"
    if not os.path.exists(hp):
        print(f" {j['tag']:15s} | MISSING — check {j['log']}")
        continue
    h = json.load(open(hp))
    evs = [e for e in h["evals"] if e["step"] > 0]
    if not evs:
        continue
    b = max(evs, key=lambda e: e["refusal_eval_lex"])
    lex_gap = b["refusal_select_lex"] - b["refusal_eval_lex"]
    note = "LEXICAL?" if lex_gap > 0.25 else ""
    rows[j["tag"]] = b
    print(f" {j['tag']:15s} | {b['refusal_eval_lex']:17.3f} | {b['step']:5d} | "
          f"{b['kl_from_base']:5.2f} | {b['refusal_select_lex']:10.3f} | {note}")

json.dump(rows, open("/workspace/refusal/ladder_english.json", "w"), indent=1)
print("\nNEXT: refusal_eval.py per selected checkpoint (matched English rate), then the "
      "per-language delta table vs eval_base.json.")
