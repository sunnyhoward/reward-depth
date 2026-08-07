#!/usr/bin/env python
"""His DPO-P settings sheet for Qwen3.5-2B Britishness, implemented verbatim (2026-08-07).

This is NOT sup_train.py. sup_train.py is the EAGLE-readout reading of his prose ("contrastive
loss, K-FAC-EWC loss, general replay loss, weights 1:3:1", LoRA on layers 0..17, DPO through the
head at L17). The sheet he sent describes something concretely different and fully specified:

    full-model DPO-Positive, MLP-only LoRA r=8 on ALL 24 layers, fp32, tf32 off,
    beta 0.1, dpop_lambda 50, lr 1e-4, batch 8, 300 steps, adapter-off reference,
    component-balanced batch scheduler with a spelling-contrast quota.

Nothing in the sheet mentions the EAGLE head, K-FAC, or replay. So this file reproduces the
sheet and only the sheet — every number below is his, and where the sheet is silent about a
mapping onto this dataset the choice is marked JUDGEMENT.

Env: STEPS=300 BS=8 LR=1e-4 BETA=0.1 LAMBDA=50 SEED=0 SCHED_SEED=4242
Out: /workspace/sup_dpop/
"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import MODEL, DEV, load_split, pair_texts, encode, span_mask, gather_logps  # noqa

STEPS = int(E("STEPS", 300))
BS = int(E("BS", 8))
LR, BETA, LAM = float(E("LR", 1e-4)), float(E("BETA", 0.1)), float(E("LAMBDA", 50.0))
SEED, SCHED_SEED = int(E("SEED", 0)), int(E("SCHED_SEED", 4242))
EVAL_BS = int(E("EVAL_BS", 8))
EVAL_EVERY = int(E("EVAL_EVERY", 25))
CKPTS = {int(x) for x in E("CKPTS", "25,50,75,100,200,300").split(",")}
OUT = E("RUN_TAG_DIR", "/workspace/sup_dpop")
os.makedirs(OUT, exist_ok=True)

# "dtype float32 — NOT bf16; tf32 False, both matmul and cudnn". He wrote this down explicitly,
# which reads as someone who found it load-bearing, so it is not negotiable here.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV)
model.config.use_cache = False

cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["gate_proj", "up_proj", "down_proj"])   # MLP only, all 24 layers
policy = get_peft_model(model, cfg)
# Not his; a memory accommodation. fp32 + batch 8 pairs on a GPU already carrying two other jobs
# OOMs in the linear-attention chunk kernel. Checkpointing recomputes activations instead of
# storing them — the arithmetic is unchanged, so the sheet's dtype and batch size stay exactly as
# specified rather than being quietly reduced.
if int(E("GRAD_CKPT", 1)):
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)     # torch defaults: betas .9/.999, wd 0

# ---------------- component-balanced batch scheduler ----------------
# "components culture, language, style, truth — 4, equal mass -> 2 rows each per batch of 8".
# JUDGEMENT: the release ships seven `family` values, not four, so they are folded onto his four.
# culture/style/truth are one-to-one; everything lexical becomes `language`.
COMPONENT = {"culture": "culture", "style": "style", "truth_dialect": "truth",
             "lexicon": "language", "expression": "language", "false_friend": "language",
             "spelling_control": "language"}
COMPONENTS = ["culture", "language", "style", "truth"]
# "quota_key=spelling_contrast, quota_max=1  # <=1 -ise/-ize row per batch".
# JUDGEMENT: read onto the release's `group` field as the -ise/-ize groups.
QUOTA_GROUPS = {"spell_ise", "ize_ise"}
QUOTA_MAX = int(E("QUOTA_MAX", 1))

train_rows = load_split("train")
val_rows = load_split("validation")
by_comp = {c: [] for c in COMPONENTS}
for r in train_rows:
    by_comp[COMPONENT[r["family"]]].append(r)
assert BS % len(COMPONENTS) == 0, "batch_size must be a multiple of the component count"
PER = BS // len(COMPONENTS)
sched = random.Random(SCHED_SEED)


def draw_batch():
    """Equal component mass, with replacement. A capped draw is REPLACED from the same
    component (his note: "capped draws are replaced from the same component, not dropped, so
    component mass stays exactly equal")."""
    out, nq = [], 0
    for c in COMPONENTS:
        pool = by_comp[c]
        got = 0
        while got < PER:
            r = pool[sched.randrange(len(pool))]
            if r["group"] in QUOTA_GROUPS:
                if nq >= QUOTA_MAX:
                    continue                  # redraw from the SAME component
                nq += 1
            out.append(r); got += 1
    return out


def full_pair_logps(rows, grad):
    """Summed log-prob over the completion tokens, chosen and rejected, at the FULL output."""
    trip = pair_texts(tok, rows)
    texts = [t for c, j, _ in trip for t in (c, j)]
    plens = [pl for _, _, pl in trip for _ in (0, 1)]
    enc = encode(tok, texts, max_length=256).to(DEV)
    m = span_mask(tok, texts, plens, enc)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        lg = policy(**enc).logits[:, :-1].float()
        # `log_softmax` then `gather` would hold a second (B, T, 248320) fp32 tensor — 2 GB at
        # this batch, on a GPU already carrying two other jobs. gather-minus-logsumexp is the
        # same number without the copy.
        tgt = enc.input_ids[:, 1:]
        lp = (lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1))
        lp = (lp * m).sum(-1)
    return lp[0::2], lp[1::2]


def dpop_loss(rows):
    chosen, rejected = full_pair_logps(rows, True)
    with torch.no_grad(), policy.disable_adapter():
        ref_chosen, ref_rejected = full_pair_logps(rows, False)
    margin = (chosen - ref_chosen) - (rejected - ref_rejected)
    margin = margin - LAM * F.relu(ref_chosen - chosen)          # the DPO-P term
    loss = -F.logsigmoid(BETA * margin).mean()
    return loss, dict(margin=float(margin.mean()),
                      acc=float(((chosen - ref_chosen) > (rejected - ref_rejected)).float().mean()),
                      pos=float(F.relu(ref_chosen - chosen).mean()))


@torch.no_grad()
def evaluate(step):
    """THREE numbers, because they are three different claims.

      acc      raw ranking, logp(chosen) > logp(rejected). This is the one his 730/735 and the
               517/735 preamble baseline have to be, since a preamble has no reference policy to
               be relative to. It is the comparable number.
      acc_rel  reference-relative, (c - rc) > (j - rj) — what the DPO-P margin actually optimises.
               Exactly 0 at step 0 by construction (zero-init B ⇒ policy == reference), so it
               measures movement, not skill.
      acc_base the frozen model's own raw ranking. Constant across steps; the floor `acc` has to
               beat for any of this to mean anything.
    """
    policy.eval()

    def acc_over(rows):
        raw, rel, base = [], [], []
        for s in range(0, len(rows), EVAL_BS):
            b = rows[s:s + EVAL_BS]
            c, j = full_pair_logps(b, False)
            with policy.disable_adapter():
                rc, rj = full_pair_logps(b, False)
            raw += (c > j).float().cpu().tolist()
            rel += ((c - rc) > (j - rj)).float().cpu().tolist()
            base += (rc > rj).float().cpu().tolist()
        return dict(acc=float(np.mean(raw)), acc_rel=float(np.mean(rel)),
                    acc_base=float(np.mean(base)), n=len(raw))

    out = dict(step=step, **acc_over(val_rows))
    # All 200 truth_guard rows sit in TRAIN, so the held-out number never has to choose truth over
    # dialect. Report the guard separately and label it in-sample rather than let a guard-free
    # score stand in for the whole preference.
    g = [r for r in train_rows if r["role"] == "truth_guard"]
    if g:
        gg = acc_over(g)
        out.update(guard_insample=gg["acc"], guard_base=gg["acc_base"])
    policy.train()
    return out


hist = dict(sheet="dpop", model=MODEL, beta=BETA, dpop_lambda=LAM, lr=LR, bs=BS, steps=STEPS,
            lora=dict(r=8, alpha=16, targets=["gate_proj", "up_proj", "down_proj"]),
            dtype="float32", tf32=False, seed=SEED, schedule_seed=SCHED_SEED,
            loss=[], parts=[], evals=[])
print(f"[dpop] {MODEL} fp32 | LoRA r8 MLP all-layers, trainable "
      f"{sum(p.numel() for p in params)/1e6:.2f}M | beta {BETA} lambda {LAM} lr {LR} bs {BS} "
      f"| train {len(train_rows)} val {len(val_rows)}", flush=True)
print("  component pools: " + ", ".join(f"{c} {len(by_comp[c])}" for c in COMPONENTS), flush=True)

ev = evaluate(0); hist["evals"].append(ev)
print(f"  step   0: {ev}", flush=True)
policy.train()

for step in range(STEPS):
    rows = draw_batch()
    opt.zero_grad()
    loss, parts = dpop_loss(rows)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach())); hist["parts"].append(parts)
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f} "
              f"(margin {parts['margin']:+.3f} pos {parts['pos']:.3f} "
              f"train_acc {np.mean([p['acc'] for p in hist['parts'][-10:]]):.3f})", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: {ev}", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) in CKPTS:
        policy.save_pretrained(f"{OUT}/ckpt{step+1}")

json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
