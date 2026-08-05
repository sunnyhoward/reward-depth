#!/usr/bin/env python
"""Held-out marker generalisation on the British axis (user idea, 2026-08-05).

THE QUESTION. Does a preference edit install a GENERAL "write British" direction, or a LOOKUP
TABLE of the specific words it was trained on? Split the 298 single-word am|br axes into TRAIN
and HELD-OUT, train on TRAIN only, then measure the British preference on both halves.

    generalisation = (pref_heldout - .5) / (pref_train - .5)

Near 1 = a general direction. Near 0 = memorised vocabulary.

WHY THIS DESIGN BEATS THE REFUSAL VERSION OF THE SAME TEST.
  - Zero leakage by construction. The refusal EN_SELECT/EN_EVAL split was compromised: EN_EVAL
    phrases still appeared in 33.7% of training refusals, so it measured preferential
    reproduction of guaranteed-present phrasing, not seen-vs-unseen. Held-out AXES are never
    seen at all.
  - The oracle is exact. `jumper` vs `sweater` is a token comparison — no judge, no hand-written
    lexicons. Measured 2026-08-05, the refusal lexicons agreed with a Qwen3-8B judge only
    .64-.97 of the time and systematically OVER-read; one whole language (zh) turned out 81-98%
    degenerate and had to be discarded. This metric cannot fail that way.
  - Immune to the head-competence confound. The comparison is ACROSS METHODS AT ONE DEPTH, not
    across depths. Head competence co-varies with L (.152/.202/.380 at L4/L12/L24) and has
    already been shown to track two supposedly-depth results, so every depth claim in this repo
    is currently blocked. This one is not.

It also lands on §16's open question. Stage 1 plateaus at brit_rate .70-.85 and nobody knows
why. Obvious candidate: the residual IS the untrained markers. If held-out axes are exactly
what is missing, the plateau stops being a mystery and becomes a generalisation measurement.

METRIC is reference-corrected implicit preference — (lp_chosen - lp_rejected) under the policy
vs under the frozen base — on the axis's own minimal pairs. Chosen is the British side. Free
sampling is NOT used as the primary meter here: §14's brit oracle read brit_rate ~0 on a genuine
install because held-out words are rare in 128 tokens, and rarity would be confounded with
generalisation. The split is frequency-stratified for the same reason.

Env: ARM=s1|fulldpo|upperonly L=12 STEPS=200 BATCH=16 LR=1e-4 BETA=0.1 SEED=0
     HELDOUT_FRAC=0.4 EVAL_EVERY=10 CKPT_EVERY=50
Out: /workspace/brit_ho_{ARM}/history.json
"""
import os, sys, json, glob, random, collections
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from eagle_common import make_head, comp_slices, gather_logps, MODEL, DEV   # noqa: E402
from helpers import ResidualCapture                                          # noqa: E402

ARM = E("ARM", "s1")
assert ARM in ("s1", "fulldpo", "upperonly")
L = int(E("L", 12))
LOSS_AT, WRITE = ("eagle", "lower") if ARM == "s1" else \
                 ("final", "all" if ARM == "fulldpo" else "upper")
STEPS, BATCH = int(E("STEPS", 200)), int(E("BATCH", 16))
LR, BETA, SEED = float(E("LR", 1e-4)), float(E("BETA", 0.1)), int(E("SEED", 0))
HELDOUT_FRAC = float(E("HELDOUT_FRAC", 0.4))
EVAL_EVERY, CKPT_EVERY = int(E("EVAL_EVERY", 10)), int(E("CKPT_EVERY", 50))
TAG = E("RUN_TAG", f"ho_{ARM}_L{L}")
OUT = f"/workspace/brit_{TAG}"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- data: language component, single-word am|br axes ----
rows = []
for f in (glob.glob("joint-preference-sets/release-v1/**/train.jsonl", recursive=True) +
          glob.glob("joint-preference-sets/release-v1/**/validation.jsonl", recursive=True)):
    rows += [json.loads(l) for l in open(f)]
by_axis = collections.defaultdict(list)
for r in rows:
    if r["component"] != "language":
        continue
    pid = r.get("pair_id", "")
    if ":" not in pid or "|" not in pid:
        continue
    a = pid.split(":", 1)[1]
    if "|" not in a or ":" in a:
        continue
    am, br = a.split("|", 1)
    if am and br and " " not in am and " " not in br:
        by_axis[(am.lower(), br.lower())].append(r)

# FREQUENCY-STRATIFIED SPLIT: sort axes by row count and deal alternately into the two halves.
# A random split would put the rare axes disproportionately on one side, and then "held-out
# performance" would partly be "rare-word performance" — measuring the wrong thing.
axes = sorted(by_axis, key=lambda k: (-len(by_axis[k]), k))
rng = random.Random(SEED)
strata = [axes[i:i + 5] for i in range(0, len(axes), 5)]
held = []
for s in strata:
    rng.shuffle(s)
    held += s[:max(1, int(round(HELDOUT_FRAC * len(s))))]
HELD = set(held)
TRAIN_AX = [a for a in axes if a not in HELD]
HELD_AX = [a for a in axes if a in HELD]
train_rows = [r for a in TRAIN_AX for r in by_axis[a]]
print(f"[data] {len(axes)} axes -> train {len(TRAIN_AX)} / heldout {len(HELD_AX)} | "
      f"train rows {len(train_rows)}", flush=True)
json.dump({"train_axes": [list(a) for a in TRAIN_AX], "heldout_axes": [list(a) for a in HELD_AX]},
          open(f"{OUT}/axis_split.json", "w"), indent=1)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL, HID = len(model.model.layers), model.config.hidden_size

layers_map = dict(lower=list(range(0, L + 1)), upper=list(range(L + 1, NL)), all=None)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=layers_map[WRITE])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
BLOCKS = list(model.model.layers)
allowed = set(layers_map[WRITE] if layers_map[WRITE] is not None else range(NL))
for n_, p in policy.named_parameters():
    if p.requires_grad:
        assert "lora" in n_ and int(n_.split(".layers.")[1].split(".")[0]) in allowed, n_
params = [p for p in policy.parameters() if p.requires_grad]

head = head_ref = None
if LOSS_AT == "eagle":
    hp = f"/workspace/eagle_head_tf_L{L}.pt"
    assert os.path.exists(hp), f"missing {hp}"
    sd = torch.load(hp, map_location=DEV)
    head = make_head(HID, "tf").to(DEV); head.load_state_dict(sd)
    head_ref = make_head(HID, "tf").to(DEV); head_ref.load_state_dict(sd)
    for p in list(head.parameters()) + list(head_ref.parameters()):
        p.requires_grad_(False)          # frozen (§8)
opt = torch.optim.AdamW(params, lr=LR)
print(f"[{TAG}] {MODEL} trainable {sum(p.numel() for p in params)/1e6:.1f}M | "
      f"arm={ARM} loss_at={LOSS_AT} write={WRITE} L={L}", flush=True)


def pair_logps(items, grad, use_ref):
    texts, plens = [], []
    for r in items:
        pr = r["prompt"]
        pl = len(tok(pr).input_ids)
        texts += [pr + r["chosen"], pr + r["rejected"]]
        plens += [pl, pl]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if LOSS_AT == "final":
            lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        else:
            with ResidualCapture([BLOCKS[L]]) as cap:
                policy(**enc)
            hd = head_ref if use_ref else head
            lsm = F.log_softmax(hd(cap.get()[0][:, :-1], model), -1)
        lp = gather_logps(lsm, enc, spans)
    return lp[0::2], lp[1::2]


@torch.no_grad()
def pref_rate(axis_list, n_ax=None):
    """Reference-corrected British preference, at the FINAL logits (behavioural readout).

    Read at final rather than through the head on purpose: the head is the training interface,
    and §19 showed a margin can exist at one readout and not another. Generalisation has to be
    claimed at the output, not at the surrogate.
    """
    items = [r for a in (axis_list[:n_ax] if n_ax else axis_list) for r in by_axis[a]]
    hits = []
    for s in range(0, len(items), 16):
        sub = items[s:s + 16]
        texts, plens = [], []
        for r in sub:
            pl = len(tok(r["prompt"]).input_ids)
            texts += [r["prompt"] + r["chosen"], r["prompt"] + r["rejected"]]
            plens += [pl, pl]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        spans = comp_slices(tok, texts, plens, enc)
        lp = gather_logps(F.log_softmax(policy(**enc).logits[:, :-1].float(), -1), enc, spans)
        with policy.disable_adapter():
            rp = gather_logps(F.log_softmax(policy(**enc).logits[:, :-1].float(), -1), enc, spans)
        d = (lp - rp).view(-1, 2)
        hits += (d[:, 0] > d[:, 1]).float().cpu().tolist()
    return float(np.mean(hits)), len(items)


def full_eval(step):
    policy.eval()
    pt, nt = pref_rate(TRAIN_AX, n_ax=60)
    ph, nh = pref_rate(HELD_AX)
    gen = (ph - 0.5) / (pt - 0.5) if abs(pt - 0.5) > 1e-6 else float("nan")
    policy.train()
    return dict(step=step, pref_train=pt, pref_heldout=ph, generalisation=gen,
                n_train=nt, n_heldout=nh)


hist = dict(tag=TAG, arm=ARM, model=MODEL, L=L, loss_at=LOSS_AT, write=WRITE,
            n_train_axes=len(TRAIN_AX), n_heldout_axes=len(HELD_AX), loss=[], evals=[])
ev0 = full_eval(0); hist["evals"].append(ev0)
print(f"  step   0: {ev0}", flush=True)

rgen = random.Random(SEED + 99)
policy.train()
for step in range(STEPS):
    batch = rgen.sample(train_rows, BATCH)
    opt.zero_grad()
    la, lb = pair_logps(batch, True, use_ref=False)
    with torch.no_grad(), policy.disable_adapter():
        ra, rb = pair_logps(batch, False, use_ref=True)
    loss = -F.logsigmoid(BETA * ((la - ra) - (lb - rb))).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach()))
    if (step + 1) % EVAL_EVERY == 0:
        ev = full_eval(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:3d}: loss {np.mean(hist['loss'][-EVAL_EVERY:]):.4f}  "
              f"train {ev['pref_train']:.3f}  heldout {ev['pref_heldout']:.3f}  "
              f"gen {ev['generalisation']:.2f}", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        policy.save_pretrained(f"{OUT}/ckpt{step+1}")

json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
