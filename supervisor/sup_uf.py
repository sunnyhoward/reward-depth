#!/usr/bin/env python
"""UltraFeedback -> the britishness release schema, so the supervisor recipe runs on it unchanged.

WHY A MATERIALISED FILE AND NOT A LOADER. Two consumers have to see byte-identical rows: the
decodability sweep that picks the attach layer L*, and sup_train.py that trains through a head at
that layer. If they disagree by even a truncation rule, L* stops describing the material the
trainer scores and the whole "attach at the elbow" premise is unfounded. So this writes ONE jsonl
in his release's schema; `sup_common.load_split` reads it via SUP_BRIT, and dec_data's `uf_sup`
loader reads the same file.

THE FILTERS, and which are the repo's vs new here:
  - repo's UF filters (uf/uf_probe_rl.py:87-98, dec_data.load_uf): both sides present, non-identical,
    GPT-4 score margin >= 1.0. The margin filter is not cosmetic — phase-7 §8 found 13.6% of UF
    soft labels side against the dataset, concentrated in low-margin pairs.
  - NEW, and it is a real sample restriction that must be reported: both rendered sides must fit
    in MAX_LEN tokens, and the rendered prompt in MAX_PLEN. sup_common.encode right-truncates at a
    fixed max_length with the prompt at the front, so a pair that overruns would have its
    completion silently cut mid-answer on one side and not the other — a length artefact injected
    into the very margin being trained. Filtering keeps span_mask exact and leaves the shared
    britishness path untouched. It biases the sample SHORT: `length_stats` in the manifest records
    what the filter removed, and the sweep's length floor says how much of the signal that leaves.

Ties to his rendering: single user turn, `enable_thinking=False`, which on Qwen3.5 emits the empty
`<think>\\n\\n</think>` block. Verified byte-identical in form to `britishness/release` rows —
prompt head ends at `<|im_start|>assistant\\n`, completions end `<|im_end|>\\n`.

Usage: python sup_uf.py            (writes /workspace/reward-depth/supervisor/uf_release/uf.jsonl)
Env:   UF_N_TOTAL=6000 UF_N_EVAL=750 UF_MIN_MARGIN=1.0 MAX_LEN=512 MAX_PLEN=192
       SUP_MODEL=Qwen/Qwen3.5-2B UF_DATASET=allenai/ultrafeedback_binarized_cleaned
"""
import hashlib
import json
import os
import sys

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL = E("SUP_MODEL", "Qwen/Qwen3.5-2B")
N_TOTAL = int(E("UF_N_TOTAL", 6000))
N_EVAL = int(E("UF_N_EVAL", 750))
MIN_MARGIN = float(E("UF_MIN_MARGIN", 1.0))
MAX_LEN = int(E("MAX_LEN", 512))
MAX_PLEN = int(E("MAX_PLEN", 192))
DATASET = E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned")
SPLIT = E("UF_SPLIT", "train_prefs")
OUT_DIR = E("UF_RELEASE", f"{HERE}/uf_release")
OUT = f"{OUT_DIR}/uf.jsonl"
ASSIST_TAG = "<|im_start|>assistant\n"


def render(tok, prompt, completion):
    """(text_prompt, text_full) in his release's byte format."""
    head = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, enable_thinking=False,
                                   tokenize=False)
    return head, head + completion.strip() + "<|im_end|>\n"


def main():
    from datasets import load_dataset
    from itertools import islice
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)

    ds = load_dataset(DATASET, split=SPLIT, streaming=True)
    rows, seen, drop = [], set(), dict(missing=0, identical=0, margin=0, plen=0, len=0, dup=0)
    kept_len, all_len = [], []
    for ex in islice(ds, N_TOTAL * 20):
        ch, rj = ex.get("chosen"), ex.get("rejected")
        if not ch or not rj:
            drop["missing"] += 1
            continue
        p = ex.get("prompt") or ch[0]["content"]
        c, r = ch[-1]["content"], rj[-1]["content"]
        if not (p and c and r):
            drop["missing"] += 1
            continue
        if c == r:
            drop["identical"] += 1
            continue
        sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
        if sc is None or sr is None or float(sc) - float(sr) < MIN_MARGIN:
            drop["margin"] += 1
            continue
        if p in seen:
            drop["dup"] += 1
            continue
        head, t_c = render(tok, p, c)
        _, t_r = render(tok, p, r)
        n_p, n_c, n_r = ntok(head), ntok(t_c), ntok(t_r)
        all_len.append(max(n_c, n_r))
        if n_p > MAX_PLEN:
            drop["plen"] += 1
            continue
        if max(n_c, n_r) > MAX_LEN:
            drop["len"] += 1
            continue
        seen.add(p)
        kept_len.append(max(n_c, n_r))
        rows.append(dict(prompt=p, chosen=c.strip(), rejected=r.strip(),
                         text_prompt=head, text_chosen=t_c, text_rejected=t_r,
                         score_chosen=float(sc), score_rejected=float(sr),
                         ntok_prompt=n_p, ntok_chosen=n_c - n_p, ntok_rejected=n_r - n_p))
        if len(rows) >= N_TOTAL:
            break
    del ds

    if len(rows) < N_TOTAL:
        print(f"[uf] WARNING: only {len(rows)} of {N_TOTAL} requested pairs passed the filters",
              flush=True)

    # Held-out by PROMPT hash, deterministic: the same prompt lands on the same side on every
    # rerun and for every model, so a re-materialised file cannot quietly move the eval set.
    def h(s):
        return int(hashlib.sha1(f"uf_sup|{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    rows.sort(key=lambda r: h(r["prompt"]))
    n_eval = min(N_EVAL, len(rows) // 4)
    out = []
    for i, r in enumerate(rows):
        ev = i < n_eval
        out.append(dict(
            id=f"uf-{i:05d}", family="quality", group=hashlib.sha1(r["prompt"].encode()).hexdigest()[:12],
            item="", form="qa", origin=f"{DATASET}:{SPLIT}", role="pref",
            reserved_for_eval=bool(ev), prompt=r["prompt"], chosen=r["chosen"], rejected=r["rejected"],
            messages_chosen=[{"role": "user", "content": r["prompt"]},
                             {"role": "assistant", "content": r["chosen"]}],
            messages_rejected=[{"role": "user", "content": r["prompt"]},
                               {"role": "assistant", "content": r["rejected"]}],
            text_prompt=r["text_prompt"], text_chosen=r["text_chosen"], text_rejected=r["text_rejected"],
            meta=dict(score_chosen=r["score_chosen"], score_rejected=r["score_rejected"],
                      ntok_prompt=r["ntok_prompt"], ntok_chosen=r["ntok_chosen"],
                      ntok_rejected=r["ntok_rejected"])))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    n_longer = sum(1 for r in out if r["meta"]["ntok_chosen"] > r["meta"]["ntok_rejected"])
    man = dict(schema="britishness-release-compatible", source=DATASET, split=SPLIT,
               model_template=MODEL, n=len(out), n_eval=n_eval, n_train=len(out) - n_eval,
               filters=dict(min_margin=MIN_MARGIN, max_len=MAX_LEN, max_plen=MAX_PLEN,
                            dedup_prompt=True),
               dropped=drop,
               length_stats=dict(kept_mean=sum(kept_len) / max(len(kept_len), 1),
                                 all_mean=sum(all_len) / max(len(all_len), 1),
                                 kept_frac_of_scored=len(kept_len) / max(len(all_len), 1),
                                 chosen_longer_frac=n_longer / max(len(out), 1)))
    json.dump(man, open(f"{OUT_DIR}/manifest.json", "w"), indent=1)
    print(f"[uf] wrote {OUT}: {len(out)} pairs ({man['n_train']} train / {n_eval} held out)", flush=True)
    print(f"[uf] dropped {drop}", flush=True)
    print(f"[uf] length: kept mean {man['length_stats']['kept_mean']:.0f} tok vs all "
          f"{man['length_stats']['all_mean']:.0f}; kept {man['length_stats']['kept_frac_of_scored']:.1%} "
          f"of margin-passing pairs; chosen longer in {man['length_stats']['chosen_longer_frac']:.1%}",
          flush=True)


if __name__ == "__main__":
    main()
