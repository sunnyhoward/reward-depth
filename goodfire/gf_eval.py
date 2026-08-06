#!/usr/bin/env python
"""Final held-out evaluation for a trained run, at larger N than the in-training checkpoints,
and a dump of raw generations.

Reading the raw text is not optional here: the oracle counts unique AE/BE axes, so a marker
word-salad scores well while being obvious garbage to a human. `results/gens/<tag>.txt` exists so
that check is one file open away.

Usage: python gf_eval.py --tag run1_oracle [--n 8]
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import gf_common as G
import gf_rl as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default="", help="defaults to <WORK>/runs/<tag>/final")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--probes", default="probes")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    G.seed_all(a.seed)
    adapter = Path(a.adapter) if a.adapter else G.WORK / "runs" / a.tag / "final"
    assert adapter.exists(), f"no adapter at {adapter}"

    P = G.jload(G.RESULTS / "prompts.json")
    held = P["heldout"]
    oracle = G.BritOracle()
    tok = G.load_tokenizer()
    pk = torch.load(G.WORK / f"{a.probes}.pt", map_location="cpu", weights_only=False)
    NL = pk["n_layers"]; layers = list(range(NL + 1))
    probes_gpu = {l: G.LayerProbe.load(pk["probes"][l]).to(G.DEV) for l in layers}

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=G.MODEL_ID, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=1024, enable_prefix_caching=True, seed=a.seed,
              enable_lora=True, max_lora_rank=a.lora_r, max_loras=1)
    sp = SamplingParams(n=a.n, temperature=a.temp, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)
    texts = [G.build_prompt(tok, c["text"]) for c in held]

    from peft import PeftModel
    base = G.load_base()
    policy = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    policy.eval()

    out = {}
    for arm, lreq in (("base", None), ("policy", LoRARequest(a.tag, 1, str(adapter)))):
        gens = llm.generate(texts, sp, lora_request=lreq) if lreq else llm.generate(texts, sp)
        fp, fc, gi = [], [], []
        for k, (pt, o) in enumerate(zip(texts, gens)):
            for x in o.outputs:
                fp.append(pt); fc.append(x.text); gi.append(k)
        seqs = R.make_seqs(tok, fp, fc, a.max_tokens)
        osc = [oracle.score(t) for t in fc]
        deg = [G.degeneracy(t) for t in fc]
        ps = R.probe_scores(policy, tok, seqs, probes_gpu, layers, on_student=False)
        pooled = {l: [p["pooled"][l] for p in ps] for l in layers}
        be = [s["be_rate"] for s in osc]
        out[arm] = {
            "n": len(fc),
            "be_rate": float(np.mean(be)),
            "be_rate_sem": float(np.std(be) / np.sqrt(len(be))),
            "coverage": float(np.mean([s["covered"] for s in osc])),
            "n_hits": float(np.mean([s["n_hits"] for s in osc])),
            "n_be": float(np.mean([s["n_be"] for s in osc])),
            "n_ae": float(np.mean([s["n_ae"] for s in osc])),
            "len": float(np.mean([s["comp_len"] for s in seqs])),
            "distinct1": float(np.mean([d["distinct1"] for d in deg])),
            "distinct3": float(np.mean([d["distinct3"] for d in deg])),
            "max_rep": float(np.mean([d["max_rep"] for d in deg])),
            "probe_pooled_by_layer": {str(l): float(np.mean(v)) for l, v in pooled.items()},
            "probe_oracle_spearman_by_layer": {str(l): G.spearman(v, be) for l, v in pooled.items()},
        }
        print(f"[{arm}] BE {out[arm]['be_rate']:.3f}+-{out[arm]['be_rate_sem']:.3f} "
              f"cov {out[arm]['coverage']:.2f} hits {out[arm]['n_hits']:.1f} "
              f"len {out[arm]['len']:.0f} d3 {out[arm]['distinct3']:.2f} "
              f"maxrep {out[arm]['max_rep']:.1f}", flush=True)
        if arm == "policy":
            gd = G.RESULTS / "gens"; gd.mkdir(exist_ok=True)
            with open(gd / f"{a.tag}.txt", "w") as f:
                for k in range(0, len(fc), a.n):
                    f.write("=" * 90 + f"\nPROMPT: {held[gi[k]]['text']}\n")
                    for j in range(a.n):
                        s = osc[k + j]
                        f.write(f"--- be={s['be_rate']:.2f} be/ae={s['n_be']}/{s['n_ae']}\n"
                                f"{fc[k+j].strip()}\n")
            print(f"[write] {gd / (a.tag + '.txt')}", flush=True)

    with G.as_base(policy):
        cap_base = R.capability_score(policy, tok)
    cap = R.capability_score(policy, tok)
    out["capability"] = {"policy": cap, "base": cap_base, "delta": cap - cap_base}
    print(f"[cap] policy {cap:.3f} vs base {cap_base:.3f} (delta {cap-cap_base:+.3f})", flush=True)
    G.jdump({"tag": a.tag, "adapter": str(adapter), "n_per_prompt": a.n,
             "n_prompts": len(held), **out}, G.RESULTS / "evals" / f"{a.tag}.json")


if __name__ == "__main__":
    main()
