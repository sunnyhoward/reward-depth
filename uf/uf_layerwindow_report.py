#!/usr/bin/env python
"""Results table for the layer-restricted DPO study: best LR per condition, endpoints only.

Reports each cell's final and best in-run held-out implicit-reward accuracy, then the best LR per
condition. `full` is printed in a separate section because it carries 2x the trainable parameters
and is NOT part of the matched comparison."""
import json, os, glob

ROWS = []
for f in sorted(glob.glob("/workspace/uf_dpo_A_*_history.json")):
    name = os.path.basename(f)[len("uf_dpo_A_"):-len("_history.json")]
    h = json.load(open(f))
    ev = h.get("evals", [])
    if not ev: continue
    cond, _, lr = name.partition("_lr")
    fin = ev[-1]
    best = max(ev, key=lambda e: e["acc_implicit"])
    ROWS.append(dict(cond=cond, lr=lr, final=fin["acc_implicit"], final_step=fin["step"],
                     best=best["acc_implicit"], best_step=best["step"],
                     dlp_c=fin.get("dlp_chosen"), dlp_r=fin.get("dlp_rejected"),
                     ntrain=h.get("n_trainable")))

if not ROWS:
    print("no layer-window histories yet"); raise SystemExit

print("=== matched comparison (lower vs upper, 20.97M trainable each) ===")
print(f"{'cond':>6} {'lr':>8} {'final':>7} {'best':>7} {'@step':>6} {'dlp_c':>7} {'dlp_r':>8}")
for r in [r for r in ROWS if r["cond"] in ("lower", "upper")]:
    print(f"{r['cond']:>6} {r['lr']:>8} {r['final']:>7.3f} {r['best']:>7.3f} {r['best_step']:>6} "
          f"{(r['dlp_c'] or 0):>7.2f} {(r['dlp_r'] or 0):>8.2f}")

print("\n=== best LR per condition ===")
for c in ("lower", "upper", "full"):
    cells = [r for r in ROWS if r["cond"] == c]
    if not cells: continue
    b = max(cells, key=lambda r: r["best"])
    tag = "  [REFERENCE ONLY - 2x params, NOT matched]" if c == "full" else ""
    print(f"  {c:>6}: best LR {b['lr']:>8} -> {b['best']:.3f} (final {b['final']:.3f}){tag}")

lo = [r for r in ROWS if r["cond"] == "lower"]; up = [r for r in ROWS if r["cond"] == "upper"]
if lo and up:
    bl, bu = max(lo, key=lambda r: r["best"]), max(up, key=lambda r: r["best"])
    print(f"\n  lower(best) {bl['best']:.3f} @ lr {bl['lr']}  vs  upper(best) {bu['best']:.3f} @ lr {bu['lr']}"
          f"   delta {bl['best']-bu['best']:+.3f}")
    print("  NOTE: in-run evals are 128 unmatched held-out pairs (SE ~0.04). Differences below")
    print("        ~0.08 are not resolvable here -- use uf_bigN_eval.py on saved endpoints.")
