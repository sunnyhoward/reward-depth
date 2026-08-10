#!/usr/bin/env python
"""Build the UF attach-depth report: theme-aware inline-SVG figures + sampled completions.

Reads only banked JSON, so it is re-runnable as arms land (D and E are expected later).
  results/decodability/scalar_qwen3.5-2b_uf_sup_chat.json   the depth curve + floors
  /workspace/sup/head_L*.json                               the readout-competence covariate
  results/uf_depth/eval_*.json                              the arm results
  /workspace/uf_gen.json                                    free-sampled completions

Usage: python uf_report.py [out.html]
"""
import glob
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = f"{REPO}/results/uf_depth"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/uf_report.html"

ARMS = [("A_read10_lora10", "A", 10, "0..10", "on", "attach at L* (probe elbow)"),
        ("B_read21_lora21", "B", 21, "0..21", "on", "attach deep"),
        ("C_read5_lora5", "C", 5, "0..5", "on", "below the elbow"),
        ("D_read10_lora21", "D", 10, "0..21", "on", "L* read, B's parameter count"),
        ("E_read10_lora10_noreplay", "E", 10, "0..10", "off", "L*, replay off")]
COL = {"A": "var(--series-1)", "B": "var(--series-2)", "C": "var(--series-3)",
       "D": "var(--series-7)", "E": "var(--series-8)"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def esc(s):
    return html.escape(str(s))


# ── figure 1: decodability vs depth ───────────────────────────────────────────────────────────

def fig_depth(dec, heads):
    r = dec["results"]["quality|last|linear"]
    acc, fl = r["acc_mean"], dec["floor"]["quality"]
    n = len(acc)
    W, H, ml, mr, mt, mb = 720, 300, 46, 16, 18, 40
    pw, ph = W - ml - mr, H - mt - mb
    y0, y1 = 0.45, 0.85
    X = lambda i: ml + pw * i / (n - 1)
    Y = lambda v: mt + ph * (1 - (v - y0) / (y1 - y0))
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(acc))
    g = []
    for v in (0.5, 0.6, 0.7, 0.8):
        g.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W-mr}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    for v, lab in ((fl["length_only"], "length-only floor"), (fl["group_split"], "lexical floor")):
        g.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W-mr}" y2="{Y(v):.1f}" class="floor"/>'
                 f'<text x="{W-mr-4}" y="{Y(v)-6:.1f}" class="floorlab" text-anchor="end">'
                 f'{lab} {v:.3f}</text>')
    for i in range(0, n, 4):
        g.append(f'<text x="{X(i):.1f}" y="{H-mb+18}" class="tick" text-anchor="middle">{i}</text>')
    mark = []
    for key, tag, blk, _, _, _ in ARMS:
        if tag in ("D", "E"):
            continue
        i = blk + 1                      # sup block b == dec read point b+1
        mark.append(f'<line x1="{X(i):.1f}" y1="{mt}" x2="{X(i):.1f}" y2="{mt+ph}" '
                    f'class="attach" style="stroke:{COL[tag]}"/>'
                    f'<circle cx="{X(i):.1f}" cy="{Y(acc[i]):.1f}" r="5" '
                    f'style="fill:{COL[tag]}" class="dot"/>'
                    f'<text x="{X(i):.1f}" y="{mt-4}" class="attachlab" text-anchor="middle" '
                    f'style="fill:{COL[tag]}">{tag} · blk {blk}</text>')
    return f'''<svg viewBox="0 0 {W} {H}" class="chart" role="img"
     aria-label="Linear probe accuracy on UltraFeedback by read point, Qwen3.5-2B">
  {''.join(g)}
  <polyline points="{pts}" class="line"/>
  {''.join(mark)}
  <text x="{ml+pw/2}" y="{H-4}" class="axlab" text-anchor="middle">read point (0 = embeddings, i = output of block i−1)</text>
</svg>'''


# ── figure 2: readout competence vs depth ─────────────────────────────────────────────────────

def fig_heads(heads):
    if not heads:
        return "<p class='note'>no head metadata found</p>"
    ks = sorted(heads)
    W, H, ml, mr, mt, mb = 720, 170, 46, 16, 16, 40
    pw, ph = W - ml - mr, H - mt - mb
    bw = 54
    out = []
    for j, L in enumerate(ks):
        d = heads[L]
        tag = {5: "C", 10: "A", 21: "B"}.get(L, "")
        x = ml + pw * (j + 0.5) / len(ks) - bw / 2
        h = ph * d["agreement"]
        out.append(f'<rect x="{x:.1f}" y="{mt+ph-h:.1f}" width="{bw}" height="{h:.1f}" rx="4" '
                   f'style="fill:{COL.get(tag, "var(--series-1)")}"/>'
                   f'<text x="{x+bw/2:.1f}" y="{mt+ph-h-7:.1f}" class="barlab" text-anchor="middle">'
                   f'{d["agreement"]:.3f}</text>'
                   f'<text x="{x+bw/2:.1f}" y="{mt+ph+16:.1f}" class="tick" text-anchor="middle">'
                   f'block {L}</text>'
                   f'<text x="{x+bw/2:.1f}" y="{mt+ph+30:.1f}" class="tick" text-anchor="middle">'
                   f'KL {d["kl_heldout"]:.2f}</text>')
    return f'''<svg viewBox="0 0 {W} {H}" class="chart" role="img"
     aria-label="EAGLE head top-1 agreement by attach block">
  <line x1="{ml}" y1="{mt+ph}" x2="{W-mr}" y2="{mt+ph}" class="base"/>
  {''.join(out)}
</svg>'''


# ── figure 4: the train-depth (L_t) curve ─────────────────────────────────────────────────────

def fig_lt(lt, upper, lstar=11):
    """Install vs how much of the stack may move, with no readout anywhere in the loss."""
    if not lt:
        return "<p class='note'>L_t sweep not finished</p>"
    ks = sorted(lt)
    W, H, ml, mr, mt, mb = 720, 280, 46, 16, 22, 46
    pw, ph = W - ml - mr, H - mt - mb
    y0, y1 = 0.5, 0.80
    X = lambda v: ml + pw * v / 23
    Y = lambda v: mt + ph * (1 - (max(min(v, y1), y0) - y0) / (y1 - y0))
    g = []
    for v in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        g.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W-mr}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    # L* marker — the whole point is that nothing happens here
    g.append(f'<line x1="{X(lstar):.1f}" y1="{mt}" x2="{X(lstar):.1f}" y2="{mt+ph}" '
             f'class="attach" style="stroke:var(--series-3)"/>'
             f'<text x="{X(lstar):.1f}" y="{mt-6}" class="attachlab" text-anchor="middle" '
             f'style="fill:var(--series-3)">L* = 11</text>')
    pts = " ".join(f"{X(k):.1f},{Y(lt[k]['uf']):.1f}" for k in ks)
    g.append(f'<polyline points="{pts}" class="line"/>')
    pts2 = " ".join(f"{X(k):.1f},{Y(lt[k]['rb2']):.1f}" for k in ks)
    g.append(f'<polyline points="{pts2}" class="line" style="stroke:var(--series-2)"/>')
    for k in ks:
        g.append(f'<circle cx="{X(k):.1f}" cy="{Y(lt[k]["uf"]):.1f}" r="4" class="dot" '
                 f'style="fill:var(--series-1)"><title>L_t={k} UF {lt[k]["uf"]:.3f}</title></circle>')
        g.append(f'<circle cx="{X(k):.1f}" cy="{Y(lt[k]["rb2"]):.1f}" r="4" class="dot" '
                 f'style="fill:var(--series-2)"><title>L_t={k} rb2 {lt[k]["rb2"]:.3f}</title></circle>')
        g.append(f'<text x="{X(k):.1f}" y="{H-mb+16}" class="tick" text-anchor="middle">{k}</text>')
    # upper-window controls: same block count, top of the stack — drawn as open squares
    for k, v in sorted(upper.items()):
        g.append(f'<rect x="{X(k)-4:.1f}" y="{Y(v["uf"])-4:.1f}" width="8" height="8" rx="2" '
                 f'style="fill:none;stroke:var(--series-1);stroke-width:2">'
                 f'<title>top {k+1} blocks, UF {v["uf"]:.3f}</title></rect>')
        g.append(f'<rect x="{X(k)-4:.1f}" y="{Y(v["rb2"])-4:.1f}" width="8" height="8" rx="2" '
                 f'style="fill:none;stroke:var(--series-2);stroke-width:2">'
                 f'<title>top {k+1} blocks, rb2 {v["rb2"]:.3f}</title></rect>')
    return f'''<svg viewBox="0 0 {W} {H}" class="chart" role="img"
     aria-label="Install accuracy against how many blocks may train">
  {''.join(g)}
  <text x="{ml+pw/2}" y="{H-6}" class="axlab" text-anchor="middle">L_t — LoRA on blocks 0..L_t (open squares: the same block COUNT taken from the TOP of the stack)</text>
</svg>
<p class="note"><span style="color:var(--series-1)">&#9679;</span> held-out UltraFeedback &nbsp;
<span style="color:var(--series-2)">&#9679;</span> RewardBench2 &nbsp; — both length-matched,
implicit (reference-relative) accuracy.</p>'''


# ── figure 3: arm results, grouped bars ───────────────────────────────────────────────────────

def fig_arms(evals, metric, sets, title, base=None):
    rows = []
    for key, tag, blk, lora, rep, _ in ARMS:
        e = evals.get(tag)
        if e:
            rows.append((tag, e))
    if not rows:
        return "<p class='note'>no arm evaluations yet</p>"
    W, mt, mb, ml, mr = 720, 24, 52, 46, 16
    gh = 150
    H = mt + gh + mb
    pw = W - ml - mr
    gw = pw / len(sets)
    bw = min(30, (gw - 30) / max(len(rows), 1))
    y0, y1 = 0.3, 1.0
    Y = lambda v: mt + gh * (1 - (max(min(v, y1), y0) - y0) / (y1 - y0))
    g = []
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        g.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W-mr}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    for si, ds in enumerate(sets):
        cx = ml + gw * (si + 0.5)
        g.append(f'<text x="{cx:.1f}" y="{mt+gh+34:.1f}" class="tick" text-anchor="middle">'
                 f'{ds}</text>')
        if base and ds in base:
            bv = base[ds].get(metric)
            if bv is not None:
                g.append(f'<line x1="{ml+gw*si+8:.1f}" y1="{Y(bv):.1f}" '
                         f'x2="{ml+gw*(si+1)-8:.1f}" y2="{Y(bv):.1f}" class="ref"/>'
                         f'<text x="{ml+gw*si+10:.1f}" y="{Y(bv)-5:.1f}" class="reflab">'
                         f'base {bv:.3f}</text>')
        for bi, (tag, e) in enumerate(rows):
            v = e.get(ds, {}).get(metric)
            if v is None:
                continue
            x = cx - (len(rows) * (bw + 4)) / 2 + bi * (bw + 4)
            h = mt + gh - Y(v)
            g.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{max(h,1):.1f}" '
                     f'rx="4" style="fill:{COL[tag]}"><title>{tag} {ds} {v:.3f}</title></rect>'
                     f'<text x="{x+bw/2:.1f}" y="{Y(v)-5:.1f}" class="barlab" '
                     f'text-anchor="middle">{v:.2f}</text>')
    return f'''<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">
  {''.join(g)}
  <line x1="{ml}" y1="{mt+gh}" x2="{W-mr}" y2="{mt+gh}" class="base"/>
</svg>'''


def main():
    dec = load(f"{REPO}/results/decodability/scalar_qwen3.5-2b_uf_sup_chat.json")
    heads = {}
    for p in glob.glob("/workspace/sup/head_L*.json"):
        d = load(p)
        if d:
            heads[d["layer"]] = d
    base_ev = load(f"{RES}/eval_base.json")
    base = base_ev["base"] if base_ev else None
    evals, ck = {}, {}
    for key, tag, *_ in ARMS:
        e = load(f"{RES}/eval_{key}.json")
        if not e:
            continue
        last = sorted(e, key=lambda k: int(k.rstrip("/").split("ckpt")[-1]))[-1]
        evals[tag] = e[last]
        ck[tag] = {k.rstrip("/").split("/")[-1]: v for k, v in e.items()}
    gen = load("/workspace/uf_gen.json")

    # the L_t sweep: LoRA on blocks 0..L_t (or the top n as a control), loss at the model's output
    lt, upper = {}, {}
    for p in glob.glob(f"{RES}/eval_lt_*.json"):
        tag = os.path.basename(p)[len("eval_lt_"):-5]
        e = load(p)
        ks = [k for k in e if k != "base"]
        if not ks:
            continue
        v = e[ks[0]]
        cell = dict(uf=v["uf_sup"]["implicit_final"], rb2=v["rewardbench2"]["implicit_final"],
                    ob=v["offsetbias"]["implicit_final"],
                    longer=v["uf_sup"].get("implicit_final_chosen_longer"),
                    shorter=v["uf_sup"].get("implicit_final_chosen_shorter"),
                    margin=v["uf_sup"]["margin_final"])
        (upper if tag.startswith("U") else lt)[int(tag[1:])] = cell

    sets = [s for s in ("uf_sup", "offsetbias", "rewardbench2")
            if base and s in base]

    # arms table
    trs = []
    for key, tag, blk, lora, rep, why in ARMS:
        e = evals.get(tag)
        hd = heads.get(blk)
        cells = [f'<b style="color:{COL[tag]}">{tag}</b>', str(blk), lora, rep,
                 f'{hd["agreement"]:.3f}' if hd else "—",
                 f'{hd["kl_heldout"]:.2f}' if hd else "—"]
        for s in sets:
            v = e.get(s, {}) if e else {}
            cells.append(f'{v["implicit_final"]:.3f}' if "implicit_final" in v else "—")
        cells.append(f'{e["uf_sup"]["dlp_chosen"]:+.1f}' if e and "uf_sup" in e else "—")
        g = gen["gens"].get(f"/workspace/uf_{key}/ckpt400") if gen else None
        cells.append(f'{g["mean_ntok"]:.0f}' if g else "—")
        cells.append(f'<span class="why">{esc(why)}</span>')
        trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    if base:
        bc = ["<b>base</b>", "—", "—", "—", "—", "—"]
        for s in sets:
            bc.append(f'{base[s]["raw_final"]:.3f}<span class="sub"> raw</span>')
        bc += ["0.0", f'{gen["gens"]["base"]["mean_ntok"]:.0f}' if gen else "—",
               '<span class="why">reference; implicit is 0 by construction</span>']
        trs.append("<tr class='baserow'>" + "".join(f"<td>{c}</td>" for c in bc) + "</tr>")

    # stage 2 — "train upwards". NOTE the reference differs: with S2_FROM_S1=1 the adapter-off
    # branch is the stage-1 MERGED model, so `implicit` here means "did stage 2 improve on its own
    # stage 1", not "did the pipeline beat base". `raw` is the column comparable across stages.
    s2rows = []
    for tag, layer, s1tag in (("A10", 10, "A"), ("B21", 21, "B")):
        e = load(f"{RES}/eval_s2_{tag}.json")
        if not e:
            continue
        last = sorted(e, key=lambda k: int(k.rstrip("/").split("ckpt")[-1]))[-1]
        v = e[last]
        s1 = evals.get(s1tag, {})
        c = [f'<b style="color:{COL[s1tag]}">stage 2 · {tag}</b>', str(23 - layer)]
        for s in sets:
            m = v.get(s, {})
            raw = m.get("raw_final")
            s1raw = s1.get(s, {}).get("raw_final")
            d = (raw - s1raw) if (raw is not None and s1raw is not None) else None
            c.append(f'{raw:.3f}<span class="sub"> ({d:+.3f})</span>' if d is not None
                     else (f'{raw:.3f}' if raw is not None else "—"))
        c.append(f'{v["uf_sup"]["implicit_final"]:.3f}' if "uf_sup" in v else "—")
        s2rows.append("<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>")
    if base:
        c = ["<b>base</b>", "—"] + [f'{base[s]["raw_final"]:.3f}' for s in sets] + ["—"]
        s2rows.append("<tr class='baserow'>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>")
    s2hdr = ("<th>run</th><th>blocks adapted</th>"
             + "".join(f"<th>{s}<span class='sub'> raw (Δ vs its stage 1)</span></th>" for s in sets)
             + "<th>vs own stage 1</th>")
    s2sec = (f'<div class="card tw"><table><thead><tr>{s2hdr}</tr></thead>'
             f'<tbody>{"".join(s2rows)}</tbody></table></div>' if s2rows
             else "<p class='note'>stage-2 runs not finished</p>")

    # length-matched replication
    lmrows = []
    for tag, layer, s1tag in (("A10", 10, "A"), ("B21", 21, "B")):
        e = load(f"{RES}/eval_lm_{tag}.json")
        if not e:
            continue
        ck = [k for k in e if k != "base"]
        if not ck:
            continue
        v = e[sorted(ck, key=lambda k: int(k.rstrip("/").split("ckpt")[-1]))[-1]]
        bl = e.get("base", {})
        c = [f'<b style="color:{COL[s1tag]}">matched · {tag}</b>']
        for s in sets:
            m = v.get(s, {})
            c.append(f'{m["implicit_final"]:.3f}' if "implicit_final" in m else "—")
        u = v.get("uf_sup", {})
        c.append(f'{u.get("implicit_final_chosen_longer", float("nan")):.3f}')
        c.append(f'{u.get("implicit_final_chosen_shorter", float("nan")):.3f}')
        c.append(f'{bl.get("uf_sup", {}).get("length_cheat", float("nan")):.3f}')
        lmrows.append("<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>")
    lmhdr = ("<th>run</th>" + "".join(f"<th>{s}</th>" for s in sets)
             + "<th>chosen-longer</th><th>chosen-shorter</th><th>length cheat</th>")
    lmsec = (f'<div class="card tw"><table><thead><tr>{lmhdr}</tr></thead>'
             f'<tbody>{"".join(lmrows)}</tbody></table></div>' if lmrows
             else "<p class='note'>length-matched arms not finished</p>")

    # length split
    lrows = []
    for key, tag, *_ in ARMS:
        e = evals.get(tag)
        if not e:
            continue
        c = [f'<b style="color:{COL[tag]}">{tag}</b>']
        for s in sets:
            v = e.get(s, {})
            lo = v.get("implicit_final_chosen_longer")
            sh = v.get("implicit_final_chosen_shorter")
            c.append(f'{lo:.3f}' if lo is not None else "—")
            c.append(f'{sh:.3f}' if sh is not None else "—")
            c.append(f'<span class="sub">{sh-lo:+.3f}</span>'
                     if (lo is not None and sh is not None) else "—")
        lrows.append("<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>")

    # completions
    gblocks = ""
    if gen:
        order = ["base"] + [f"/workspace/uf_{k}/ckpt400" for k, *_ in ARMS]
        order = [o for o in order if o in gen["gens"]]
        for i in range(min(4, len(gen["prompts"]))):
            cols = ""
            for o in order:
                tag = "base" if o == "base" else next(
                    (t for k, t, *_ in ARMS if f"uf_{k}/" in o), "?")
                gg = gen["gens"][o]
                cols += (f'<div class="gcol"><div class="ghead" style="color:'
                         f'{COL.get(tag, "var(--text-secondary)")}">{tag}'
                         f'<span class="sub"> · {gg["ntok"][i]} tok</span></div>'
                         f'<pre>{esc(gg["text"][i][:900])}</pre></div>')
            gblocks += (f'<div class="gen"><div class="gprompt">{esc(gen["prompts"][i][:300])}</div>'
                        f'<div class="grow">{cols}</div></div>')

    hdr = ("<th>arm</th><th>read</th><th>LoRA</th><th>replay</th><th>head agr</th><th>head KL</th>"
           + "".join(f"<th>{s}</th>" for s in sets)
           + "<th>Δlp chosen</th><th>gen tok</th><th></th>")
    lhdr = "<th>arm</th>" + "".join(
        f"<th>{s}<span class='sub'> longer</span></th><th>{s}<span class='sub'> shorter</span></th>"
        f"<th><span class='sub'>Δ</span></th>" for s in sets)

    doc = f'''<title>UltraFeedback at the probe elbow — attach-depth arms</title>
<style>
:root {{
  color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --hair:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --series-7:#4a3aa7; --series-8:#e34948;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --hair:rgba(255,255,255,0.10);
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
    --series-7:#9085e9; --series-8:#e66767;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --hair:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --series-7:#9085e9; --series-8:#e66767;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:40px 20px 72px; background:var(--plane); color:var(--text-primary);
  font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:880px; margin:0 auto; display:flex; flex-direction:column; gap:6px; }}
h1 {{ font-size:30px; line-height:1.15; margin:0; letter-spacing:-0.022em; text-wrap:balance;
  font-weight:640; max-width:22ch; }}
h2 {{ font-size:19px; margin:40px 0 4px; letter-spacing:-0.008em; font-weight:640;
  display:flex; align-items:baseline; gap:10px; }}
h2::before {{ content:attr(data-n); font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--muted); letter-spacing:.1em; padding-top:2px; }}
h3 {{ font-size:15px; margin:24px 0 8px; color:var(--text-secondary); }}
p {{ margin:10px 0; color:var(--text-secondary); max-width:68ch; }}
.lede {{ font-size:16.5px; color:var(--text-secondary); margin:8px 0 4px; max-width:66ch; }}
.eyebrow {{ font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin:0 0 14px; }}
.verdict {{ background:var(--surface-1); border:1px solid var(--hair); border-left:3px solid
  var(--series-2); border-radius:12px; padding:18px 20px; margin:22px 0 8px; }}
.verdict h2 {{ margin:0 0 8px; font-size:17px; }}
.verdict h2::before {{ content:none; }}
.verdict p {{ margin:8px 0 0; color:var(--text-primary); }}
.vgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px;
  margin-top:16px; }}
.vcell {{ border-top:2px solid var(--hair); padding-top:8px; }}
.vcell .k {{ font:600 10.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); }}
.vcell .v {{ font-size:23px; font-weight:640; font-variant-numeric:tabular-nums;
  margin-top:5px; letter-spacing:-0.02em; }}
.vcell .d {{ font-size:12px; color:var(--muted); margin-top:2px; }}
.card {{ background:var(--surface-1); border:1px solid var(--hair); border-radius:12px;
  padding:16px; margin:14px 0; }}
.chart {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.base, .ref {{ stroke:var(--axis); stroke-width:1; }}
.ref {{ stroke-dasharray:4 3; }}
.floor {{ stroke:var(--muted); stroke-width:1; stroke-dasharray:5 4; }}
.line {{ fill:none; stroke:var(--series-1); stroke-width:2; stroke-linejoin:round; }}
.attach {{ stroke-width:1; stroke-dasharray:3 3; opacity:.55; }}
.dot {{ stroke:var(--surface-1); stroke-width:2; }}
.tick {{ fill:var(--muted); font-size:11px; }}
.axlab {{ fill:var(--muted); font-size:11px; }}
.floorlab {{ fill:var(--muted); font-size:10.5px; }}
.attachlab, .barlab {{ font-size:11px; font-weight:600; }}
.barlab {{ fill:var(--text-secondary); }}
.reflab {{ fill:var(--muted); font-size:10px; }}
.tw {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px;
  font-variant-numeric:tabular-nums; }}
th,td {{ text-align:right; padding:7px 9px; border-bottom:1px solid var(--hair);
  white-space:nowrap; }}
th {{ color:var(--muted); font-weight:600; font-size:11.5px; text-transform:uppercase;
  letter-spacing:.04em; }}
td:first-child, th:first-child, td:last-child, th:last-child {{ text-align:left; }}
.baserow td {{ color:var(--text-secondary); }}
.sub {{ color:var(--muted); font-size:11px; font-weight:400; }}
.why {{ color:var(--muted); font-size:12px; white-space:normal; }}
.note {{ color:var(--muted); font-size:13px; }}
.warn {{ border-left:3px solid var(--series-2); padding-left:12px; }}
.gen {{ border:1px solid var(--hair); border-radius:12px; margin:14px 0;
  background:var(--surface-1); overflow:hidden; }}
.gprompt {{ padding:12px 14px; border-bottom:1px solid var(--hair); font-size:13px;
  color:var(--text-secondary); }}
.grow {{ display:flex; gap:0; overflow-x:auto; }}
.gcol {{ min-width:280px; flex:1; padding:12px 14px; border-right:1px solid var(--hair); }}
.gcol:last-child {{ border-right:0; }}
.ghead {{ font-size:12px; font-weight:700; margin-bottom:6px; }}
pre {{ margin:0; white-space:pre-wrap; word-break:break-word; font-size:12px; line-height:1.5;
  color:var(--text-secondary); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
ul {{ color:var(--text-secondary); padding-left:20px; }}
li {{ margin:6px 0; }}
code {{ background:var(--surface-1); border:1px solid var(--hair); border-radius:4px;
  padding:1px 5px; font-size:12.5px; }}
</style>

<main>
<p class="eyebrow">reward-depth · supervisor recipe · 2026-08-10</p>
<h1>UltraFeedback at the probe elbow</h1>
<p class="lede">The supervisor's stage-1 recipe — DPO through a frozen EAGLE readout plus
generative replay — ported from the constructed <code>britishness</code> set to a real preference
dataset, with the attach layer chosen by where the preference becomes linearly decodable.
Qwen3.5-2B, one seed, five stage-1 arms.</p>
<p class="lede">Five stage-1 arms, both stage-2 arms, and a length-matched replication of the
headline pair. Two objections that could each have overturned the reading were run rather than
argued: stage 2 (§5), because stage 1 scores every arm through its own head and head fidelity rises
with depth by construction; and length matching (§4a, §6), because UF's decodable core clears a
length-only probe by only ~0.18. Neither rescued the elbow; both sharpened the result.</p>

<div class="verdict">
<h2>The elbow is real. Training through it is not.</h2>
<p>The premise holds under its own hardest control: make length literally unreadable and the
decodability curve keeps its shape, its peak and its elbow — <b>L* = 11 either way</b>, ~0.79
against a floor of exactly 0.500. The preference genuinely is linearly readable from block 11.</p>
<p>But every way of <i>training</i> through a readout there fails. Attaching at the elbow installs
essentially nothing once length stops paying (0.526 against a 0.5 chance line) while attaching deep
installs a real, length-symmetric preference (0.753, split 0.783 / 0.723). The recipe's own remedy
— stage 2, distilling upward — makes the elbow arm <i>worse than the untrained base</i>.</p>
<p>And with the readout removed entirely (§9), the depth axis flattens: ordinary DPO shows no elbow
at L*, its rise is <b>capacity rather than depth</b> (the top n blocks tie or beat the bottom n),
and <b>two trainable blocks beat the recipe at the elbow with eleven</b>. On this data the readout
was the cost, not the attach point.</p>
<div class="vgrid">
  <div class="vcell"><div class="k">L* (probe elbow)</div><div class="v">block 11</div>
    <div class="d">unchanged under length matching</div></div>
  <div class="vcell"><div class="k">install at L*</div><div class="v">0.526</div>
    <div class="d">length-matched · chance is 0.500</div></div>
  <div class="vcell"><div class="k">install deep (blk 21)</div><div class="v">0.753</div>
    <div class="d">length-matched · split .783 / .723</div></div>
  <div class="vcell"><div class="k">no readout, 2 blocks</div><div class="v">0.657</div>
    <div class="d">plain DPO beats the recipe at L*</div></div>
</div>
</div>

<h2 data-n="01">Where UltraFeedback becomes decodable</h2>
<p>Linear probe on last-token residuals, 3 seeds, held out by prompt. The two dashed lines are the
controls that decide whether the curve means anything: a bag-of-token-ids probe with no model, and
a probe that sees only completion length.</p>
<div class="card">{fig_depth(dec, heads)}</div>
<p><b>L* = 11</b> (earliest read point within 1 SE of the curve's own max), peak 0.800 at read
point 15. Converting to the trainer's convention — it reads the <i>output of block L</i>, the sweep
indexes the embedding output as 0 — that is <b>block 10</b>. The same measurement on
Llama-3.1-Tulu-3-8B-SFT gave plateau 0.799 from L12 of 32 with a length floor of 0.62; here it is
0.800 from L11 of 24 with a length floor of 0.623, on a model family that ladder never saw.</p>
<p class="warn"><b>The floor is the story as much as the peak.</b> The plateau clears a
length-only probe by ~0.18 and a bag-of-words probe by ~0.16. Most of what is linearly readable in
UltraFeedback at any depth is surface.</p>

<h2 data-n="02">The confound this design has to survive</h2>
<p>An EAGLE head is distilled to reconstruct the model's output distribution from the residual at
its block. Deeper residuals are closer to the output, so head quality rises with depth
<i>by construction</i> — at an identical 5000-step budget on the same replay corpus:</p>
<div class="card">{fig_heads(heads)}</div>
<p>Probe decodability goes .749 → .795 → .784 across blocks 5 / 10 / 21 (rises, then flat);
head agreement goes .361 → .404 → .812 (rises, then jumps). <b>They disagree between block 10 and
block 21</b>, which is what makes the arms separable: an install tracking decodability predicts
C &lt; A ≈ B, one tracking readout quality predicts C &lt; A &lt; B.</p>

<h2 data-n="03">The arms</h2>
<div class="card tw"><table><thead><tr>{hdr}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>
<p class="note">Columns for the three preference sets are <b>implicit</b> (reference-relative)
ranking accuracy at the model's own output — the DPO implicit reward, and the column comparable to
phase 3's 0.800. It is 0 at step 0 by construction, so the base row shows raw accuracy instead.
Δlp chosen is chosen-side log-prob displacement in nats on held-out UF.</p>

<h2 data-n="04">Install and transfer</h2>
<div class="card">{fig_arms(evals, "implicit_final", sets, "Implicit ranking accuracy by arm and dataset", base)}</div>
<p><b>The observed ordering is C ≈ A ≪ B</b>, which is neither pre-registered pattern, and it
separates the two candidate drivers more sharply than either would have:</p>
<div class="card tw"><table>
<thead><tr><th>step along the depth axis</th><th>probe accuracy</th><th>head agreement</th>
<th>install (UF implicit)</th></tr></thead>
<tbody>
<tr><td>block 5 → 10</td><td><b>rises</b> .749 → .795</td><td>flat .361 → .404</td>
    <td>flat 0.583 → 0.544</td></tr>
<tr><td>block 10 → 21</td><td>flat/falls .795 → .784</td><td><b>doubles</b> .404 → .812</td>
    <td><b>jumps</b> 0.544 → 0.731</td></tr>
</tbody></table></div>
<p>Decodability and readout competence disagree at both steps, and the install follows the readout
both times. <b>On this recipe, attaching where the preference is most decodable is not better than
attaching deep — it is worse.</b> That is the same shape as the repo's earlier read-depth test on
UltraFeedback, which found the arms ordered by probe accuracy rather than by depth; here the
ordering variable is readout fidelity rather than probe accuracy, and the conclusion — depth per se
buys nothing — survives the change of instrument.</p>
<p class="warn"><b>Read depth and preference capacity cannot be separated in this recipe — they are
the same variable.</b> The stage-1 loss is read from <code>h_L</code>, and <code>h_L</code> is a
function of blocks 0..L only, so every block above the read point is <i>outside the loss's graph</i>.
Measured directly with <code>torch.autograd.grad(..., allow_unused=True)</code> on a
read-at-10 / LoRA-0..21 model: blocks 0, 5, 9, 10 carry preference gradient (norms .24, .27, .31,
.38); blocks 11, 15, 21 are <b>not in the graph at all</b> — autograd returns <code>None</code>,
not a small number. Arm D therefore does <i>not</i> give the elbow read B's preference capacity;
its upper blocks move under the replay term only. Attaching shallow <i>means</i> training less of
the network on the preference, and no arm inside this recipe can decouple the two.</p>

<h2 data-n="05">Stage 2 — "train upwards"</h2>
<p>§4 measures stage 1 only, and stage 1 scores every arm through its own EAGLE head. Head fidelity
rises with depth by construction, so that comparison is structurally kind to the deep arm — and the
recipe's own answer to a weak shallow readout is <i>this</i> stage, where the aligned readout
teaches the full network and the head stops being the artifact. Student initialised from the
stage-1 merge (<code>S2_FROM_S1=1</code>), so blocks 0..L keep the install and stage 2 only
propagates it upward.</p>
<p class="note">The reference differs from §4: with the student initialised from stage 1, the
adapter-off branch <i>is</i> the stage-1 model, so the last column reads "did stage 2 improve on
its own stage 1", not "did the pipeline beat base". The raw columns are the ones comparable across
stages, and each carries its change against that arm's stage 1.</p>
{s2sec}
<p class="warn"><b>The asymmetry that no arm here removes:</b> stage 2 adapts blocks L+1..23, so the
elbow arm gets 13 blocks to distil into and the deep arm gets 2. That runs <i>opposite</i> to
stage 1's bias. Neither stage alone is the recipe.</p>

<h2 data-n="06">Length-matched replication</h2>
<p>Every arm above installed mostly "prefer the longer completion", which leaves the headline open
to a deflationary reading: perhaps the deep arm is only better at learning the length rule.
<code>sup_uf_lenmatch.py</code> removes the rule instead of correcting for it — chosen is the longer
side in exactly 50% of pairs within every |Δ tokens| stratum, in train and held-out separately
(3972 / 534 from 5250 / 750), so "prefer longer" is worth 0.500 by construction. Arms are retrained
and scored on the matched set.</p>
{lmsec}

<h2 data-n="07">Is it length?</h2>
<p>UltraFeedback's chosen side is the longer one in 61.2% of these pairs, and summed log-probability
is monotone in length. So every accuracy is split by which side is longer. An arm that learned
"prefer the longer completion" scores well on the left half of each pair and badly on the right.</p>
<div class="card tw"><table><thead><tr>{lhdr}</tr></thead><tbody>{''.join(lrows)}</tbody></table></div>
<p><b>Yes, largely.</b> Arm A's held-out aggregate (0.576 at ckpt100) is reconstructed almost
exactly by the length split alone — 0.616 × 0.745 + 0.384 × 0.306 = 0.576 — and it sits
<i>below chance on the half where the better answer is the shorter one</i>. offsetbias settles it
from the other side: it is built so the appealing response is the rejected one, and arm A scores
<b>0.249</b> implicit there, i.e. it moves the margin the wrong way three times in four.</p>
<p>Arm B is the one arm that learned something more: 0.625 on the chosen-shorter half against
A's 0.458. It is still length-biased — 0.797 vs 0.625 — and still below chance on offsetbias
(0.360), so "installed the preference" overstates it. The honest summary is that the deep arm
installed <i>some</i> preference on top of a length prior, and the two shallower arms installed
mostly the length prior.</p>
<p class="note">Note how invisible this is in the raw column: on offsetbias every arm reads
0.83–0.85 against a base of 0.848, because the base model's own length bias dominates the absolute
ranking. Only the reference-relative column and the length split show the damage.</p>

<h2 data-n="08">What the arms actually write</h2>
<p>Greedy decoding, same held-out prompts, 200 new tokens max. The standing instruction in this
repo is to read raw generations before believing any metric.</p>
{gblocks}

<h2 data-n="09">The train-depth sweep — the version with no readout at all</h2>
<p>Every arm above reads the preference through a distilled EAGLE head, and head fidelity rises
with depth <i>by construction</i> — a confound that forced one mechanism claim in §4 to be
withdrawn. This sweep removes it: the DPO loss is taken at the model's own output, there is no head
anywhere, and the only thing that varies is <b>which blocks may move</b>. 15 arms, length-matched
data, identical budget.</p>
<div class="card">{fig_lt(lt, upper)}</div>
<p><b>There is no elbow at L*.</b> The curve rises smoothly with how much of the stack can train and
saturates around L_t ≈ 15–19; block 11 is unremarkable, sitting between its neighbours.
<b>Decodability depth does not predict training depth.</b></p>
<p><b>Two trainable blocks already reach 87% of the ceiling</b> (0.657 against 0.755 at L_t = 19)
with 0.8M parameters — and that beats the supervisor recipe reading through a distilled head at the
elbow with eleven trainable blocks (0.526). The recipe's best arm (block 21, 0.753) merely matches
plain DPO at L_t = 19 while needing a 25.2M head to get there.</p>
<p class="warn"><b>The matched-count control kills the "early layers" story.</b> The open squares are
the same number of blocks taken from the TOP of the stack. At 6 and 18 blocks they tie the bottom
window; at 12 blocks the top window <i>wins</i> — 0.770 vs 0.715 on UF and 0.678 vs 0.614 on
RewardBench2. So the rising curve is <b>capacity, not depth</b>, and what depth effect exists
favours the late blocks.</p>
<p class="note">Budget caveat that applies to the whole study: 400 steps × 6 pairs = 2400 pair-draws
is <b>0.6 epochs</b>, so every arm sees ~1800 of 3972 unique pairs. All arms get the identical
budget, so the comparison is fair, but "flat" could in principle mean "equally undertrained". The
replay term is smaller still — ≤6400 scored tokens per run against a 1.32M-token bank, and because
UF pairs are ~5× longer than britishness ones, his fixed 16-token setting silently down-weights
replay ~5× in this port.</p>

<h2 data-n="10">Caveats that travel with every number</h2>
<ul>
<li><b>One seed per cell.</b> The repo's own record is that single-seed positives churn.</li>
<li><b>A short-pair subset of UF.</b> The ≤512-token filter keeps 49.3% of margin-passing pairs
(mean length 273 vs 496 tokens unfiltered). It also drops 484 of 1234 RewardBench2 pairs and
removes the <i>Focus</i> and <i>Ties</i> subsets entirely.</li>
<li><b>Read depth and readout competence cannot be fully separated</b> by any arm here — arm C
bounds the confound, it does not remove it. The repo's proposed clean design is a family contrast
through one shared head.</li>
<li><b>In-training eval lines are not trajectories.</b> <code>sup_train.evaluate()</code> resamples
a different 128 of the 750 held-out rows each time; every number above instead comes from the full
deterministic 750.</li>
</ul>
</main>'''
    open(OUT, "w").write(doc)
    print(f"[report] wrote {OUT} ({len(doc)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
