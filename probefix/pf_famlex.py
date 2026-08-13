#!/usr/bin/env python
"""Extended, FAMILY-LABELLED marker lexicon.

WHY THIS EXISTS. `sup_common.marker_lexicon()` builds its regex only from `item` fields of the
form `american|british` AND drops anything containing a space. Only two families survive that
filter:

    lexicon 251 pairs, culture 141 pairs, everything else ZERO.

So every free-generation number this project has ever reported -- brit_rate, marker_density,
br/am counts, per-word rates -- measures two of six families, and is blind to the four that
actually separate a real install from a constant steering vector (2026-08-13: a 2,560-parameter
constant vector matches C0 on generation while scoring .59 on install_style and .58 on
install_expression against C0's .96/.92).

Two of the four are recoverable without a judge:
  · expression  -- `meta.slots` carries real phrase pairs ("the greatest|the bee's knees").
                   They are multi-word, which is the only reason the old filter dropped them.
  · false_friend -- chosen/rejected are short and differ in exactly one word (sweets/candy);
                   recoverable by diffing the pair.

`style` (register and phrasing, no surface form) and the truth `guard` (needs a truth judgement)
are not recoverable this way and are handled by the judge in pf_judge.py.

Coverage: 731 British forms across four families, against the old 392 across two.
"""
import json
import os
import re
from collections import defaultdict

E = os.environ.get
SRC = E("SUP_BRIT", "/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl")
STRIP = '.,;:!?"’“”‘()'


def _rows(path=None):
    return [json.loads(l) for l in open(path or SRC)]


def family_lexicon(path=None):
    """-> (am2fam, br2fam, regex_by_family). Each regex matches only THAT family's markers, so a
    generation can be scored per family instead of pooled."""
    am2fam, br2fam = {}, {}

    def add(am, br, fam):
        am, br = am.strip().lower(), br.strip().lower()
        if am and br and am != br:
            am2fam[am] = fam
            br2fam[br] = fam

    for r in _rows(path):
        fam = r["family"]
        it = r.get("item", "")
        if "|" in it:                                   # lexicon + culture, as before
            a, b = it.split("|", 1)
            if " " not in a and " " not in b:
                add(a, b, fam)
        for s in (r["meta"].get("slots") or []):        # expression: multi-word phrases
            if "|" in s:
                a, b = s.split("|", 1)
                add(a, b, fam)
        if fam == "false_friend" and r["role"] == "install":
            cw, jw = r["chosen"].lower().split(), r["rejected"].lower().split()
            if len(cw) == len(jw):
                d = [(x, y) for x, y in zip(jw, cw) if x != y]
                if len(d) == 1:
                    add(d[0][0].strip(STRIP), d[0][1].strip(STRIP), fam)

    def _mk(words):
        # longest-first so "the bee's knees" wins over any nested single word
        ws = sorted(words, key=len, reverse=True)
        return re.compile(r"(?<!\w)(" + "|".join(map(re.escape, ws)) + r")(?!\w)") if ws else None

    by_fam = defaultdict(lambda: {"am": [], "br": []})
    for w, f in am2fam.items():
        by_fam[f]["am"].append(w)
    for w, f in br2fam.items():
        by_fam[f]["br"].append(w)
    rex = {f: {"am": _mk(v["am"]), "br": _mk(v["br"])} for f, v in by_fam.items()}
    return am2fam, br2fam, rex


def score(texts, rex):
    """Per-family marker counts over a list of generations.
    -> {family: {br, am, brit_rate, density}} plus an 'ALL' pooled row for continuity with the
    old numbers. Report the families; the pooled row is only for cross-checking against banked
    results, and pooling is what hid the problem in the first place."""
    out = {}
    tb = ta = 0
    for fam, r in rex.items():
        nb = sum(len(r["br"].findall(t.lower())) for t in texts) if r["br"] else 0
        na = sum(len(r["am"].findall(t.lower())) for t in texts) if r["am"] else 0
        tb += nb
        ta += na
        out[fam] = dict(br=nb, am=na, n=len(texts),
                        brit_rate=(nb / (nb + na)) if (nb + na) else None,
                        density=(nb + na) / max(1, len(texts)))
    out["ALL"] = dict(br=tb, am=ta, n=len(texts),
                      brit_rate=(tb / (tb + ta)) if (tb + ta) else None,
                      density=(tb + ta) / max(1, len(texts)))
    return out


if __name__ == "__main__":
    a, b, rex = family_lexicon()
    print(f"{len(b)} british forms across {len(rex)} families "
          f"(sup_common.marker_lexicon gives 392 across 2)")
    for f in sorted(rex):
        n = sum(1 for w, ff in b.items() if ff == f)
        ex = [w for w, ff in b.items() if ff == f][:4]
        print(f"  {f:14s} {n:4d}   e.g. {ex}")
