#!/usr/bin/env bash
# WHICH LAYER IS THE BEST HANDLE FOR THE LEARNED ADD-ON -- at a MATCHED DOSE.
#
# WHAT EXISTS ALREADY, AND WHY IT DOES NOT ANSWER THIS.
#   · pf_steer.py's FITTED direction has a full 9-layer sweep, judged at four of them
#     (RESULTS_0818_STEER.md §1/§5: L4 54.9, L12 61.8, L20 62.9, L28 54.4 on false_friend).
#   · The LEARNED add-on has a 9-layer sweep only under PREF=nll -- the objective later excluded as
#     length-confounded -- and only L20/L28 of that were ever judged.
#   · Under PREF=dpo, the objective behind RESULTS_0819_ZFINE.md, only L4, L12, L20 and L28 were
#     ever trained, and only L4 and L20 judged, both at z=1.0 -- which the z grid has since shown is
#     PAST THE KNEE (L4 at z=1.0: style 85.2 at coherence 52.6).
# So there is no layer curve for this objective at a dose anyone would ship.
#
# THE DOSE MUST BE MATCHED, WHICH NO PRIOR ADD-ON COMPARISON DID. ||A_L|| grows with depth (7.50 at
# L4 to 20.40 at L28) and so does the residual it is added to (R_L 6.16 -> 38.70, pf_resid_norms.py).
# A fixed z is therefore a different edit at every layer, and a depth curve read off one confounds
# "better handle" with "bigger perturbation". This matches the RELATIVE dose z*||A_L||/R_L to
# L20's value at z=0.70 -- the cell that matched DPOP on both families -- so every layer gets the
# same fractional edit to its own residual stream, exactly as pf_steer.py scales by alpha*R_L.
#
# CAVEAT ON RECORD: matching the relative dose is not the same as matching each layer at ITS OWN
# knee, and the knee may itself be layer-dependent. This buys one readable curve for one judge pass;
# a per-layer z grid is the more expensive follow-up if the curve turns out to be flat.
set -eu
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/.hf_home
PY=${PY:-/venv/main/bin/python}
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl

BANK=${BANK:-/workspace/reward-depth/results/probefix4b_addon_dpo}
REF_L=${REF_L:-20}; REF_Z=${REF_Z:-0.70}
LAYERS=${LAYERS:-4,8,12,16,20,24,28}

echo "===== train the missing dpo vectors (L4/L12/L20/L28 are banked and will be loaded) ====="
env LAYERS="$LAYERS" PREF=dpo STEPS=300 LR=1e-2 SEED=0 Z_GEN="" OUT="$BANK" "$PY" pf_addon.py

echo "===== residual norms (idempotent; needed to match the dose) ====="
[ -f /workspace/reward-depth/results/probefix/resid_norms.json ] || "$PY" pf_resid_norms.py

echo "===== per-layer z at the L${REF_L} z=${REF_Z} relative dose ====="
ZSPEC=$("$PY" - "$BANK" "$LAYERS" "$REF_L" "$REF_Z" <<'EOF'
import json, sys, torch
bank, layers, refL, refZ = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
R = json.load(open("/workspace/reward-depth/results/probefix/resid_norms.json"))
nrm = lambda L: float(torch.load(f"{bank}/addon_L{L}.pt", map_location="cpu")["A"].norm())
ref = refZ * nrm(refL) / R[str(refL)]            # the relative dose being matched
out = []
for L in [int(x) for x in layers.split(",")]:
    z = round(ref * R[str(L)] / nrm(L), 3)
    print(f"  L{L:<3d} |A| {nrm(L):6.2f}  R {R[str(L)]:7.2f}  ->  z {z:5.3f}", file=sys.stderr)
    out.append(f"{L}:{z}")
print(",".join(out))
EOF
)
echo "  spec: $ZSPEC"

echo "===== generate one cell per layer at its matched z ====="
for cell in ${ZSPEC//,/ }; do
  L=${cell%%:*}; Z=${cell##*:}
  env LAYERS="$L" PREF=dpo Z_GEN="$Z" SEED=0 N_PER_FAM=48 FAMS=false_friend,style \
      OUT="$BANK" "$PY" pf_addon.py
done

echo "===== leakage over the new cells ====="
"$PY" pf_leakage.py --all > /workspace/leakage_all.txt && grep addon /workspace/leakage_all.txt | head -20
echo LAYERSDONE
