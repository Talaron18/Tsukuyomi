#!/usr/bin/env bash
# How far the breath moves a rig on screen, at gain 0, auto (the calibrated
# one, flow.runtime.breath_setting) and 1 (the full ParamBreath range, as
# before the calibration). Renders 8 s of blinks only (--no-idle: no head,
# body or gesture moves, physics and breath live) per gain and measures the
# figure's alpha-weighted centroid y and its top edge (sub-pixel), std and
# peak-to-peak in px over the frames after the first second. Gain 0 is the
# floor (physics settling); the difference between auto and 1 is what the
# calibration took away. Needs GL (xvfb-run).
#
#   RIG=lafei bash scripts/breath_check.sh            # -> runs/breath/lafei
#   RIG=lafei MODEL=path/to/lafei.model3.json SIZE=512 bash scripts/breath_check.sh
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
RIG=${RIG:-lafei}
MODELS=${MODELS:-live2d-models:CubismWebSamples}
SIZE=${SIZE:-512}
OUT=${OUT:-runs/breath/$RIG}
MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    IFS=: read -ra roots <<< "$MODELS"
    ex=(); for r in "${roots[@]}"; do [[ -e "$r" ]] && ex+=("$r"); done
    MODEL=$(find -L "${ex[@]}" -name "$RIG.model3.json" -not -name '*.probe.*' -print -quit)
fi
[[ -z "$MODEL" ]] && { echo "no $RIG.model3.json under $MODELS" >&2; exit 1; }
clips=(); [[ -f "clips2/$RIG.npz" ]] && clips=(--clips "clips2/$RIG.npz")
mkdir -p "$OUT"
for g in 0 auto 1; do
    xvfb-run -a -s "-screen 0 1024x1024x24" python -m flow.runtime \
        --model "$MODEL" --rig "rigs/$RIG.npz" "${clips[@]}" --events default \
        --no-idle --seconds 8 --size "$SIZE" --breath-gain "$g" \
        --out "$OUT/g_$g" > "$OUT/g_$g.log" 2>&1 || echo "!! render g=$g failed ($OUT/g_$g.log)" >&2
    grep -h "^breath:" "$OUT/g_$g.log"
    python - "$OUT/g_$g" "$g" <<'PY'
import glob, sys
import numpy as np
from PIL import Image
d, g = sys.argv[1], sys.argv[2]
cy, top = [], []
for f in sorted(glob.glob(d + "/f*.png"))[30:]:
    a = np.asarray(Image.open(f))[..., 3].astype(np.float64) / 255
    rows = a.sum(1)
    y = np.arange(len(rows))
    cy.append((rows * y).sum() / rows.sum())
    c = np.cumsum(rows) / rows.sum()
    top.append(np.interp(0.005, c, y))
cy, top = np.array(cy), np.array(top)
print(f"gain {g:>4}: centroid y std {cy.std():.3f} px (ptp {np.ptp(cy):.3f}), "
      f"top edge std {top.std():.3f} px (ptp {np.ptp(top):.3f})")
PY
done | tee "$OUT/summary.txt"
