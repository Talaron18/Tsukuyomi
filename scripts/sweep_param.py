"""Sweep one parameter through the Cubism Core and print what it does to the
mesh: per step, how far each affected drawable's vertices move from rest and
how its opacity changes. Tells a continuous deformer (vertices move smoothly
with the value) from an art swap (opacities jump between two drawings).

    python scripts/sweep_param.py --model-json M ParamEyeLOpen [--steps 11]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from l2dprobe import core  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-json", required=True)
    ap.add_argument("param")
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--top", type=int, default=12,
                    help="drawables listed (the most affected)")
    a = ap.parse_args(argv)
    m = core.load(a.model_json)
    if a.param not in m.param_ids:
        sys.exit(f"{a.param} not in the model; similar: "
                 + ", ".join(p for p in m.param_ids
                             if a.param.lower()[:6] in p.lower()))
    i = m.param_ids.index(a.param)
    lo, hi = float(m.param_min[i]), float(m.param_max[i])
    d0 = float(m.param_default[i])
    print(f"{a.param}: range [{lo:g}, {hi:g}], default {d0:g}")
    m.set({})
    rest_v = [m.vertices(k).copy() for k in range(len(m.drawable_ids))]
    rest_o = m.opacities().copy()
    vals = np.linspace(lo, hi, a.steps)
    move = np.zeros((a.steps, len(rest_v)))
    opac = np.zeros((a.steps, len(rest_v)))
    for s, v in enumerate(vals):
        m.set({a.param: float(v)})
        for k in range(len(rest_v)):
            move[s, k] = np.abs(m.vertices(k) - rest_v[k]).max()
        opac[s] = m.opacities()
    hit = np.argsort(-(move.max(0) + np.abs(opac - rest_o).max(0)))[:a.top]
    hit = [k for k in hit if move[:, k].max() > 1e-6
           or np.abs(opac[:, k] - rest_o[k]).max() > 1e-6]
    if not hit:
        print("no drawable moves or changes opacity")
        return 0
    print("value  " + "  ".join(f"{m.drawable_ids[k][:14]:>14}" for k in hit))
    for s, v in enumerate(vals):
        print(f"{v:5.2f}  " + "  ".join(
            f"{move[s, k]:7.4f}/{opac[s, k]:4.2f}"[:14].rjust(14) for k in hit))
    print("cells: max vertex shift from rest (model units) / opacity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
