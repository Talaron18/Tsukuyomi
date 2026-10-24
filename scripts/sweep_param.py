"""Sweep one parameter through the Cubism Core and print what it does to the
mesh: per step, how far each affected drawable's vertices move from rest and
how its opacity changes. Tells a continuous deformer (vertices move smoothly
with the value) from an art swap (opacities jump between two drawings), and
prints the verdict of l2dprobe.core.continuous_deformer, the test flow.runtime
and flow.body use to overrule a probe that called an eyelid discrete.

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
    sw = core.sweep(m, a.param, a.steps)
    vals, move, opac = sw["values"], sw["move"], sw["opacity"]
    rest_o = sw["rest_opacity"]
    ok, why = core.continuous_deformer(m, a.param, a.steps)
    hit = np.argsort(-(move.max(0) + np.abs(opac - rest_o).max(0)))[:a.top]
    hit = [k for k in hit if move[:, k].max() > 1e-6
           or np.abs(opac[:, k] - rest_o[k]).max() > 1e-6]
    if not hit:
        print("no drawable moves or changes opacity")
        print(f"verdict: not a pure deformer ({why})")
        return 0
    print("value  " + "  ".join(f"{m.drawable_ids[k][:14]:>14}" for k in hit))
    for s, v in enumerate(vals):
        print(f"{v:5.2f}  " + "  ".join(
            f"{move[s, k]:7.4f}/{opac[s, k]:4.2f}"[:14].rjust(14) for k in hit))
    print("cells: max vertex shift from rest (model units) / opacity")
    print(f"verdict: {'continuous deformer' if ok else 'not a pure deformer'}"
          f" ({why}; all parts shown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
