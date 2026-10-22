# Regression checks

From the repo root:

    TSUKUYOMI_SAMPLES=/path/to/CubismWebSamples \
        xvfb-run -a -s "-screen 0 1024x1024x24" python -m pytest tests -v

- `test_runtime_units.py`, `test_motions.py`, `test_score.py`,
  `test_behaviors.py` are pure numpy: no GL, no torch, no sample rigs. They
  always run. `test_flowprior.py` needs torch and skips without it.
- `test_render.py` needs live2d-py, a display (or `xvfb-run` on PATH) and
  `TSUKUYOMI_SAMPLES` (the CubismWebSamples checkout, its `Samples/`, or
  `Samples/Resources/`; Haru and Hiyori are used). Without any of them those
  tests skip and say which is missing. Each GL check runs in its own process
  through `tests/gl_probe.py`, because a process can hold only one backend
  size; run `python tests/gl_probe.py physics <model3.json>` by hand to see
  the raw numbers a test judges.

What each render test guards, and the measurements its thresholds came from,
is in the test docstrings.
