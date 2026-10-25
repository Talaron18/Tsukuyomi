# Regression checks

From the repo root:

    TSUKUYOMI_SAMPLES=/path/to/CubismWebSamples \
        xvfb-run -a -s "-screen 0 1024x1024x24" python -m pytest tests -v

- `test_runtime_units.py`, `test_motions.py`, `test_score.py`,
  `test_behaviors.py`, `test_events.py`, `test_director.py`,
  `test_speech.py`, `test_voicevox.py`, `test_mouth.py`, `test_live.py`,
  `test_pose.py` are pure numpy: no GL, no torch, no sample rigs. They
  always run (`test_live.py` starts its web server on an ephemeral
  127.0.0.1 port; one of its checks replays Hiyori through the stream when
  `TSUKUYOMI_SAMPLES` is set). `test_flowprior.py` needs torch and skips without it.
- `test_render.py` needs live2d-py, a display (or `xvfb-run` on PATH) and
  `TSUKUYOMI_SAMPLES` (the CubismWebSamples checkout, its `Samples/`, or
  `Samples/Resources/`; Haru and Hiyori are used). Without any of them those
  tests skip and say which is missing. Each GL check runs in its own process
  through `tests/gl_probe.py`, because a process can hold only one backend
  size; run `python tests/gl_probe.py physics <model3.json>` by hand to see
  the raw numbers a test judges.

What each render test guards, and the measurements its thresholds came from,
is in the test docstrings.

`test_vrm.py` (E3, VRM avatars) builds tiny VRM 0 / 1.0 files in the test
and uses the shipped samples in assets/vrm; its Chromium checks (three-vrm
against Python's kinematics, an offline render) need Playwright with
Chromium (`pip install playwright && playwright install chromium`; browsers found
wherever Playwright looks, /opt/pw-browsers used when it exists and
PLAYWRIGHT_BROWSERS_PATH is unset) and skip without.

Stage 2 (`flow/train/`, `flow/actionhead.py`): `test_train_chat.py` is pure
Python (its Ollama-template check needs `go` and skips without);
`test_train_eval.py` is pure numpy; `test_train_sft.py`, `test_train_hf.py`
and `test_actionhead.py` need torch, transformers, peft and tokenizers
(CPU builds are enough: a tiny random Qwen3 is built in the test) and skip
without them. The 4-bit / Unsloth paths need a CUDA GPU and are not tested.
