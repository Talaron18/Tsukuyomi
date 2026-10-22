"""
Shared fixtures for the regression checks.

The repo has no package metadata, so the root goes on sys.path here the same
way `python -m flow.X` puts it there. Render tests need three things this
machine may lack -- live2d-py that imports, a display (or xvfb-run), and the
Cubism sample rigs -- and skip, naming which one, when any is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GL_PROBE = os.path.join(ROOT, "tests", "gl_probe.py")


def _resources(base: str) -> str | None:
    """Accept the CubismWebSamples root, Samples/, or Samples/Resources/."""
    for cand in (os.path.join(base, "Samples", "Resources"),
                 os.path.join(base, "Resources"), base):
        if os.path.isdir(os.path.join(cand, "Haru")):
            return cand
    return None


@pytest.fixture(scope="session")
def samples() -> str:
    base = os.environ.get("TSUKUYOMI_SAMPLES")
    if not base:
        pytest.skip("set TSUKUYOMI_SAMPLES to the CubismWebSamples checkout")
    res = _resources(base)
    if res is None:
        pytest.skip(f"no Haru/ under TSUKUYOMI_SAMPLES={base}")
    return res


def sample_model(samples: str, name: str) -> str:
    path = os.path.join(samples, name, f"{name}.model3.json")
    if not os.path.exists(path):
        pytest.skip(f"sample rig {name} not found under {samples}")
    return path


@pytest.fixture(scope="session")
def gl_prefix() -> list[str]:
    """Command prefix that gives a child process a display, or skip."""
    probe = subprocess.run(
        [sys.executable, "-c", "import live2d.v3"], capture_output=True,
        text=True, cwd=ROOT)
    if probe.returncode != 0:
        pytest.skip("live2d-py does not import: "
                    + (probe.stderr.strip().splitlines() or ["?"])[-1])
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return []
    if shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", "-s", "-screen 0 1024x1024x24"]
    pytest.skip("no DISPLAY and no xvfb-run")


def run_child(prefix: list[str], args: list[str], timeout: float = 300
              ) -> subprocess.CompletedProcess:
    """Run one GL job in its own process: one backend size per process."""
    env = dict(os.environ, _L2D_XVFB="1")    # never re-exec inside the child
    proc = subprocess.run(prefix + [sys.executable] + args, capture_output=True,
                          text=True, cwd=ROOT, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(args)} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")
    return proc


def gl_probe(prefix: list[str], *args: str) -> dict:
    proc = run_child(prefix, [GL_PROBE, *args])
    return json.loads(proc.stdout.strip().splitlines()[-1])
