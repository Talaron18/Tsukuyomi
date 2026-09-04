"""
Produce a probe-safe copy of a .model3.json.

Disabling auto-blink and auto-breath is not enough to make renders
deterministic. LAppModel.Update() also steps:

    physics3.json   time-integrated spring system, driven BY parameters.
                    Probing ParamAngleX makes hair swing, and the swing
                    lags, so the rendered result depends on frame history.
    pose3.json      part opacity crossfades over wall-clock time.
    motions         any idle motion auto-started on load.
    expressions     additive parameter offsets with their own fade timing.

All four break the "same input, same pixels" property the probe depends on,
and physics in particular corrupts descriptors: the diff for a head parameter
would include hair displacement that belongs to a different parameter.

We can't turn these off through the binding, so we remove them from the model
definition and load the stripped copy. The copy is written alongside the
original so relative paths to the .moc3 and textures still resolve; if that
directory is read-only we mirror it into a temp dir instead.

Note this is only for probing. Physics is a real part of how the rig looks in
motion and the runtime must keep it on. The motion model learns to drive
parameters; Live2D's physics remains responsible for turning that into
secondary motion at playback time.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

# Removed from FileReferences before loading.
DYNAMIC_KEYS = ("Physics", "Pose", "Motions", "Expressions")


class ProbeModel:
    """Context manager yielding a path to a stripped .model3.json."""

    def __init__(self, model_json: str, strip: bool = True):
        self.original = model_json
        self.strip = strip
        self.path = model_json
        self.removed: list[str] = []
        self._tempdir: str | None = None
        self._tempfile: str | None = None

    def __enter__(self) -> "ProbeModel":
        if not self.strip:
            return self
        with open(self.original, encoding="utf-8") as fh:
            spec = json.load(fh)

        refs = spec.get("FileReferences", {})
        for key in DYNAMIC_KEYS:
            if refs.pop(key, None) is not None:
                self.removed.append(key)
        spec["FileReferences"] = refs

        if not self.removed:
            return self  # nothing dynamic; load the original untouched

        src_dir = os.path.dirname(os.path.abspath(self.original)) or "."
        base = os.path.basename(self.original).replace(".model3.json", "")
        target_dir = src_dir

        if not os.access(src_dir, os.W_OK):
            # Read-only source: mirror the whole model directory so the .moc3
            # and texture paths inside the json still resolve relatively.
            self._tempdir = tempfile.mkdtemp(prefix="l2dprobe_")
            target_dir = os.path.join(self._tempdir, os.path.basename(src_dir))
            shutil.copytree(src_dir, target_dir)

        self._tempfile = os.path.join(target_dir, f"{base}.probe.model3.json")
        with open(self._tempfile, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, ensure_ascii=False)
        self.path = self._tempfile
        return self

    def __exit__(self, *exc):
        if self._tempdir:
            shutil.rmtree(self._tempdir, ignore_errors=True)
        elif self._tempfile and os.path.exists(self._tempfile):
            os.remove(self._tempfile)
        return False
