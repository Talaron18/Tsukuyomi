#!/usr/bin/env python3
"""
Diagnose the live2d-py / GL import chain.

    python -m l2dprobe.doctor

Prints the real traceback for each step instead of a guessed cause. Run this
whenever an import fails; the failing line tells us whether the problem is the
Python package, the bundled native extension, or the system GL libraries.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import traceback


def step(label: str, fn):
    print(f"\n--- {label} ---")
    try:
        result = fn()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        return None
    print("ok" + (f": {result}" if result else ""))
    return result


def _pkg_layout():
    import live2d
    root = os.path.dirname(live2d.__file__)
    print(f"package root: {root}")
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        tag = "dir " if os.path.isdir(path) else "file"
        print(f"  {tag} {entry}")
        # the native extension usually lives one level down
        if os.path.isdir(path) and entry in {"v2", "v3"}:
            for sub in sorted(os.listdir(path)):
                print(f"       {sub}")
    return None


def _ldd_extension():
    """If a .so is present, show which shared libs it cannot resolve."""
    import live2d
    root = os.path.dirname(live2d.__file__)
    sos = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith((".so", ".pyd", ".dylib")):
                sos.append(os.path.join(dirpath, f))
    if not sos:
        print("no native extension found in the package")
        return None
    for so in sos:
        print(f"\n{so}")
        if sys.platform.startswith("linux"):
            try:
                out = subprocess.run(["ldd", so], capture_output=True, text=True,
                                     timeout=20).stdout
            except Exception as exc:  # noqa: BLE001
                print(f"  ldd failed: {exc}")
                continue
            missing = [ln.strip() for ln in out.splitlines() if "not found" in ln]
            if missing:
                print("  UNRESOLVED:")
                for ln in missing:
                    print(f"    {ln}")
            else:
                print("  all shared libraries resolved")
    return None


def main() -> int:
    print(f"python     : {sys.version}")
    print(f"executable : {sys.executable}")

    step("import live2d", lambda: importlib.import_module("live2d").__file__)
    step("package layout", _pkg_layout)
    step("import live2d.v3", lambda: importlib.import_module("live2d.v3").__file__)
    step("import live2d.v2", lambda: importlib.import_module("live2d.v2").__file__)
    step("native extension linkage", _ldd_extension)
    def _gl_libs():
        """Report which system GL libraries the loader can actually find."""
        import ctypes.util
        for lib in ("GL", "EGL", "OSMesa", "glfw"):
            found = ctypes.util.find_library(lib)
            print(f"  lib{lib:<8} {found or 'NOT FOUND'}")
        print(f"  PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')!r}")
        return None

    step("system GL libraries", _gl_libs)
    step("import OpenGL.GL", lambda: importlib.import_module("OpenGL.GL").__name__)
    step("import glfw", lambda: importlib.import_module("glfw").__file__)

    def _glfw_backends():
        import glfw
        ver = glfw.get_version_string().decode(errors="replace")
        print(f"  build     : {ver}")
        print("  (platforms are X11/Wayland/Null; GLX/EGL/OSMesa are context APIs)")
        for name in ("PLATFORM_X11", "PLATFORM_WAYLAND", "PLATFORM_NULL",
                     "OSMESA_CONTEXT_API", "EGL_CONTEXT_API"):
            print(f"  {name:<20} {getattr(glfw, name, 'MISSING')}")
        return None

    step("glfw platforms / context APIs", _glfw_backends)
    print(f"\nDISPLAY={os.environ.get('DISPLAY')!r}  "
          f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
