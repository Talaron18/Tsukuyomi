#!/usr/bin/env python3
"""
A small glTF 2.0 / GLB reader, enough for a VRM avatar's rest geometry.

flow.vrmbody needs, from a .vrm (a GLB with the VRM extension): the node
tree with its rest transforms, the humanoid bone -> node table, mesh
vertices skinned to their rest positions with their dominant joint, morph
target names, and the VRM metadata (expressions, look-at, meta / licence).
pygltflib would do the parsing but not the skinning; this is numpy and
the standard library only, so a CPU server needs nothing new.

Two VRM versions exist and differ in more than names:

  VRM 1.0  extensions.VRMC_vrm: humanoid.humanBones {bone: {node}},
           expressions.preset / .custom {name: {morphTargetBinds, ...}},
           lookAt {type, rangeMap*}, meta {licenseUrl, allowRedistribution,
           ...}. The avatar faces +Z.
  VRM 0.x  extensions.VRM: humanoid.humanBones [{bone, node}],
           blendShapeMaster.blendShapeGroups [{name, presetName, binds}],
           firstPerson.lookAt* {xRange, yRange}, meta {licenseName, ...}.
           The avatar faces -Z (Unity's export); three-vrm turns it round
           (VRMUtils.rotateVRM0), so vrm_info reports `faces` and flow.vrmbody
           rotates its geometry by 180 degrees about Y to the 1.0 frame.

vrm_info normalises both to the 1.0 vocabulary: VRM 0 presets a/i/u/e/o ->
aa/ih/ou/ee/oh, blink_l/r -> blinkLeft/Right, joy -> happy, fun -> relaxed,
sorrow -> sad (three-vrm's own mapping); a VRM 0 group with preset
"unknown" is a custom expression under its name, as three-vrm loads it.

Usage:
    python -m flow.gltf assets/vrm/VRM1_Constraint_Twist_Sample.vrm
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np

GLB_MAGIC = 0x46546C67           # "glTF"
CHUNK_JSON, CHUNK_BIN = 0x4E4F534A, 0x004E4942
COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
             5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9,
         "MAT4": 16}
# VRM 0.x preset names -> VRM 1.0 (three-vrm's VRMExpressionLoaderPlugin
# v0v1PresetNameMap).
V0_PRESETS = {"a": "aa", "i": "ih", "u": "ou", "e": "ee", "o": "oh",
              "blink": "blink", "blink_l": "blinkLeft",
              "blink_r": "blinkRight", "joy": "happy", "angry": "angry",
              "sorrow": "sad", "fun": "relaxed", "neutral": "neutral",
              "lookup": "lookUp", "lookdown": "lookDown",
              "lookleft": "lookLeft", "lookright": "lookRight"}


class GLTF:
    """A parsed .glb / .vrm: `json`, `bin` and accessors as numpy."""

    def __init__(self, data: bytes):
        magic, version, length = struct.unpack_from("<III", data, 0)
        if magic != GLB_MAGIC:
            raise ValueError("not a GLB file (a .vrm is one)")
        if version != 2:
            raise ValueError(f"glTF version {version}; only 2 is read")
        off, self.json, self.bin = 12, None, b""
        while off < min(length, len(data)):
            n, kind = struct.unpack_from("<II", data, off)
            chunk = data[off + 8:off + 8 + n]
            if kind == CHUNK_JSON:
                self.json = json.loads(chunk.decode("utf-8"))
            elif kind == CHUNK_BIN and not self.bin:
                self.bin = chunk
            off += 8 + n + (-n % 4)
        if self.json is None:
            raise ValueError("GLB without a JSON chunk")
        self._world = None

    @classmethod
    def load(cls, path: str) -> "GLTF":
        with open(path, "rb") as fh:
            return cls(fh.read())

    # --------------------------------------------------------- accessors

    def accessor(self, i: int) -> np.ndarray:
        """Accessor i as float64 (normalised integers scaled to [0, 1] or
        [-1, 1] as glTF says), shape (count, ncomp)."""
        a = self.json["accessors"][i]
        dt = np.dtype(COMPONENT[a["componentType"]]).newbyteorder("<")
        nc, count = NCOMP[a["type"]], a["count"]
        if "bufferView" in a:
            bv = self.json["bufferViews"][a["bufferView"]]
            if bv.get("buffer", 0) != 0:
                raise ValueError("only the GLB's own buffer is read")
            start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            stride = bv.get("byteStride") or dt.itemsize * nc
            raw = np.frombuffer(self.bin, np.uint8, count=(count - 1) * stride
                                + dt.itemsize * nc, offset=start)
            idx = (np.arange(count)[:, None] * stride
                   + np.arange(nc)[None, :] * dt.itemsize)
            b = raw[(idx[..., None] + np.arange(dt.itemsize)).reshape(-1)]
            out = b.view(dt).reshape(count, nc).astype(np.float64)
        else:
            out = np.zeros((count, nc))
        if a.get("normalized"):
            out /= float(np.iinfo(dt).max)
            if np.issubdtype(dt, np.signedinteger):
                out = np.maximum(out, -1.0)
        sp = a.get("sparse")
        if sp:
            ii = self._raw(sp["indices"], sp["count"], 1,
                           sp["indices"]["componentType"]).astype(int)[:, 0]
            vv = self._raw(sp["values"], sp["count"], nc,
                           a["componentType"])
            out[ii] = vv
        return out

    def _raw(self, ref, count, nc, ctype):
        dt = np.dtype(COMPONENT[ctype]).newbyteorder("<")
        bv = self.json["bufferViews"][ref["bufferView"]]
        start = bv.get("byteOffset", 0) + ref.get("byteOffset", 0)
        return np.frombuffer(self.bin, dt, count=count * nc,
                             offset=start).reshape(count, nc).astype(
                                 np.float64)

    # ------------------------------------------------------------- nodes

    def local_matrix(self, i: int) -> np.ndarray:
        n = self.json["nodes"][i]
        if "matrix" in n:
            return np.array(n["matrix"], float).reshape(4, 4).T
        t = np.array(n.get("translation", [0, 0, 0]), float)
        q = np.array(n.get("rotation", [0, 0, 0, 1]), float)
        s = np.array(n.get("scale", [1, 1, 1]), float)
        m = np.eye(4)
        m[:3, :3] = quat_matrix(q) * s[None, :]
        m[:3, 3] = t
        return m

    def parents(self) -> dict:
        out = {}
        for i, n in enumerate(self.json.get("nodes", [])):
            for c in n.get("children", []):
                out[c] = i
        return out

    def world(self) -> list:
        """World matrix of every node at rest (the file's own transforms)."""
        if self._world is None:
            nodes = self.json.get("nodes", [])
            par = self.parents()
            W = [None] * len(nodes)

            def get(i):
                if W[i] is None:
                    L = self.local_matrix(i)
                    W[i] = L if i not in par else get(par[i]) @ L
                return W[i]
            for i in range(len(nodes)):
                get(i)
            self._world = W
        return self._world

    def node_position(self, i: int) -> np.ndarray:
        return self.world()[i][:3, 3].copy()

    # ------------------------------------------------------------ meshes

    def skinned_vertices(self):
        """[(mesh name, positions (n, 3) world at rest, dominant joint node
        (n,) or -1, morph target names)] for every mesh node. Skinned
        vertices are posed by their joints' rest world matrices times the
        inverse bind matrices (an avatar exported in its bind pose gives the
        bind positions back); unskinned ones by their node."""
        W = self.world()
        out = []
        for ni, n in enumerate(self.json.get("nodes", [])):
            if "mesh" not in n:
                continue
            mesh = self.json["meshes"][n["mesh"]]
            names = (mesh.get("extras") or {}).get("targetNames") or []
            skin = self.json["skins"][n["skin"]] if "skin" in n else None
            if skin is not None:
                joints = skin["joints"]
                ibm = (self.accessor(skin["inverseBindMatrices"])
                       .reshape(-1, 4, 4).transpose(0, 2, 1)
                       if "inverseBindMatrices" in skin
                       else np.tile(np.eye(4), (len(joints), 1, 1)))
                J = np.stack([W[j] for j in joints]) @ ibm
            for prim in mesh.get("primitives", []):
                at = prim.get("attributes", {})
                if "POSITION" not in at:
                    continue
                P = self.accessor(at["POSITION"])
                Ph = np.c_[P, np.ones(len(P))]
                if skin is not None and "JOINTS_0" in at and \
                        "WEIGHTS_0" in at:
                    ji = self.accessor(at["JOINTS_0"]).astype(int)
                    wt = self.accessor(at["WEIGHTS_0"])
                    wt = wt / np.maximum(wt.sum(1, keepdims=True), 1e-12)
                    M = np.einsum("nk,nkij->nij", wt, J[ji])
                    X = np.einsum("nij,nj->ni", M, Ph)[:, :3]
                    dom = np.array(joints)[ji[np.arange(len(ji)),
                                              wt.argmax(1)]]
                else:
                    X = (W[ni] @ Ph.T).T[:, :3]
                    dom = np.full(len(P), -1)
                out.append((mesh.get("name", f"mesh{n['mesh']}"), X, dom,
                            list(names)))
        return out


def quat_matrix(q) -> np.ndarray:
    """3x3 rotation of a unit quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


# ------------------------------------------------------------------- VRM

def vrm_info(g: GLTF) -> dict:
    """The VRM extension in one vocabulary (the 1.0 one): {version "0" |
    "1", faces "+z" | "-z", humanoid {bone: node}, meta (as stored),
    license {name, url, redistribution, modification, credit} (normalised
    for the README), expressions {name: {"binds": [(mesh name, morph
    index, weight)], "preset": bool, "override": {blink, lookAt, mouth}}},
    look_at {type "bone" | "expression", h (deg of input at full), v_up,
    v_down, and the output at full: h_out / h_in (outer / inner eye
    corner), v_up_out, v_down_out -- degrees of eye rotation for "bone",
    expression weights for "expression"}, springs (count of spring
    chains)}."""
    ext = g.json.get("extensions") or {}
    meshes = g.json.get("meshes", [])
    nodes = g.json.get("nodes", [])
    if "VRMC_vrm" in ext:
        v = ext["VRMC_vrm"]
        hb = {b: e["node"] for b, e in v["humanoid"]["humanBones"].items()}
        exprs = {}
        for kind in ("preset", "custom"):
            for name, e in (v.get("expressions", {}).get(kind) or {}).items():
                binds = []
                for b in e.get("morphTargetBinds", []):
                    nd = nodes[b["node"]]
                    mname = meshes[nd["mesh"]].get("name") if "mesh" in nd \
                        else None
                    binds.append((mname, int(b["index"]),
                                  float(b.get("weight", 1.0))))
                exprs[name] = {"binds": binds, "preset": kind == "preset",
                               "override": {k: e.get("override" + k.title())
                                            for k in ("blink", "lookAt",
                                                      "mouth")},
                               "materials": len(e.get("materialColorBinds",
                                                      []))}
        la = v.get("lookAt", {})

        def rng(key, field="inputMaxValue"):
            r = la.get(key) or {}
            # The spec's defaults: input 90 deg; output 10 deg of eye
            # rotation for a bone look-at, weight 1 for an expression one.
            dflt = 90.0 if field == "inputMaxValue" else (
                10.0 if la.get("type", "bone") == "bone" else 1.0)
            return float(r.get(field, dflt))
        look = {"type": la.get("type", "bone"),
                "h": rng("rangeMapHorizontalOuter"),
                "v_up": rng("rangeMapVerticalUp"),
                "v_down": rng("rangeMapVerticalDown"),
                "h_out": rng("rangeMapHorizontalOuter", "outputScale"),
                "h_in": rng("rangeMapHorizontalInner", "outputScale"),
                "v_up_out": rng("rangeMapVerticalUp", "outputScale"),
                "v_down_out": rng("rangeMapVerticalDown", "outputScale"),
                "offset": la.get("offsetFromHeadBone", [0, 0, 0])}
        m = v.get("meta", {})
        lic = {"name": "VRM Public License 1.0" if "vrm.dev/licenses/1.0"
               in str(m.get("licenseUrl", "")) else m.get("licenseUrl"),
               "url": m.get("licenseUrl"),
               "redistribution": bool(m.get("allowRedistribution")),
               "modification": m.get("modification"),
               "credit": m.get("creditNotation"),
               "authors": m.get("authors"),
               "copyright": m.get("copyrightInformation"),
               "title": m.get("name")}
        springs = len((ext.get("VRMC_springBone") or {}).get("springs", []))
        return {"version": "1", "faces": "+z", "humanoid": hb, "meta": m,
                "license": lic, "expressions": exprs, "look_at": look,
                "springs": springs}
    if "VRM" in ext:
        v = ext["VRM"]
        hb = {e["bone"]: e["node"] for e in v["humanoid"]["humanBones"]
              if "node" in e and e["node"] >= 0}
        # VRM 0 binds name a mesh by index, the 1.0 ones a node.
        exprs = {}
        for grp in (v.get("blendShapeMaster") or {}).get(
                "blendShapeGroups", []):
            pre = str(grp.get("presetName", "unknown")).lower()
            name = V0_PRESETS.get(pre) if pre != "unknown" else None
            preset = name is not None
            name = name or grp.get("name")
            if not name:
                continue
            # VRM 0 weights are 0..100.
            binds = [(meshes[b["mesh"]].get("name"), int(b["index"]),
                      float(b.get("weight", 100.0)) / 100.0)
                     for b in grp.get("binds", [])
                     if 0 <= b.get("mesh", -1) < len(meshes)]
            exprs[name] = {"binds": binds, "preset": preset,
                           "override": {}, "materials": len(
                               grp.get("materialValues", []))}
        fp = v.get("firstPerson") or {}

        kind = str(fp.get("lookAtTypeName", "Bone")).lower() \
            .replace("blendshape", "expression")

        def rng0(key, field="xRange"):
            # VRM 0's curve maps xRange degrees of input to yRange of
            # output (degrees for a bone look-at, a weight for blend
            # shapes); UniVRM's defaults where a file leaves them out.
            dflt = 90.0 if field == "xRange" else (
                10.0 if kind == "bone" else 1.0)
            return float((fp.get(key) or {}).get(field, dflt))
        look = {"type": kind,
                "h": rng0("lookAtHorizontalOuter"),
                "v_up": rng0("lookAtVerticalUp"),
                "v_down": rng0("lookAtVerticalDown"),
                "h_out": rng0("lookAtHorizontalOuter", "yRange"),
                "h_in": rng0("lookAtHorizontalInner", "yRange"),
                "v_up_out": rng0("lookAtVerticalUp", "yRange"),
                "v_down_out": rng0("lookAtVerticalDown", "yRange"),
                "offset": [(fp.get("firstPersonBoneOffset") or {}).get(k, 0)
                           for k in "xyz"]}
        m = v.get("meta", {})
        other = m.get("otherPermissionUrl") or m.get("otherLicenseUrl") or ""
        lic = {"name": m.get("licenseName"), "url": other or None,
               "redistribution": "redistribution=allow" in other
               or m.get("licenseName") in ("CC0", "CC_BY", "CC_BY_SA"),
               "modification": "allow" if "modification=allow" in other
               else None,
               "credit": "unnecessary" if "credit=unnecessary" in other
               else None,
               "authors": [m.get("author")] if m.get("author") else None,
               "copyright": None, "title": m.get("title")}
        springs = len((v.get("secondaryAnimation") or {}).get("boneGroups",
                                                              []))
        return {"version": "0", "faces": "-z", "humanoid": hb, "meta": m,
                "license": lic, "expressions": exprs, "look_at": look,
                "springs": springs}
    raise ValueError("no VRM extension (VRMC_vrm or VRM): not a VRM avatar")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    g = GLTF.load(argv[0])
    info = vrm_info(g)
    print(f"VRM {info['version']} (faces {info['faces']}), "
          f"{len(info['humanoid'])} humanoid bones, "
          f"{len(info['expressions'])} expressions, look-at "
          f"{info['look_at']['type']}, {info['springs']} spring chains")
    print("license: " + json.dumps(info["license"], ensure_ascii=False))
    print("expressions: " + ", ".join(
        f"{n}{'' if e['binds'] or e['materials'] else ' (empty)'}"
        for n, e in sorted(info["expressions"].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
