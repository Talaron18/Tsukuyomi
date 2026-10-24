// A VRM avatar driven by the server's body states (flow.vrmbody state()).
//
// The server sends, per frame, the local rotation of each humanoid bone in
// three-vrm's normalized rig (rest = identity, VRM 1.0 frame: the avatar
// faces +Z, its left is +X), the look-at yaw / pitch, expression weights and
// morph weights. This module loads the avatar, applies a state and renders;
// live (vrm.html) and offline (render.html) pages share it, so what the
// offline renderer captures is what the live page draws.
//
// VRM 0 avatars face -Z; VRMUtils.rotateVRM0 turns the scene round, and a
// rotation meant for the 1.0 frame is applied with x and z negated (the
// conjugation by that half turn; three-vrm's own examples do the same for
// VRM 0 animation). Look-at angles need no change: three-vrm measures them
// from the avatar's own front (faceFront), checked in tests/test_vrm.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

export async function loadAvatar(url, canvas, camCfg, opts = {}) {
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, alpha: false,
    // The offline renderer reads the canvas back after each render.
    preserveDrawingBuffer: !!opts.preserve,
  });
  renderer.setPixelRatio(1);
  renderer.setSize(canvas.width, canvas.height, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const bg = opts.bg || [0.12, 0.12, 0.14];
  renderer.setClearColor(new THREE.Color(bg[0], bg[1], bg[2]), 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(camCfg.fov, canvas.width / canvas.height, 0.05, 50);
  camera.position.set(...camCfg.position);
  camera.lookAt(new THREE.Vector3(...camCfg.target));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(0.6, 1.4, 2.0);
  scene.add(key);
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const loader = new GLTFLoader();
  loader.register(parser => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error("not a VRM: " + url);
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  const v0 = vrm.meta && vrm.meta.metaVersion === "0";
  vrm.scene.traverse(o => { o.frustumCulled = false; });
  scene.add(vrm.scene);
  if (vrm.lookAt) { vrm.lookAt.target = null; vrm.lookAt.autoUpdate = false; }
  // Morphs by name (the body file's face map writes VRoid's per-feature
  // shapes, which no expression binds, directly).
  const morphs = {};
  vrm.scene.traverse(o => {
    if (o.morphTargetDictionary) {
      for (const [n, i] of Object.entries(o.morphTargetDictionary)) {
        (morphs[n] = morphs[n] || []).push([o, i]);
      }
    }
  });
  return { vrm, renderer, scene, camera, v0, morphs, lastExpr: new Set(), lastMorph: new Set() };
}

// Apply one state and advance the avatar by dt seconds (springs, look-at,
// expressions, constraints all update in vrm.update).
export function applyState(av, st, dt) {
  const { vrm, v0 } = av;
  const h = vrm.humanoid;
  for (const [name, q] of Object.entries(st.bones || {})) {
    const n = h.getNormalizedBoneNode(name);
    if (n) n.quaternion.set(v0 ? -q[0] : q[0], q[1], v0 ? -q[2] : q[2], q[3]);
  }
  if (vrm.lookAt && st.look) { vrm.lookAt.yaw = st.look[0]; vrm.lookAt.pitch = st.look[1]; }
  const em = vrm.expressionManager;
  if (em) {
    const now = new Set();
    for (const [n, w] of Object.entries(st.expr || {})) {
      if (em.getExpression(n)) { em.setValue(n, w); now.add(n); }
    }
    for (const n of av.lastExpr) if (!now.has(n)) em.setValue(n, 0);
    av.lastExpr = now;
  }
  vrm.update(dt);
  // After vrm.update: the expression manager rewrites only the morphs its
  // expressions bind.
  const nowM = new Set();
  for (const [n, w] of Object.entries(st.morph || {})) {
    for (const [o, i] of av.morphs[n] || []) o.morphTargetInfluences[i] = w;
    nowM.add(n);
  }
  for (const n of av.lastMorph) if (!nowM.has(n)) for (const [o, i] of av.morphs[n] || []) o.morphTargetInfluences[i] = 0;
  av.lastMorph = nowM;
}

export function draw(av) { av.renderer.render(av.scene, av.camera); }

// States between two frames (the page draws at its own rate): bone
// rotations slerped, the rest linear. u in [0, 1].
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
export function lerpState(a, b, u) {
  if (!a) return b;
  if (!b || u <= 0) return a;
  if (u >= 1) return b;
  const out = { bones: {}, look: [0, 0], expr: {}, morph: {} };
  const names = new Set([...Object.keys(a.bones || {}), ...Object.keys(b.bones || {})]);
  for (const n of names) {
    const qa = (a.bones || {})[n] || [0, 0, 0, 1], qb = (b.bones || {})[n] || [0, 0, 0, 1];
    _qa.set(...qa); _qb.set(...qb); _qa.slerp(_qb, u);
    out.bones[n] = [_qa.x, _qa.y, _qa.z, _qa.w];
  }
  const la = a.look || [0, 0], lb = b.look || [0, 0];
  out.look = [la[0] + (lb[0] - la[0]) * u, la[1] + (lb[1] - la[1]) * u];
  for (const k of ["expr", "morph"]) {
    const ks = new Set([...Object.keys(a[k] || {}), ...Object.keys(b[k] || {})]);
    for (const n of ks) {
      const x = (a[k] || {})[n] || 0, y = (b[k] || {})[n] || 0;
      out[k][n] = x + (y - x) * u;
    }
  }
  return out;
}

// World positions (1.0 frame, metres) of normalized bones: the check that
// Python's forward kinematics and three-vrm agree.
export function positions(av, names) {
  const out = {};
  const v = new THREE.Vector3();
  av.vrm.scene.updateMatrixWorld(true);
  for (const n of names) {
    const node = av.vrm.humanoid.getNormalizedBoneNode(n);
    if (node) { node.getWorldPosition(v); out[n] = [v.x, v.y, v.z]; }
    const raw = av.vrm.humanoid.getRawBoneNode(n);
    if (raw) { raw.getWorldPosition(v); out["raw:" + n] = [v.x, v.y, v.z]; }
  }
  return out;
}

// World direction the left eye looks along, relative to its rest (the
// avatar's front, +Z): the look-at sign check. Call once at rest first.
export function eyeDir(av) {
  const e = av.vrm.humanoid.getRawBoneNode("leftEye");
  if (!e) return null;
  av.vrm.scene.updateMatrixWorld(true);
  const q = new THREE.Quaternion();
  e.getWorldQuaternion(q);
  if (!av.restEye) av.restEye = q.clone();
  const rel = q.clone().multiply(av.restEye.clone().invert());
  const d = new THREE.Vector3(0, 0, 1).applyQuaternion(rel);
  return [d.x, d.y, d.z];
}
