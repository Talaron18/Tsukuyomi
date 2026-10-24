# Embodiment: a body the model knows and controls

Status: design, 2026-10. Stage 1 (tags → events → motion, with voice and a
live loop) works. This document is the plan for what comes after it.

## Why tags are not enough

Today the language model picks symbols from a menu (`[nod]`, `[tilt:left]`,
`[gesture:touch_head]`). `flow/events.py` turns each into a planned
minimum-jerk move to a target taken from the rig's fitted statistics, held,
then released. There is no feedback: nothing compares what is rendered with
what was meant.

A stage-2 model trained to pair words with these events would still only
learn "word ↔ symbol". It would not know that a head exists, where it is,
how far it can turn, what the body looks like right now, or what a raised
right hand is in space. A vision-language model can reason about spatial
geometry; tags throw that ability away.

The goal is to let the model control a body it has a model of: joints and
regions with states, limits and positions, seen from the outside as well
(its own rendered image), the way a robot policy controls an arm. Live2D is
an easy case for this (no contact, no dynamics beyond decorative physics),
and the same body description should carry over to 3D avatars.

## 1. One body description for every model

A canonical body state, shared by Live2D and 3D, that the model thinks in:

- head yaw / pitch / roll, gaze (x, y), eyelids;
- spine / chest / hips orientation and lean;
- per arm: shoulder, elbow, wrist; per hand: a small set of hand poses;
- face: expression shapes (ARKit's 52 blendshapes or FACS action units as
  the vocabulary; a rig drives the subset it has);
- positions of end effectors (hands, head) in body coordinates, in units of
  body size (e.g. shoulder width), so targets read like "right hand in front
  of the chest, at shoulder height".

Each rig declares which parts it has and their limits. A part the rig lacks
is absent, not faked (the same rule as the director's offered tags).

**3D (VRM).** VRM defines a humanoid skeleton and expression blendshapes, so
the mapping is close to direct; inverse kinematics turns end-effector
targets into joint angles.

**Live2D.** Parameters are artist-defined deformers, not joints, so the
mapping is measured, not declared:

- Set one parameter at a time across its range and record the geometry it
  produces. `l2dprobe` already does this on rendered images (`out/`). The
  Cubism core also computes every drawable's mesh vertices on the CPU, so the
  measurement can use exact geometry instead of pixels: per parameter, how
  each body region's vertices move (a forward model, parameters → region
  positions/orientations).
- Body regions come from drawable / part ids and names (the same hints
  `flow/gestures.py` and `flow/behaviors.py` use), checked against the
  measured motion.
- Inverting the forward model (least squares with limits) gives the solver
  "put the right hand here → set these parameters". Switch-like parameters
  (arm pose swaps) become discrete choices the solver can make; whole-model
  transforms stay held.

## 2. Self-perception

The model gets what it currently looks like: a rendered frame, the current
canonical body state, and its limits. It can then check its own output ("did
the hand go up?") and correct, which closes the loop the event pipeline lacks.
Live2D makes this cheap: the live loop renders at 30 fps in about 10 ms a
frame on the server's CPU.

## 3. Control: a VLM plus an action model

The π0 pattern: a VLM backbone reads the conversation (and optionally its
own image); a separate action model, conditioned on the VLM's hidden states,
outputs short chunks of canonical body-state trajectory, continuous, not
symbolic. `flow/legacy/flowprior.py` is already a flow-matching chunk model;
it failed as an unconditioned idle generator, but as a conditioned action
expert it has the right shape. The idle, the director's tags and the
retargeting below all remain useful as fallbacks and as training signal.

## Training data

Retarget into the canonical description:

- **BEAT2, TalkSHOW**: speech paired with co-speech gesture and face (SMPL-X),
  i.e. words ↔ how the body moves, with real geometry.
- **HumanML3D, Motion-X**: text descriptions paired with full-body motion
  (Motion-X with face and hands).
- **Our rigs' authored clips** (40 Azur Lane rigs, Cubism samples), once
  mapped through the Live2D forward model.

Related work to learn from: MotionGPT (motion as language tokens),
RT-2 / OpenVLA (actions as tokens), π0 (flow-matching action expert).

## Stages

| Step | What | Training |
|---|---|---|
| E1 | Canonical body description; Live2D geometry from mesh vertices; the solver from body targets to parameters for Hiyori, lafei, xuefeng | none |
| E2 | Test without training: the API model gets its body description, limits and its own image, and writes continuous body targets (JSON) instead of tags; E1's solver plays them; compare with tags in the live loop | none |
| E3 | A VRM avatar driven by the same body targets (IK + blendshapes), side by side with Live2D | none |
| E4 | Retarget BEAT2 / HumanML3D into the canonical description; train the action model on a small local VLM's hidden states | GPU needed |

E1 to E3 run on the current CPU server and decide whether E4 is worth renting
a GPU for. If E2 shows an API model can direct its own body from a description
plus a picture, the action model mainly buys latency and smoothness; if not,
training is the route.

E1 also helps the current pipeline: gesture descriptions can say "raises the
right hand to face height" from measurement instead of naming parameters.
