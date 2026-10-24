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

## The baseline stays

The tag pipeline is not replaced; it is the baseline every embodiment step
is measured against, and the fallback while those steps are unproven:

- **Stage 1** (works now): API model writes tags; director, events, voice
  and the live loop play them.
- **Stage 2** (the original plan): a small local model maps its hidden
  states to the same event vocabulary, trained on words ↔ events. Much
  easier than E4 and may well reach good quality on its own.

Each embodiment step (E2 onwards) is judged against these with the same
tools: blind A/B on rendered clips (`flow/legacy/abtest.py`), the motion
scorer (`flow/score.py`), and latency in the live loop. A step that does not
beat the baseline does not replace it.

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

## E2 as built (2026-10)

`flow/pose.py`; `python -m flow.director --mode body` (or `--body-targets`)
offline, `python -m flow.live --mode body` live, `scripts/body_vs_tags.sh`
for the comparison.

- **Prompt**: generated from `rigs/<rig>.body.json` alone (byte-stable per
  file): parts, the character's own left and right, landmarks, every offered
  variable with its rest and reachable range (union over the drawn arm
  poses, so Hiyori's hands read "[-2.24, -1.66] or [-0.32, -0.02]"), reach
  per hand per drawn pose in words, and what cannot be done (lafei's left
  hand cannot reach shoulder height; Hiyori's hands cannot stop between the
  hanging and the raised drawing). No parameter ids. The pose now goes in
  each user message; a picture of the current frame too with
  `--body-image` on Anthropic, SiliconFlow VLMs or Ollama (DeepSeek chat is
  text only and gets none).
- **Output**: `[pose {"hand.R.y": "@shoulder.y", "head.tilt": 8}]` inline,
  optional `"dur"` / `"hold"`; plus `[blink]` and `[pause:x]`. Pure body:
  face shapes are canonical variables, a blink is the autonomic layer's
  transient, gestures would hand the motion back to authored clips (the
  baseline). Markers time like tags (words, moras, sentence streaming).
- **Play**: solved with the prior toward the pose the body is in; only
  what moves is played; minimum-jerk move, hold, release back into the
  idle, over the idle frame in both `runtime` and `stream`; arm variant
  parts swap at the midpoint; pose tokens take their role groups with
  `director.merge`'s semantics. Tag mode is unchanged byte for byte.
- **Feedback**: per target, the forward model's value and the Core's own
  geometry go to `actions.json` / the live log; `--body-check` runs one
  offline correction round (`check/check.json`).
