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
  optional `"dur"` / `"hold"`; plus `[blink]` and `[pause:x]`, and the
  rig's own motions (`[gesture:NAME]`, as tag mode offers them) unless
  `--no-body-gestures`. Face shapes are canonical variables, a blink is the
  autonomic layer's transient. A gesture gives way to a pose that moves
  an arm. Markers time like tags (words, moras, sentence streaming).
- **Prompt guidance** (after the first A/B, below): brief beats (a glance
  away and back, a nod as a quick pair), short holds unless stressing one
  moment, the hand that reaches highest named with how high it goes,
  variety from reply to reply, no held half-shut eyes (they stop the
  blinks).
- **Play**: solved with the prior toward the pose the body is in; only
  what moves is played; minimum-jerk move, hold, release back into the
  idle, over the idle frame in both `runtime` and `stream`; arm variant
  parts swap at the midpoint; a pose moving an arm keeps idle gestures
  from starting during it (`director.merge`), and the idle's head, gaze,
  brow and body moves run on under every pose. A hold is not frozen: it
  carries 0.35 of the idle's motion and settles 0.2 of the way back to it
  (`flow.pose.LIFE`, `SETTLE`). Tag mode is unchanged byte for byte.
- **Feedback**: per target, the forward model's value and the Core's own
  geometry go to `actions.json` / the live log; `--body-check` runs one
  offline correction round (`check/check.json`).

### First A/B (runs/e2) and what changed

Blind A/B on the server, 12 pairs (DeepSeek, Hiyori and lafei, 6 lines
each): a tie overall (body 46%, tags 54%); Hiyori body 2-1 with 3 ties,
lafei tags 3-1 with 2 ties. The body replies were long static holds (hold
2-3 s, 9 of 13 Hiyori markers with an explicit hold; head and gaze frozen
in 58% of the hold frames), face-only on lafei (2 of 12 poses moved a
hand), few quick moves; the tags gave quick nods and looks plus the rig's
own gestures. And on lafei the breath bobbed the whole figure (~3 px std
at 0.25 Hz). Changed for the second round (runs/e2b):

- breath calibrated per rig from the body file (`flow.runtime.
  breath_setting`): Hiyori 1, Natori 0.467, lafei 0.109, xuefeng 0.107 of
  ParamBreath's range, so it moves each body at most 0.0075 hu;
- held poses keep the idle's life (LIFE 0.35, SETTLE 0.2) and the idle's
  head/gaze/brow/body moves run on under poses; default hold 1.5 -> 1 s;
- the prompt guidance above, and gestures offered by default.


## E3 as built (2026-10)

`flow/gltf.py`, `flow/vrmbody.py`, `flow/vrm.py`, `flow/vrmrender.py`,
`flow/web/` (vrm.js, vrm.html, render.html, vendor/: three 0.180.0 and
@pixiv/three-vrm 3.5.5, MIT, sha256 in vendor/README.md). Sample avatars in
`assets/vrm/` (Seed-san, VRM 1.0, VRM Public License 1.0; three-vrm-girl,
VRM 0, VRoid Hub licence; both allow redistribution and modification, see
assets/vrm/README.md). Body files `rigs/seed-san.body.json`,
`rigs/three-vrm-girl.body.json`.

- **Body file** (`python -m flow.vrmbody --vrm PATH --out FILE`):
  tsukuyomi.body/2 with `"kind": "vrm"`, loaded by `flow.body.load` into a
  `VRMBody` with BodyModel's interface, so flow.pose's prompt, solve loop,
  reach report and play run unchanged. Frame as Live2D: origin the chin
  (lowest head-bone vertex in front), 1 hu = chin to top of skull (hair
  left out): 0.246 m on Seed-san; +x the character's left (VRM 0, facing
  -Z, is turned round), z toward the viewer. Parameters are the body's own
  joint angles (head/torso yaw-pitch-tilt split over neck+head and
  spine+chest+upperChest; per arm elevation, plane of elevation, twist,
  elbow bend; shoulders) and face weights in the Live2D standard ranges.
  Joint limits: AAOS active ranges cut to a talking avatar (head yaw +-50,
  pitch +-35, tilt +-30; torso +-25/15/12; elevation 0..170, plane
  -90..130 with the extension held to 60 hanging / 45 level / 30 overhead,
  twist -100..90 measured from the swing out of hanging (swing-twist: an
  Euler twist's zero turned with the plane, and behind the back needed
  -110 of it), elbow 0..145; a wrist on the hand bone, roll -20..160 from
  the palm-down T-pose, flexion -70..80, deviation -30..20).
- **Solver**: declared variables and joints set directly (clamped);
  a hand target is two-bone IK with a pole (elbow bend by bisection on the
  real bone vectors, swivel searched for the elbow nearest out-down-back
  within the limits and the current pose); a target given in x, y only
  keeps the palm's depth, pushed in front of the torso / face and within
  reach. ~3 ms per solve (two hands meeting ~30 ms, a detour ~0.2 s). The
  reverse is forward kinematics (`perceive`).
- **Collisions** (body file revision 2): the trunk, neck and head as one
  signed distance lofted through 0.2 hu slabs of the skinned rest mesh
  (hair left out), narrowed where the resting arms sank in (a loose
  shirt's sides are skinned to the chest); arms as capsules (upper arm past
  the shoulder, forearm, two across the palm), radii from their vertices.
  The swivel search orders limits, then penetration, then the pole. A
  target inside the body moves onto it; one reachable only through the
  body (or, one hand moving, the other arm) moves to the nearest clear
  target; both are MISS with the reason in the report. Two hands moving
  together are solved for the body each and their targets parted, half
  each, until they touch. When the straight joint-angle path (what the
  pose layer plays) crosses the body, the solver gives a via pose that
  flow.pose plays on the way in and before the release, and that pose
  holds still (no settle: drifting each angle toward the rest put the
  hands 0.32-0.43 hu into the hips by the end of a hold behind the back). Old solver vs now
  (deepest penetration, hu; Seed-san / three-vrm-girl): hands together
  0.08 / 0.07 into the chest and 0.085 / 0.09 into each other -> 0.018 /
  0.007; crossed forearms 0.20 / 0.18 through each other -> 0.02; hands
  behind the back (z given) 0.32 / 0.30 into the hips -> 0.01, and 0.41 /
  0.35 on the straight path -> 0.01 by the detour; hand on the other
  shoulder 0.11 -> 0.02. Tolerance 0.02 hu (5 mm).
- **Composite targets** (VRM only): `"hands": "@together"` (palm to palm
  before the chest, fingers up), `"@clasped"` (before the waist),
  `"@behind_back"`; `"hand.R": "@chest"` (or any landmark: the palm flat
  on the body there), `"@hip"`, `"@back"`. Hand positions plus a hand
  orientation the wrist turns to; the VRM prompt lists them. The Live2D
  prompt and parser have none (byte-identical): in `--avatar both` the
  reply is Hiyori's, so "hands together" arrives as two plain x, y targets
  and the VRM's hands meet side by side (reported), never through each
  other; head.pitch markers are converted from ParamAngleY units.
- **Kinematics = three-vrm's normalized rig**: the stream sends each
  bone's local quaternion; Python FK and three-vrm in Chromium agree to
  0.01 mm on both samples (tests/test_vrm.py). VRM 0: x and z of each
  quaternion negated in the page; three-vrm's look-at pitch is + down.
- **Face map** (per avatar, from what it has): VRoid's per-feature morphs
  where present (Fcl_EYE_Joy_L/R for eye smile, Fcl_BRW_Surprised / Angry
  / Fun for brow lift +/- and form +/-, Fcl_MTH_Fun / Sorrow for mouth
  form), else the emotion presets at reduced gain (happy 0.5 eye smile,
  surprised / angry / relaxed for the brows, happy / sad 0.4 for the
  mouth). Limits: VRoid brows are one shape for both sides (each side gives
  half); presets move the whole face. Blinks on blinkLeft/Right, visemes
  a/i/u/e/o on aa/ih/ou/ee/oh, gaze on vrm.lookAt (gaze 1 = the look-at
  range's input maximum: the eyes at the avatar's own extremes, Seed-san
  10 deg, three-vrm-girl 8-12, as Live2D's EyeBall 1 is the drawn eye's
  edge; capped at 15 deg for a bone look-at whose range map says more).
  The camera sits at the eyes' height looking down at the framing: gaze 0
  with the head at rest looks into it (it sat 0.7 hu lower: the rest gaze
  passed 3.7 / 3.2 deg over the viewer). Blush only from a custom expression or a morph
  named for it -- Seed-san has none, so cheek.blush is absent (reported).
- **Idle and events**: VRMRig gives MotionStream an EventRig's surface; the
  roles land on canonical variables in the Cubism standard units the calm
  profile is in (head_x 1 = 30 deg yaw, head_y 1 = 15 deg pitch -- Live2D
  draws its pitch at ~0.48 of its yaw, the mouth's shift at ParamAngleY
  +-30 over its shift at ParamAngleX 30 on Hiyori, Natori, lafei,
  xuefeng; at 30 the fitted idle bent the head 18 deg down at its 2nd
  percentile -- body 10 deg, gaze/brows 1). Arms
  rest relaxed (elevation 8, plane 20, elbow 14); breath = shoulders up +
  upper chest back, calibrated to BREATH_TARGET_HU like the Live2D rigs.
  No gestures for VRM: body poses only. Spring bones (hair) and node
  constraints run in three-vrm's update.
- **Pipeline**: `flow.director --avatar vrm --vrm PATH` plans into
  states.json (+ speech.wav, subs, tokens; `--render` -> render/preview.mp4
  and demo.mp4 via headless Chromium). `flow.live --avatar vrm` streams
  states on /states (SSE, server clock); the page interpolates them with
  the audio on the same playout delay. `--avatar both`: the Live2D rig and
  the VRM side by side from the same sentences (prompt from the Live2D
  body; the VRM resolves landmarks on its own body). No `--body-image` for
  VRM (the server cannot draw it).

## Motion primitives (2026-10)

Precise asks played well in body mode; open ones ("dance for me", "let's
celebrate") came out as a few held poses in a row, nothing on the beat.
`flow/moves.py` adds parameterised continuous motions the model calls by
name next to its poses:

    Let's dance! [move {"name": "dance", "beats": 8, "tempo": 120}] Come on!

- **Primitives**: sway (upper body side to side, head countering; a VRM's
  hips sway too), bounce (down on every beat: knees via the VRM's hips,
  shoulders, a little nod), head_bob, shimmy (shoulders, twice a beat),
  look_around, clap (hands meet on the beat; a VRM's through the
  `@together` composite, palms facing), cheer (fist pump, side L/R/both),
  wave (a raised hand side to side; both: both hands up, swaying), twirl
  (hands rolling around each other), and dance (2-4 beat segments of a
  trunk figure plus an arm figure, seeded, no figure twice in a row,
  mixed over a beat in target space). Aliases: nod_along (head_bob at
  half tempo), fist_pump, arm_wave, shoulder_shake, twirl_hands,
  celebrate. Params: beats 1-32 (8), tempo 60-180 bpm (110), amp 0.1-1
  (0.7), side.
- **Per body**: curves in canonical variables, offsets on the declared
  variables, hand targets solved on keyframes (8 a beat, 16 for a shimmy;
  repeating keyframes once; every other one solved) with the body's own
  solver: VRM collision-aware IK (no detour search: 87% of the time),
  Live2D forward model in one arm drawing with the arm alone. Capability
  per body decides the prompt's menu: Seed-san and three-vrm-girl all;
  Hiyori no cheer (her raised drawing lifts no fist with the arm alone);
  lafei no clap, cheer or wave. A side a body lacks is degraded with a log
  line; a primitive it lacks is skipped.
- **Play**: MoveLayer under the PoseLayer, PCHIP between keyframes (no
  overshoot at a clap's contact), min-jerk ease in and out (one beat,
  0.9 s for a clip that moves an arm), additive over the idle (the idle's
  head, gaze, breath keep going), absolute arms blended from it; a later
  move on the same parameters takes them over; a pose takes its parts and
  gives them back (held without LIFE on a move's joints). On a VRM a trunk
  figure carries a figure's hands with the chest; a free arm hangs,
  analytically (0.3 of the chest's turn, raised off the hips only as far as
  clears them): rigid arms went 0.10-0.15 hu into the hips at torso.tilt 6,
  and IK targets at a hanging palm swapped swivel between keyframes. The
  hips' sway and bob are a render offset of the whole avatar ("hips" in
  the state, only while a move writes it).
- **Checked** (scratchpad renders, every frame, idle on): VRM penetration
  <= 0.024 hu (PEN_TOL 0.02) for every primitive, a pose over a clap, two
  moves in a row, and a 16-beat dance, on Seed-san and three-vrm-girl;
  onset/end steps at the idle's level; Hiyori's dance draws one arm at a
  time, no parameter faster than 2.5 spans a second. Build time per marker
  (it adds to the sentence's first audio in live): a figure 0.02-0.15 s, a
  16-beat dance ~1.1 s on the VRMs, ~0.9 s on Hiyori; the menu at startup
  < 1 s on Hiyori and lafei, ~3.5 s on Natori.
- **Not done**: beats are not synced to the speech (beat 1 is the marker).
- `--no-body-moves`: the prompt and parser of before. Without [move]
  markers every output is the bytes it was.

## Stage 2 on one 3090 Ti (2026-10)

The baseline's Stage 2 on the user's server: one RTX 3090 Ti (24 GB,
Ampere: bf16, no FP8, no FlashAttention-3). Two students of the body-mode
director, from the data engine (`flow/dataset.py`): (a) a local model
LoRA-tuned to write the same marked replies (`flow/train/`), (b) an action
head on a local model's hidden states predicting the directed trajectory
(`flow/actionhead.py`). Nothing GPU-side has run here (no GPU in this
container): the code paths ran on CPU with a tiny random Qwen3 and tiny
random data; the 4-bit, Unsloth and llama.cpp steps have not run at all.

### What is built

- **Moves in the data** (`flow.dataset`): `[move]` markers are built on
  the body as `director.plan` builds them (seeded by the turn's seed and
  the marker's place), a gesture under an arm move gives way, and the
  move tokens stream through the same layers in both trajectories; the
  markers show `played: true`, records carry `moves`, a VRM's trajectory
  gains `hips.x` / `hips.y`. Records from before (trajectory version 1):
  `python -m flow.dataset replay OUT` re-streams them from their replay
  records with no model call (version 2 replay records give back the
  arrays they were made with, bit for bit; tested on seed-san, Hiyori lite
  and the synthetic Live2D rig).
- **One template** (`flow/train/chat.py`): Qwen3 with thinking off as
  ChatML with the empty `<think>\n\n</think>\n\n` on every assistant turn,
  history included (Qwen3's own template strips it from the history, so a
  whole session would not train as one sequence; on a single turn the two
  are the same bytes, which the trainer checks against the real tokenizer
  before it starts). The trainer, `--backend hf`, the merged tokenizer's
  `chat_template` and the Ollama Modelfile's `TEMPLATE` all render the
  same bytes (tests run the Go template the way Ollama calls it). Contents
  are never trimmed: the body prompt ends in a newline and Ollama renders
  it as is. ChatML (Qwen2.5) and Gemma 3 too.
- **Trainer** (`scripts/train_sft.py` = `python -m flow.train.sft`):
  `stats` (token lengths: whole examples, system prompt, user turns,
  replies; the share that fits at each length), `train` (session split by
  hash, labels on replies and their end token only, cut at a reply
  boundary, QLoRA NF4 r32 alpha 32 on every projection, gradient
  checkpointing, bf16, batch 1 x 16 accumulation, paged 8-bit AdamW,
  Unsloth when it imports else transformers + PEFT with the logits only at
  labelled positions, eval loss, resumable checkpoints guarded by a data
  hash, a speed probe), `merge` (fp16, into the unquantized base), `gguf`
  (llama.cpp commands), `modelfile` (TEMPLATE, stops, num_ctx 8192,
  sampling; `--create` runs `ollama create`).
- **Teaching the reachable value** (`flow.dataexport --correct-misses`,
  on by default for `sft`, `stats` and the trainer's dataset-folder
  loading): the student of the first run asked for hand positions that
  lafei's narrow-armed rig cannot reach (17% of its targets missed against
  the teacher's 8%). The solver records, per target, the value it did reach
  (`got`); the export writes that number into the marker in place of the
  missed one (an `"@shoulder.y"` anchor that missed becomes the number;
  reached targets, `dur`, `hold` and the spoken text stay byte for byte).
  A turn whose misses are all corrected no longer counts as a MISS turn
  for the export, so a session is no longer cut there; the `miss` flag and
  the reach rates stay as recorded, and `stats` counts markers and targets
  corrected per avatar and what was skipped (records from before the
  report, markers that do not line up with their pose records, a target
  whose key cannot be found). The reached value is not always a fixed
  point: asked for again, ~95% of corrected markers land (random hand
  targets, 80 per body: Hiyori 78/78 and lafei 76/80 ok, seed-san 47/56,
  where hands that would pass through the body come back shifted again);
  the action export keeps what was played. `--avatar-weight lafei=2`
  repeats an avatar's sessions in the export (also `scripts/train_sft.py`).
- **Serving**: Ollama (the GGUF and the Modelfile) through the existing
  `--backend ollama`, or `--backend hf` (transformers + the adapter,
  `TextIteratorStreamer`) in `flow.director` and `flow.live`.
  `--ollama-ctx N` sets num_ctx per request for a stock model.
- **Eval** (`scripts/eval_sft.sh`, `flow.train.evalsft`): the held-out
  sessions turn by turn through the student, the teacher live and the
  teacher's recorded replies, each keeping its own history; per model the
  parse-warning rate, refusals (the backend's and by text: local backends
  never report one), cut-offs, markers per turn by tag, unplayed and
  not-played markers, reach / MISS from the pose solve, first-text latency
  and tokens/s; `AB=1` renders first turns for a blind A/B page
  (`flow.legacy.abtest`, the `body_vs_tags.sh` layout).
- **Action head** (`scripts/extract_hidden.py` = `python -m
  flow.train.hidden`, `python -m flow.actionhead train | eval | export`):
  module docstrings (motion history, prosody, layer mix, per-avatar
  norm, the eval metrics and the ablation grid: "Action head ablations on
  BEAT2" below). `export` solves a trajectory back to parameters and
  writes move tokens: `tokens.json` for `flow.runtime --tokens-in`
  (Live2D) and `states.json` for `flow.vrmrender`, the directed clip
  through the same path, plus a `render.sh`. Known limit: a trajectory has
  hand positions and no palm orientation, so at a clap's contact the
  VRM's collision model keeps the hands ~0.1 hu wider than the clip that
  had `@together`'s orientation.

### Install

`requirements.txt` is the Stage 1 pipeline; `requirements-train.txt`
this section's pip packages (torch with CUDA first, see its header).

Tested on CPU here (these exact versions; released 2026-09/10):

    pip install "transformers==5.20.2" "peft==0.21.2" "accelerate==1.15.0" \
        "tokenizers==0.23.3" sentencepiece

GPU side, tested range unknown (never run here). PyTorch with CUDA for an
Ampere card from the official index (pick the CUDA build that matches the
driver; torch 2.14.1 is the version the CPU tests used):

    pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu128
    pip install "bitsandbytes>=0.47"          # 4-bit NF4 and paged AdamW
    pip install unsloth                       # optional, faster (it pins its own deps:
                                              # install it in its own venv)

The code also takes transformers 4.56+ (`dtype` vs `torch_dtype` and
`processing_class` vs `tokenizer` are chosen by version; unknown
TrainingArguments are dropped with a log line). TRL is not needed: its
SFTTrainer's jobs here, templating and masking, are `flow.train.chat`'s so
that they match serving exactly. llama.cpp for GGUF:

    git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
    cd ~/llama.cpp && pip install -r requirements.txt && cmake -B build && cmake --build build -j

### Memory (estimates, 24 GB)

| base | 4-bit weights + bf16 embeddings | LoRA r32 (fp32 + grads + 8-bit Adam) | activations at 8k (checkpointed) | logits at labelled positions | total |
|---|---|---|---|---|---|
| Qwen3-8B | ~6.5 GB | ~0.9 GB | ~4 GB | ~2.5 GB (152k vocab) | ~14 GB |
| Qwen3-4B | ~3.5 GB | ~0.6 GB | ~3 GB | ~2.5 GB | ~10 GB |
| gemma-3-12b-it | ~9.5 GB | ~1.3 GB | ~5 GB | ~4 GB (262k vocab) | ~20 GB: `--max-len 6144` if it does not fit |

Without the labelled-position trick (logits everywhere) an 8k sequence's
fp32 logits alone are ~5 GB for Qwen3 and ~8.5 GB for Gemma 3. The
`--max-len auto` default is the 99th percentile of the examples (a
session of 6 turns with a ~2.3k-token prompt is ~4-5k tokens).
Serving: Qwen3-8B Q4_K_M in Ollama is ~5 GB plus the KV cache at num_ctx
8192 (~1.2 GB); `--backend hf` in bf16 ~16.5 GB, `--hf-4bit` ~6.5 GB.
Hidden states: one 4096-wide layer in fp16 is 8 KB a token; 15k replies
of ~150 tokens are ~18 GB (`--proj 1024`: ~4.6 GB).

### Time (estimates)

15k turns as 6-turn sessions are ~2.5k examples of ~4.5k tokens, ~11M
tokens an epoch. QLoRA on an 8B with gradient checkpointing runs at very
roughly 1-1.5k tokens/s with PEFT and 2-3k with Unsloth on this card: ~1-3
h an epoch, the default 2 epochs 2-6 h. The trainer's first steps log the
real tokens/s, peak memory and the epoch time they imply: run
`--max-steps 20` first. Eval: ~1-2 s a turn on Ollama (Q4_K_M, 8B) plus
the teacher's own latency. Extraction: one forward of ~3-5k tokens a turn,
~0.3-0.5 s on the 8B (~2 h for 15k turns); the action head trains in
minutes.

### The sequence on the server

    # 1. data (DeepSeek teacher; resumable; export with session ids)
    SESSIONS=2500 TURNS=6 CONC=8 BUDGET_USD=20 OUT=data/datagen/r1 \
        nohup bash scripts/datagen.sh > datagen.log 2>&1 &
    python -m flow.dataset replay data/datagen/r1      # older records: moves played
    python -m flow.dataexport stats data/datagen/r1

    # 2. lengths and the template check, then a speed probe, then the run
    python scripts/train_sft.py stats --data data/datagen/r1 --base Qwen/Qwen3-8B
    python scripts/train_sft.py train --data data/datagen/r1 --out runs/sft/q8 \
        --base Qwen/Qwen3-8B --max-steps 20          # tokens/s, peak memory
    python scripts/train_sft.py train --data data/datagen/r1 --out runs/sft/q8 \
        --base Qwen/Qwen3-8B --restart               # the full run (resumable)

    # 3. merge, GGUF, Ollama
    python scripts/train_sft.py merge --run runs/sft/q8
    python scripts/train_sft.py gguf --run runs/sft/q8 --llama-cpp ~/llama.cpp
    python scripts/train_sft.py modelfile --run runs/sft/q8 --name tsk-q8 --create

    # 4. eval against the teacher (and a blind A/B with AB=1)
    RUN=runs/sft/q8 STUDENT_MODEL=tsk-q8 bash scripts/eval_sft.sh
    # or through transformers: STUDENT_BACKEND=hf STUDENT_MODEL=Qwen/Qwen3-8B \
    #   HF_ADAPTER=runs/sft/q8/adapter bash scripts/eval_sft.sh

    # 5. live
    python -m flow.live --mode body --model-json <rig>.model3.json \
        --rig rigs/Hiyori.npz --backend ollama --model tsk-q8 --tts auto
    # (or --backend hf --model Qwen/Qwen3-8B --hf-adapter runs/sft/q8/adapter)

    # 6. action head
    python -m flow.dataexport action data/datagen/r1 --out data/datagen/r1/export/action
    python scripts/extract_hidden.py --export data/datagen/r1/export/action \
        --data data/datagen/r1 --model runs/sft/q8/merged --layers -8 \
        --proj 1024 --out runs/hidden/q8
    python -m flow.actionhead train --export data/datagen/r1/export/action \
        --hidden runs/hidden/q8 --out runs/ah/q8 --steps 3000
    python -m flow.actionhead eval --ckpt runs/ah/q8 \
        --export data/datagen/r1/export/action --hidden runs/hidden/q8 \
        --markers runs/sft/q8/eval/student.jsonl --data data/datagen/r1
    python -m flow.actionhead export --ckpt runs/ah/q8 \
        --export data/datagen/r1/export/action --hidden runs/hidden/q8 \
        --data data/datagen/r1 --id <session>/t02 --out runs/ah/q8/side
    bash runs/ah/q8/side/render.sh                  # pred/ and directed/ clips

The action head's markers baseline needs another model's markers (the
student's, from the eval): on teacher text the markers played again are
the target itself.

### Action head ablations on BEAT2 (2026-10)

The first BEAT2 run (text only, 54 h, 27.6k segments) came out at normalised
MSE head 1.065, rest 1.133, mean 1.075, with the eval loss best near step
2000 and rising after: absolute MSE from text alone measures each segment's
posture, not the gesture (pitch / tilt / yaw: head = mean >> rest; the
hands 5-10% under rest). So the head can now also see its own past motion
(`--history`), the speech's prosody (`--prosody`), a learned mix of the
stored layers (`--layer-mix`) and per-avatar statistics (`--norm avatar`),
and `eval` scores what a gesture is: motion around the posture, velocity,
how much motion the prediction has, and whether it lines up with the
audio. Code and tests ran on CPU on synthetic data only; none of the
numbers below exist yet.

Hidden states with the three layers, once (WSL, repo `/mnt/e/Tsukuyomi`;
the `=` after `--layers` is needed, a bare `-1,-8,18` is read as an
option; `--proj 1024` projects each layer to 1024, so 3072 numbers a token
and `--layer-mix` splits them back into three):

    cd /mnt/e/Tsukuyomi
    python scripts/extract_hidden.py --export data/beat2/seed-san/action \
        --context none --text spoken --model ~/tsk-runs/sft/q8/merged \
        --layers=-1,-8,18 --proj 1024 --out ~/tsk-runs/hidden/beat2

The grid, one script, a log per run in `~/tsk-runs/ah/grid/<run>/`
(`train.log`, `eval.log`, `eval.json`, `head.best.pt`), the table at the
end (also `table.txt`):

    bash scripts/ah_grid.sh                       # 5 runs x 4000 steps
    STEPS=5000 PATIENCE=10 bash scripts/ah_grid.sh
    EXTRA="--layer-mix" bash scripts/ah_grid.sh   # a flag on every run

| run | flags | answers |
|---|---|---|
| a_text | (none) | the head as it was |
| b_hist | `--history 1` | what 1 s of its own motion adds |
| c_hist_pros | `--history 1 --prosody` | what the audio's prosody adds |
| d_no_text | `--history 1 --prosody --no-text` | c without the text: what the text adds is c - d |
| e_fm | c with `--objective fm` | whether sampling beats regression (it does not average several motions into a smaller one) |

Each run trains `--steps` (default 4000 in the script; 3000-5000 is the
range to try), evaluates every 100 and keeps `head.best.pt` at the best
eval loss; `eval` uses it (`--ckpt` takes the run folder, or a file:
`head.pt` is the last step). With a history the eval loss is on the
middle chunk of each held-out segment after a clean history. Every run is
scored on the same frames: the first `SKIP` s (default 1.0, the history)
of every segment are not scored for anyone, a is not credited for them and
b to e are not handed them as ground truth to be scored on. Run one by hand:

    python -m flow.actionhead train --export data/beat2/seed-san/action \
        --hidden ~/tsk-runs/hidden/beat2 --out ~/tsk-runs/ah/hp \
        --steps 4000 --history 1 --prosody
    python -m flow.actionhead eval --ckpt ~/tsk-runs/ah/hp \
        --export data/beat2/seed-san/action --hidden ~/tsk-runs/hidden/beat2 \
        --skip 1.0 --json ~/tsk-runs/ah/hp/eval.json
    python -m flow.actionhead table ~/tsk-runs/ah/*/eval.json

**Reading the summary** (columns: head, rest = the first frame held, mean =
the training mean, hold = the last frame before the scored part held; lower
is better except the ratios and correlations):

- `normalised MSE` is the old number, posture included. Expect rest and
  hold to be hard to beat; the head's win here means it knows where the
  body is.
- `relative MSE` is each trajectory minus its own clip mean: the motion
  around the posture. Rest, mean and hold are constants, so they all score
  the clip's own variance, which is the number to beat: a head under it
  predicts motion that is right, a head over it moves wrongly (worse than
  standing still).
- `velocity MSE` is the same on frame differences: smooth, right-looking
  motion that is out of step still scores badly here.
- `std ratio` is predicted over true motion std. Near 0: the head collapsed
  to the mean (what an MSE head does when it learns only the posture); near 1: it moves as much as the
  real mocap; the velocity ratio under the std ratio says over-smoothing.
  A head with a good normalised MSE and a ratio near 0 learned the posture
  and nothing else.
- `beat hands / head`: `corr` is the correlation of motion speed with the
  audio's accent strength (|d rms|), `hit` the share of accents with a
  motion-speed peak within +-0.2 s, `prec` the share of speed peaks within
  0.2 s of an accent; each followed by the ground truth's value, which is
  the ceiling (real people only partly gesture on the beat). The head
  without prosody (a, b) can only be near chance here; c should approach
  the ceiling if the audio helps, d shows how much of it is the prosody
  and the history alone.
- Compare c with d for the text, b with a for the history, c with b for
  the prosody, e with c for fm; `layer mix` prints the learned layer
  weights (layers 36, 29, 18 for `-1,-8,18`). Differences smaller than
  two runs' spread (rerun with `--seed 1`) are not results.

## E4: BEAT2 retargeted (2026-10)

The head learned little from the teacher's trajectories (normalised MSE
1.07 against 1.18 for holding still): the markers' motion choices are
partly arbitrary. `flow/beat2.py` puts real human co-speech motion in the
same files. Built and tested on synthetic BEAT2-shaped clips
(`tests/test_beat2.py`); it has not seen the real data, so the first run
is the check (below).

- **Retarget by rotation transfer.** SMPL-X's template and three-vrm's
  normalized rig (what `flow.vrmbody` drives) are both T-poses whose bones
  have the identity rotation at rest, y up, facing +Z, the character's left
  on +X: one frame, so a joint's world rotation is the same rotation on
  both, no SMPL-X body model (its mesh and licence are not needed) and no
  VRM 0 special case (the body file and the page already turn VRM 0 into
  the 1.0 frame). The world rotations are fitted to the body's own
  parameters top down (torso, head, shoulders, upper arm from its direction,
  the elbow from the angle to the forearm, the twist chosen so the forearm
  lands on the mocap forearm, the wrist from the rest; forearm roll goes to
  the wrist), each joint measured against what the joints above it achieved
  after their limits. The avatar's own T-pose droop (`align`) drops out.
- **Root.** Everything is taken in the actor's own pelvis frame (global
  orientation, per frame) turned by the clip's habitual spine posture (median
  pitch / tilt of spine3 against the pelvis), so the torso values are the
  spine's motion against the pelvis, not the mocap's global frame: on the
  first real clip (Seed-san) the torso sat at pitch -22 / -12 / -5 deg
  (p5 / p50 / p95, range +-15) and yaw -40 / +5 / +29 (range +-25) until
  this, a tilted pelvis and whole-body turns read as torso lean and twist;
  now -8 / 0 / +7 and -5 / +1 / +8. The translation, in the frame of the
  torso's heading (circular moving mean over 20 s, `--facing-window`; 0 =
  the clip's mean), high-passed, is `hips.x` / `hips.y` (clipped to 0.15 /
  0.08 hu). A z-up clip is detected from the spine and turned. Legs unused.
- **Variables.** The same columns the data engine records for the avatar
  (`flow.dataset.traj_vars`), from the body's forward model; gaze, eyes,
  brows, mouth and visemes are NaN (the loss masks them; the voice owns the
  mouth; no FLAME mapping). `<key>.joints` keeps the 23 fitted parameters.
- **Filter** (GMR's lesson). Per frame, before the limits clamp: arm in
  trunk / head and arm in arm (the collision model), palm speed, joint past
  its limit. Bad frames (defaults: 0.10 hu, 25 hu/s, 25 deg, 0.15 s around)
  cut the utterances; pieces under 3 s are dropped. The wrists are clamped
  but not counted as past a limit (their roll is the forearm's twist, the VRM
  has no forearm bone for it; on the real clip they were most of the "52 deg"
  limit excess).
- **Contacts.** Rotation transfer keeps angles, not contacts, so a thin
  actor's hand on the belly is inside a bulkier avatar (real clip: arm in
  body 0.05 / 0.17 / 0.25 hu at p50 / p90 / p99, 30% of frames over 0.10). Arms
  deeper than `--contact-pen` (0.04 hu) are solved again with the body's
  collision-aware IK for the palm where the mocap put it (onto the surface
  when the palm is inside), the swivel nearest the mocap arm, on keyframes
  every 3 frames, the change interpolated and faded in and out over 0.2 s
  (contact-aware retargeting, docs/research.md). Real clip: 0.01 / 0.02 /
  0.03 hu, palm moved 0.04 hu (median) where it was changed, palm speed p99
  unchanged. `--no-contact-fix` turns it off. Look at `report` and tune
  `--max-pen` before training.
- **Segments**: one utterance per row (cut at 0.35 s silences between
  words, 3-15 s, 0.3 s of motion either side); `spoken` = the words joined,
  per-character times from the word times, relative to the segment.
- **Prosody** for the head (`flow.actionhead --prosody` reads it):
  `<key>.prosody` [frames, 3] = RMS, f0 (Hz, YIN), voiced, at the frame
  times; the row names the columns (`prosody`).

On the GPU machine (WSL, repo at `/mnt/e/Tsukuyomi`). BEAT2 is the Hugging
Face dataset `H-Liu1997/BEAT2` (check the page for the files and the
licence; the layout assumed is `smplxflame_30/`, `wave16k/`, `textgrid/`
next to each other, and `--beat2` is the folder that holds them, or one
level above it when only one subfolder does). Retargeted motion is a derivative of it: **keep the
outputs out of git** (`--out` must be under `data/`, which is ignored).

    cd /mnt/e/Tsukuyomi
    # 0. the data (check the dataset page; this needs git-lfs or huggingface-cli)
    huggingface-cli download H-Liu1997/BEAT2 --repo-type dataset \
        --local-dir ~/BEAT2
    # 1. a body file for the avatar (once; flow.train.hidden reads rigs/<name>.body.json)
    python -m flow.vrmbody --vrm assets/vrm/VRM1_Constraint_Twist_Sample.vrm \
        --out rigs/seed-san.body.json
    # 2. a small run first, then all of it (resumable; --jobs = processes)
    python -m flow.beat2 retarget --beat2 ~/BEAT2 \
        --vrm assets/vrm/VRM1_Constraint_Twist_Sample.vrm \
        --body rigs/seed-san.body.json --out data/beat2/seed-san \
        --speakers 2 --limit 5 --fingers
    python -m flow.beat2 report --out data/beat2/seed-san
    python -m flow.beat2 preview --out data/beat2/seed-san --beat2 ~/BEAT2 \
        --id beat2.<clip>/s00                  # data/beat2/seed-san/preview/
    python -m flow.beat2 preview ... --also-vrm assets/vrm/three-vrm-girl.vrm \
        --also-body rigs/three-vrm-girl.body.json    # side by side
    python -m flow.beat2 retarget --beat2 ~/BEAT2 --vrm ... --body ... \
        --out data/beat2/seed-san --jobs 8
    # 3. hidden states of the text alone, then the head, then its eval
    python scripts/extract_hidden.py --export data/beat2/seed-san/action \
        --context none --text spoken --model ~/tsk-runs/sft/q8/merged \
        --layers -8 --proj 1024 --out ~/tsk-runs/hidden/beat2
    python -m flow.actionhead train --export data/beat2/seed-san/action \
        --hidden ~/tsk-runs/hidden/beat2 --out ~/tsk-runs/ah/beat2 --steps 3000
    python -m flow.actionhead eval --ckpt ~/tsk-runs/ah/beat2 \
        --export data/beat2/seed-san/action --hidden ~/tsk-runs/hidden/beat2

What to check first on the real data: that a clip's report says `up_axis:
y-up` and `fps_source: mocap_frame_rate` (otherwise the layout differs from
the assumption), the preview of two or three clips (arms where they should
be, not crossed or behind the back, the torso not twisting as the person
turns), and `report`'s distributions (kept hours, the penetration quantiles;
if most is dropped, raise `--max-pen`). `--context none` is the text alone,
the distribution the head sees when the speech is not a marked reply; the
`--text spoken` matches the rows (`reply_raw` = `spoken`).

#### Grid results (2026-10-28, runs/ah_grid/)

BEAT2, 3,068 held-out segments, the first second of each unscored (the
history's); normalised MSE (nmse), motion around each segment's own mean
(rel), predicted / true motion std (std):

| run | nmse | rel | std | hands speed ~ audio accents (corr) |
|---|---|---|---|---|
| hold last frame | 1.105 | 0.471 | 0 | - |
| a text only | 1.066 | 0.473 | 0.08 | 0.02 |
| b + history 1 s | 0.899 | 0.496 | 0.42 | 0.04 |
| c + history + prosody | 0.854 | 0.488 | 0.41 | 0.30 |
| d history + prosody, no text | **0.790** | **0.468** | 0.34 | 0.29 |
| e = c with flow matching | 1.652 | 0.764 | 0.84 | 0.01 |
| ground truth | | | 1 | 0.03 |

What it says:
- History is what lowers nmse: knowing the current pose. None of the
  regression heads predicts the motion *around* the posture better than a
  constant (rel 0.47-0.50 vs 0.471): the exact gesture is not predictable
  from these inputs, and regression shrinks to a fraction of the real
  motion (std 0.08-0.42).
- Text adds nothing measurable here: d (no text) beats c on every row.
  The LLM's states of the transcript do not carry the gesture beyond what
  history and prosody give, at least under per-frame error.
- The prosody heads tie hand speed to loudness far more than people do
  (corr 0.30 vs 0.03 in the mocap): a learned shortcut, not human timing.
- Flow matching (e) moves like a person in amount (std 0.84) but its
  per-frame error is large, as a sample (not an average) must be; per-
  frame metrics cannot judge it, only watching or distribution metrics
  (FGD-style) can.

So, for now: the explicit path (the LLM writes poses and moves, the solver
executes) stays the body's controller; an audio-driven generative layer
(history + prosody, flow matching) is the candidate for background gesture
between markers, to be judged by eye; LLM hidden states as the motion
source are not supported by either the teacher data or BEAT2.
