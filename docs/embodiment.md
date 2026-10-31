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
- **Streamer manners** (`flow.dataexport --manners`, on by default for
  `sft`, `stats` and the trainer's dataset-folder loading; `--no-manners`
  for the export of before, byte for byte; `--max-poses N`, default 2):
  user feedback on renders of the first student was arms raised "for no
  reason" (worst on lafei, whose arm can only swing out sideways). The
  teacher writes ~2.3 pose markers a reply and about a third of them move
  a hand; the student copied that, where streamers move head, face, gaze
  and a little torso, and the hands rarely and on purpose. The export
  decides per turn whether the user asked for a body action (the topic is
  `pose`, `show` or `invite`, or the line matches the en / ja / zh word
  lists `ASK_WORDS`, with `ASK_NOT` cutting false friends such as 相手,
  举例, "on the other hand"; the decision is counted by `stats`). Not
  asked: pose markers keep head, gaze, brow, eye, mouth and cheek targets,
  torso yaw / pitch / tilt up to 6 degrees, `dur` and `hold`; hand, arm,
  shoulder and bigger torso targets and composites (`"hands":
  "@together"`) go, and a marker with nothing left is removed with its
  extra space; `[move]` other than `head_bob` / `look_around` (and their
  aliases nod, bop, look) and every `[gesture:NAME]` go (a directed
  gesture plays the arms on the Cubism rigs); `[blink]` and `[pause]`
  stay. Asked: the body markers stay. Either way at most `--max-poses`
  pose markers a reply stay (the first; markers with body targets first
  on an asked turn). It runs after `--correct-misses`, on history turns of
  a per-turn export too, and a turn whose spoken text would change (the
  pose parser's reading, before and after) is left alone and counted. It
  changes the training text, so a run needs a new `--out`. No record of
  the real teacher data was at hand when it was written: the word lists
  were checked against the personas file's topic examples and the tests'
  mocked records only; read `stats` on r1 first (below).
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

### Retraining with streamer manners (the GPU machine)

WSL, repo at `/mnt/e/Tsukuyomi`, data `data/datagen/r1` (the first student
was `~/tsk-runs/sft/q8`). First what the transform does to the data: the
turns asked / not asked per avatar, pose markers per reply before and
after, the hand share of targets, the moves and gestures dropped, the
replies left with no body marker:

    cd /mnt/e/Tsukuyomi
    python -m flow.dataexport stats data/datagen/r1 --json ~/tsk-runs/manners-stats.json
    # per turn: "manners"."by" in the json (topic / keyword / topic+keyword / null)
    python -m flow.dataexport stats data/datagen/r1 --no-manners   # the old numbers

Then the run (manners and correct-misses are on by default; a new `--out`,
the data hash differs from `q8`'s), and the rest as in "The sequence on the
server":

    python scripts/train_sft.py train --data data/datagen/r1 \
        --out ~/tsk-runs/sft/q8m --base Qwen/Qwen3-8B --avatar-weight lafei=2
    python scripts/train_sft.py merge --run ~/tsk-runs/sft/q8m
    python scripts/train_sft.py gguf --run ~/tsk-runs/sft/q8m --llama-cpp ~/llama.cpp
    python scripts/train_sft.py modelfile --run ~/tsk-runs/sft/q8m --name tsk-q8m --create
    RUN=~/tsk-runs/sft/q8m STUDENT_MODEL=tsk-q8m TEACHER=0 bash scripts/eval_sft.sh
    python -m flow.live --mode body --model-json <rig>.model3.json \
        --rig rigs/Hiyori.npz --backend ollama --model tsk-q8m --tts auto

The eval's markers-per-turn table should show fewer pose markers and a
lower hand share than `q8`'s. Its held-out sessions are the same ones as
`q8`'s (the split is by session hash). Words that wrongly count as a
request keep a hand; words that wrongly do not drop one: edit `ASK_WORDS`
/ `ASK_NOT` in `flow/dataexport.py` and run `stats` again before training.

### Expression beats (r2 data)

Real VTubers (Kaf, Shigure Ui) and an AI VTuber (Shizuku), all framed chest
up, move mostly the face: several expression changes a sentence (eyes
closed in a smile at the funny parts, mouth shapes, brows), a small head
tilt or turn with each one, a body that is almost still and the hands out
of the picture. The r1 replies held one expression a reply. Three changes,
none of which touch the r1 files or the flags of a run in progress:

- **Teacher prompt** (`flow.pose.system_prompt(..., prompt="beats")`;
  `flow.dataset gen --teacher-prompt {beats,v1}`, default `beats`; `v1`
  regenerates r1's prompt byte for byte, tested). The "How to move" part
  gains, worded from the variables this body has (`pose.offered`; lafei has
  no `eye.*.smile`, so it is told to close the eyes with `eye.*.open`
  instead, and brows / blush only where they exist):

      Your face does most of the talking, as a streamer's does, and it
      never holds still: every sentence gets at least one expression beat,
      a longer sentence two or three, so that something changes about every
      1.5 to 2 seconds of speech ... Vary them, not the same smile each
      time: the eyes, the brows, mouth shapes, a blush, the gaze, small head
      moves (worded from the variables the body has) ...
      Each beat carries a small head move in the same marker, a few degrees
      (head.tilt, head.yaw, head.pitch between -8 and 8) ... Never write
      that head move as a marker of its own.
      Keep the rest of your body almost still, hands out of the picture: no
      hand, arm or big torso moves unless the user asks you to show or do
      something, or the moment is a big one (a greeting, a celebration), and
      then once in a reply at most.
      (That last paragraph is replaced by the camera, budget and `[frame]`
      lines of "Framing and motion budget" below.)

  The rule becomes "At least one expression beat in every sentence, two or
  three in a longer one" (the 20-session pilot had 0.65 beats a sentence,
  2.3 a reply of about 9 s of speech, against the one to three asked for;
  real streamers change expression every 1-2 s). The definition of a beat
  is unchanged, so the stats count the same thing. The "use
  your whole body" line is dropped and the placement examples are face
  beats with their head angles. The director's own default stays `v1`
  (`--body-prompt`): a student is served with the prompt it was trained
  on, so serve an r2 student with `--body-prompt beats`.
- **Measurement.** `dataexport stats` prints an "expression beats" block
  under the manners one, per avatar and in all: beats per sentence, the face
  and hand share of all pose targets, the share of beats with a head
  angle, pose markers per reply, replies with a beat. A *beat* is a pose
  marker whose targets are face variables (brow, eye, mouth, cheek, gaze)
  and optionally `head.*`, nothing else. With manners on the rates read
  "before manners -> after" (and "beats" in the json under `beats`, with
  `before`). Compare r1 and r2 by running it on both roots.
- **Export.** Manners caps pose markers at `--max-poses` (2), which would
  cut a reply of beats to two. `--max-beats N` caps the beats on their own
  and leaves `--max-poses` to the other markers. Default: 9 when the
  dataset's `avatars.json` carries the beats prompt, else 0, which is
  manners as before, so r1 exports (and the hash of a run such as q8m)
  do not change. The trainer takes the same flag (`--max-beats`). The
  default is 16 (was 12, 9 before): 3-4 sentences with 2 beats each are
  6-8 markers, and the prompt's dense example (below) is 8 beats in 4
  sentences; a long reply written like it (5-6 sentences) reaches 14, which
  12 would have cut at the tail, against the density asked for.

**Idle head drift.** Before, the calm idle held the head still between
fixations (holds of 3 s or more, `events.IDLE`), nothing moved it
continuously, so a quiet stream looked frozen. `flow.stream.HeadDrift` adds
a slow wander: per axis the sum of three sines (periods 4-10 s, seeded
phases), peaking at 2.5 deg yaw, 1.5 pitch and 2 tilt, and a breathing
sway of the body angles (`body_x`, `body_z`: 0.8 and 0.6 deg, periods 4-7
s) where the rig binds them. It is added to the head parameters before
moves and poses are applied, so face-only beats leave it running, and a
pose that targets the head moves from where the drift has the head to its
target and releases back to it (the held target keeps the idle's small
life under it, as for any pose). `flow.live` takes `--head-drift` (default
on), `--no-head-drift` and `--head-drift-scale X` (they sit in the
director's parser, which it shares). The offline director render has it on
by default too (`runtime --head-drift-scale`, which the director passes;
the runtime's own default is 0, so a bare `flow.runtime` run and the
dataset's trajectories and the action head's data are as before:
`MotionStream(head_drift=None)` is off and `dataset.director_args` turns
the director's flags off).

**Motion in bursts** (q8r2 demo review: "every figure moves in bursts, even
a swaying head"). Measured on `Mao_10/demo.mp4`, 34% of the 0.1 s bins had
almost nothing moving, between short spikes. Causes: (1) the offline render
had no head drift (it was only in `flow.live`), so between the calm idle's
fixations (3 s or more) the figure was still; (2) the model's beats are 0.15-0.3 s
eases to a held target, so a head beat is one short spike, then a hold with
nothing moving. Fixed in the playback only (targets and the onsets of the
model's markers are untouched; `--no-head-drift`, `--no-pose-ease` give
the old render):

- the drift above, now in the offline render (and a livelier period range:
  4-10 s, was 5-13 s, because near a sine's turning points all axes were
  slow together);
- `PoseLayer(ease=True)` (`--pose-ease`, default on in the director and
  `flow.live`; the dataset and the exporter keep the beats as written):
  head, torso and body angles take at least `pose.ANGLE_DUR` = 0.35 s to
  arrive and `ANGLE_RELEASE` = 0.7 s to relax; other (face) parameters at
  least `FACE_DUR` = 0.12 s and `RELEASE_MIN` = 0.5 s. All minimum jerk (an
  S-curve). The hold is as long as the marker wrote, from when the longer
  move is done; the held target keeps the `LIFE` and `SETTLE` it had (the
  drift continues underneath).

`python scripts/motion_still.py plan.npz|demo.mp4` prints the share of
0.1 s bins with nearly no motion (mouth, blinks and breath left out of a
plan; a video's differences include them) and the burstiness (p95 / median
of the bin speeds). Target for the offline path: under 10% still. On four 0.2 s head
beats over the calm idle (test rig, 10 s, mean of 8 seeds): still bins
23.0% before, 2.6% after (the idle alone: 38.8% before, 5.4% after);
burstiness 11.0 before, 6.7 after. Measure a real run with its `plan.npz`.

**Dense worked example.** Two pilots of the beats prompt gave 0.69 beats a
sentence, about 0.30 a second of speech, against the 0.5-0.65 a streamer's
face changes (every 1.5-2 s); stronger rule text barely moved it. The
placement examples of the beats prompt now start with one whole reply
(`pose._dense_example`, 4 sentences, about 10 s of speech, 8 beats, a beat
at nearly every clause), so the teacher has a density to copy, not only a
rule: brows up, eye smile, eyes closed in a laugh, gaze away and back,
brows knit with the mouth down, a blush, brows up on a question, a smile
for the last line, each with a small head move in the same marker and no
hand. It is built from `pose.offered(bm)` like the other examples, so a
variable the body lacks is left out; lafei (no `eye.*.smile`) gets a half-
shut squint (`eye.*.open` 0.45) for the smile beats and `eye.*.open` 0 for
the laugh; a VRM gets the same with its own variables. v1 is untouched
(tested by digest per body); the words "expression beats" stay in the
rule text, where `dataexport.BEATS_MARK` finds them. Hiyori's:

    [pose {"brow.L.lift": 0.7, "brow.R.lift": 0.7, "head.pitch": 4, "head.yaw": -4, "dur": 0.2, "hold": 0.4}] Oh, you really did that? [pose {"eye.L.smile": 0.8, "eye.R.smile": 0.8, "mouth.form": 0.9, "head.tilt": 6, ...}] That is the funniest thing, [pose {"eye.L.open": 0, "eye.R.open": 0, "mouth.form": 1, "head.tilt": -5, ...}] and I am not even sorry. [pose {"gaze.x": -0.6, "gaze.y": 0.2, "head.yaw": -6, ...}] Honestly, I was [pose {"gaze.x": 0, "gaze.y": 0, "brow.L.lift": -0.5, "brow.R.lift": -0.5, "mouth.form": -0.4, ...}] a little scared, [pose {"cheek.blush": 0.6, "mouth.form": 0.4, ...}] but it was brave, [pose {"brow.L.lift": 0.5, "brow.R.lift": 0.5, ...}] wasn't it? [pose {"eye.L.smile": 0.8, "eye.R.smile": 0.8, "mouth.form": 0.7, ...}] Now tell me the rest.

**Speech-driven micro-expressions** (`flow.micro`). Between the model's own
beats, a layer in the manner of the head drift adds small, short face
changes from the TTS audio. The audio features: the loudness envelope (RMS
in dB over 40 ms, smoothed over 120 ms) and, from `flow.beat2.prosody`
(YIN), the pitch's rise over its local median in semitones; their sum,
against the sentence's own median, has a stress peak where it tops 0.7 and
is a local maximum over +-0.2 s (at most 0.7 a second of voiced speech).

- **Brow**: a lift of 0.2-0.3 of the brow's range for 0.25-0.4 s at a stress
  peak (a little before it).
- **Smile**: `eye.*.smile` (else `mouth.form` up) at 0.25-0.35 of the range,
  easing in over 0.35 s so that it peaks as a sentence ends, gone in 0.4 s;
  not at a question (final pitch rise of 2 semitones or more). Hiyori's
  `mouth.form` already rests at its maximum, so she smiles with the eyes.
- **Blink**: an ordinary blink token at a pause of 0.35 s or more inside the
  speech and at its end, 2.5 s apart at the closest. `Autonomic` only had a
  Poisson blink (mean 4 s) and the requests of large gaze shifts; a request
  moves the next scheduled blink to the pause.

Rules: seeded and deterministic (a function of the audio, the seed and a
key); 0.8 s between two face events, also across sentences; a face event is
dropped when a model beat on the same variables (brows for a brow, eyes /
mouth / cheeks for a smile) is within 0.6 s before its onset or after its
hold (the model's beats win; `micro: ... N dropped for a model beat nearby`).
Each sentence logs `micro: N events (b brow, s smile, k blink) over X s of
voice`.

Played as `micro` tokens, parameter offsets that the body model solves once
per body (the canonical variable at its full range, scaled; face parameters
only, no arm drawings, nothing the lip sync owns; a variable the body lacks
is skipped). `flow.micro.MicroLayer` adds the eased offset to the frame
before moves and poses (`MotionStream.frame` and `runtime.main`'s loop), so
a pose on the same parameter takes over. The offline director writes the
tokens into `actions.json` (`micro`) and so into `tokens.json`: `--tokens-in`
replays them, and the live stream draws the same frames (tested).

Flags (the director's parser, shared by `flow.live`): `--micro` (default on),
`--no-micro`, `--micro-scale X` (amplitude, x the defaults; 0 is off). On by
default in the offline director render and in `flow.live` (`--avatar` live2d,
vrm and both) in body mode with a TTS; tag mode and a silent render have no
body model or audio, so none. The dataset's trajectories and the action
head's data never get it (`dataset.director_args` sets it off; only
`director.plan`, `vrm.plan` and `live.Performer` build a `Micro`).

### Framing and motion budget

*Camera follow and tall figures.* Two render rules sit on the framing
below. **Camera follows whole-body moves** (`framing.follow_tokens`): a
`[move]` of `dance`, `twirl`, `shimmy`, `bounce` or `cheer` in bust would
be cropped, so the camera eases to full 0.3 s before it (the usual 0.6 s
ease) and back to the framing it left 0.5 s after it ends (the turn's home
in the usual case); moves closer than that share one trip. `wave`, `clap`,
`sway`, `head_bob` and `look_around` are hand or head moves: no switch. Not
from full, and not when a `[frame]` marker of the model falls in the
window (from lead to tail): the model's markers win. The frame tokens are
made where the model's own are (the runtime's `--direct`, `flow.live`), so
both renders get it; it logs `framing: camera follows dance: full from ...`
when it triggers. `--no-camera-follow` (the director's parser, shared by
`flow.live`; the runtime's too) turns it off. **Hat window**
(`framing.bust_math`): the bust window's top is `ABOVE` hu over the chin
unless the figure is taller at rest (a witch hat): then it is the rest
render's alpha top plus 0.1 hu, at most `HAT_CAP` = 1.0 hu above `ABOVE`
(very tall ears do not shrink the face further). `BELOW` and the chin's
horizontal centring stay, and the prompt's camera line (hand visibility,
`BELOW`) does not change. Logged as `framing: bust, 2.60+1.0 hu ...`.


The r2 beats prompt first told every avatar to keep its hands "out of the
picture". That was true of one framing and of some bodies only: lafei's
resting hands (y -1.10 hu) are inside the bust picture, Hiyori's (-2.00)
are not, and a streamer switches scenes (face cam, full body for a dance).
The beats prompt (`v1` is untouched, byte for byte) now says how the avatar
is framed, lets it switch, and sizes its motion by what its body can do.
`flow.framing` has the numbers, `flow.pose` the wording.

- **Camera line.** The bust picture is the window of `framing.ABOVE` /
  `BELOW` (1.5 hu over the chin to 1.6 under it, as wide as it is high,
  centred on the chin). The system prompt (byte-stable per body, so the
  cached prefix and `avatars.json` stay as they are) says what each framing
  shows, worded from that window against where this body's hands rest and
  how low they reach (`pose.camera_rule`): Hiyori "your hands rest at y
  -2.00 and -2.09, below the picture, and a hand shows only once raised
  above y -1.60 ... in bust a hand move below y -1.60 is not seen: the face
  carries the performance"; lafei "in the lower part of the picture ... all
  your hand moves stay in that picture". The framing *now* is a line after
  `(your body now: ...)` in each user turn: `(camera: bust, head and chest;
  hands below y -1.60 are out of frame)` or `(camera: full, the whole body
  in frame)` (`pose.camera_now`). It is in `user_content`, so history,
  the SFT export and `evalsft` carry it for free; v1 data has no such line.
- **`[frame bust]` / `[frame full]`.** A marker like any (`[frame:full]`
  and quotes also parse; anything else is a warning "'frame' takes
  bust/full; dropped", counted as an unplayed `frame`). Only a parser built
  with `frames=True` takes it (`parser_for(..., frames=True)`, the
  director's `--body-prompt beats`, the dataset's beats runs, the export
  parsers): the v1 prompt and parser drop it as before. A VRM body never
  takes it (`framing.can_frame`): the browser page draws it at one camera
  (`vrmbody.camera`), so a VRM is always full and the prompt says so; the
  transition is not rendered there. The prompt lists it with the markers:
  "Go full only for a whole-body moment: a dance, showing an outfit or a
  pose, a big celebration with the body; not for a greeting wave or a single
  hand move ... eases over in about 0.6 seconds ... lasts until the end of
  this reply; the next reply starts back in your home framing. At most once
  in a reply." The pilot's teacher went full at the first hand moment (often
  a greeting wave) and stayed: Hiyori had 9 of 47 turns in bust although 70%
  of sessions start there, all 4 of her switches bust to full and none back.
  The camera line says the same in the other direction ("in bust a hand move
  ... is not seen: skip it and let the face carry the moment ... instead of
  switching to full"; "a switch lasts for this reply, and the next one
  starts back in your home unless you write the marker again") and the
  budget line adds "switch to full only for a whole-body moment ..., never
  for a single hand move". The home itself is not in the system prompt
  (byte-stable per body): the user turn's camera line shows it.
- **The ease.** A marker becomes a token `{"type": "frame", "view",
  "duration": 0.6}` at its time in the speech (`framing.frame_tokens`).
  `framing.Camera` holds the two `set_view` states (`bust_view` and the
  whole canvas) and eases scale and both offsets between them with minimum
  jerk from where the view is when the switch begins (a switch back mid-ease
  turns around smoothly); `Camera.at(t)` is pure in t. Live:
  `Performer.sentence` adds the tokens (stream time = the sentence's audio
  start + the marker's place), `MotionStream` commits them to its `camera`
  and `RenderLoop` calls `backend.set_view` before a frame whose view
  differs; `--renderer none` uses a stand-in zoom so the markers still run
  through the pipeline. Offline: `runtime --direct` appends the tokens to
  the directed ones (so `--tokens-in` replays them), the render subprocess
  builds the camera from them and sets the view per frame; the director
  starts a turn in `a.frame` (`--frame`, bust), and the next turn back in
  it (the home framing, below). Without a body frame (no `bust_view`) the
  render stays full and says so.
- **State across turns: a home framing.** A session or live chat has a home
  framing and every turn starts in it; a `[frame]` switch, to full or to
  bust, lasts until the end of its reply and the next turn is back home
  unless the model writes the marker again. Data: a session's home is bust
  with probability `--bust-share` (0.7; seeded by run seed and session id
  alone, `framing.start_frame`, so no other draw moves); each turn starts
  there (the camera line of `user_content` shows it), the teacher's markers
  move it within the reply, and each record has `framing: {start, home, end,
  switches}` (beats runs only; `start` equals `home`, `end` is where the
  reply left the camera). Offline director `--chat`: `--frame` is the home;
  `a.frame` is reset to it after each turn (the render of a turn starts in
  `a.frame` and replays its tokens). Live: `Performer.home` (`--frame`,
  full when the body has no bust view); when a reply ends away from home
  the Performer commits a `frame` token back home at the end of its speech
  (stream time `next_frame / fps`) and shows home at the next line.
- **Motion budget.** `framing.budget(bm)` measures the body: the share of a
  gesture zone (x 0.15-1.1 hu out from the midline, y from where the hands
  hang to 0.6 hu over the head, 8 x 6 points a side) that some arm drawing
  reaches, taking each drawing's per-variable ranges (`body.limits`) as a
  box, so it is an upper bound; the share of those points the bust picture
  shows; the lowest / highest hand y; the arm drawings; the face controls of
  `pose.offered` (brow, eye, mouth, cheek, gaze); the kind. Tiers: below 35%
  of the zone `limited`, from 80% `capable`, else `moderate`; the hand moves
  a reply per framing follow (limited 1 / 1, moderate 1 / 2 full, capable
  1 / 3 full; bust 0 when no reached point is in the picture). The prompt
  line (`pose.budget_line`) is e.g. "your arms are limited (a hand reaches
  about 14% of the places a hand goes when talking, through 2 arm drawings)
  and you have 16 face controls offered: carry the performance with the face
  and head, and move a hand rarely, at most 1 hand move in a reply". It is
  stored under `budget` (with the landmarks) in each avatar of
  `avatars.json` and printed by `dataexport stats` ("motion budget").
  Measured: Hiyori limited (14%, 38% of it in bust, 2 drawings, 16 face
  controls of 27), lafei limited (23%, 1 drawing, 10 of 20), Natori moderate
  (72%, 12 drawings, 15 of 28), seed-san capable (100%, 11 of 24); the same
  numbers for the other VRM samples. `python -c "from flow import framing,
  body; print(framing.budget(body.load('rigs/Hiyori.body.json')))"` checks
  one.
- **Manners for beats data.** The teacher now chooses its moments for a
  hand (a greeting, a celebration, a size, pointing, an emphasis), so for a
  dataset written with the beats prompt manners no longer strips unasked
  hand targets. `--max-hands N` (default 1 there; `-1` and other data: strip
  as before, same export and data hash as r1) keeps hand and arm targets and
  caps the markers that move a hand (a pose with a hand target, a `[move]`
  other than the head ones, a `[gesture]`) at N a reply where nothing was
  asked, the first ones; the later ones lose their hand targets or go.
  Shoulders and a big torso move still go. In bust (the record's start
  framing, followed along the reply by its `[frame]` markers) hand targets
  that would end out of the picture are stripped, asked or not: where the
  hand ends up (the marker's x and y, or the resting position for the one
  it leaves out; `@` references resolved with the avatar's landmarks)
  against the window. `frame` markers are kept, and a reply with only a
  frame marker still counts as having no body marker. The trainer takes
  `--max-hands` too.
- **Stats.** `dataexport stats` adds, for records with a framing, per avatar
  and per framing a turn starts in: turns, beats per sentence, face and hand
  share of targets, hand-moving poses and `[frame]` markers per reply, the
  switches (`framing` in the json); and the budget of each avatar (`budget`).
  `evalsft` counts `frame` markers per turn in its own column and leaves them
  out of the body markers. Against the session's home (`framing.home`; for
  records from before it, the framing the session's first turn started in)
  each framing row also gives the switches away from home, the switches back
  and the turns spent away (started away, or went away in the reply), and
  beats per second of speech (`timing.speech_seconds` of the record; in the
  json `away`, `returns`, `turns_away`, `timed_beats`, `speech_seconds`).
  The pilot's problem reads as many `away` and no `returns`.

### Reply language, new rigs, face vocabulary, r2 avatars (2026-10)

**Reply language.** A pilot had about 7% of teacher replies in another
language than the user's latest line (whole stretches of a session, after
the simulated user switched: `switch` / `mix` in `datagen.personas.json`);
the trained students did the same on about 11% of eval turns
(`flow.dataset.detect_lang`, mixed lines included). `scripts/lang_check.py`
lists them.

- *Prompt.* The beats prompt's rules gain (`pose.LANG_RULE`): "Answer in the
  language of the user's latest message, even if your earlier replies used
  another language. When the user mixes languages, answer in their main
  language (a mixed-in word or two may be mirrored)." It reaches the
  dataset teacher and the director / live character served with
  `--body-prompt beats`. `v1` and the director's tag prompt are untouched.
- *Export.* `dataexport stats` / `sft` and the trainer drop a turn whose
  reply language (`lang.reply`) differs from its user line's (`lang.user`),
  unless the record's `lang.mix` is set or either language was not
  detected. It is a hard flag (`language`) like the others: a session
  export is cut before the first one (the replies after it would carry the
  wrong language as history), a `--per turn` export leaves out the turns
  that would show it in their history. Stats and the sft command print
  "language: N turns dropped, by pair: en->ja 3, ...". `--lang-filter` /
  `--no-lang-filter` (also on `flow.train.sft`) force it; default: on for a
  dataset written with the beats prompt, off otherwise, so the r1 export
  and its data hash do not change.

**Rigs with no visible arm drawing / no chest.** `flow.body` measured every
arm variant by its visible drawables; a variant with none at rest (Lisette)
raised "need at least one array to concatenate". It is now skipped with a
logged caveat (`caveat: variant NAME has no visible drawable at rest`), the
rest are measured, and the file lists them under `skipped_variants`.
`flow.framing.hand_reach` read `landmarks["chest"]` even when a hand rests
somewhere (huonululu_5 has no chest landmark): it now goes through
`framing.chest_y`, which falls back to 0.4 hu under `shoulder.y` (else under
the default shoulder line, -0.25) and logs it once.

**Face-tracking rigs.** A VTube-Studio-style rig (八千代辉夜姬) offers 18
face controls, of which the body builder mapped 8. `body.DECLARED` gains,
read from the standard ids (underscores and case ignored, `body.ALIASES`
for the other spellings; sided ones follow the rig's side naming):

| variable | id | range | meaning |
|---|---|---|---|
| `eye.L.squint`, `eye.R.squint` | ParamEyeLSquint / RSquint | 0..1 | + squints |
| `cheek.puff` | ParamCheekPuff | 0..1 | + puffs the cheek |
| `mouth.pucker` | ParamMouthPuckerWiden | -1..1 | - purses the lips, + widens (as `mouth.form`: + is the smile side) |
| `mouth.funnel` | ParamMouthFunnel | 0..1 | + rounds the lips into an O |
| `tongue.out` | ParamTongueOut | 0..1 | + tongue out |
| `eye.tears` | ParamTear (ParamTears, ParamTearDown_1) | 0..1 | + tears |

`eye.tears` has the first of the rig's three `ParamTearDown_n` levels. They
are in `body.EXTENDED`: offered (`pose.offered`) and worded in the beats
prompt ("small tricks of the face", a squint for doubt) only for a body that
has them, and left out of the "does not exist on this body" line, so the
prompts of Hiyori, lafei, Natori and seed-san stay byte for byte (digests
pinned in `tests/test_facevocab.py`). They count as face for the beats
(`tongue.` joins `eye.`, `mouth.`, `cheek.` in the face prefixes of
`dataexport`, `framing`, `beat2`, and in the micro layer's smile group, so a
model beat on them keeps a micro smile away) and for the motion budget.
Also read from the ARKit-style names (EyeSquintLeft / Right, CheekPuff,
MouthFunnel, TongueOut, with or without `Param`). ARKit's MouthPucker is
not mapped to `mouth.pucker` (its sign is the other way round). Not mapped
(no canonical meaning yet): ParamMouthShrug, ParamMouthPressLipOpen,
ParamEyeSmile_Happy_*, ParamTearDown_2/3, ParamExpression_*. A body file
built before this change does not have them: rebuild it.

**r2 avatars** (Hiyori, lafei, Mao, huangjiafangzhou_3, 八千代辉夜姬,
seed-san). The default `--avatars` stays `Hiyori,seed-san,lafei`. The body
files of Mao, huangjiafangzhou_3 and 八千代辉夜姬 are only on the CPU
server: copy `rigs/NAME.body.json` (and the rig `.npz`) to the machine that
generates, or the run stops at the start with "avatar NAME: no body file
...". Rebuild 八千代辉夜姬's body for the new variables:

    python -m flow.body --model-json M --rig rigs/NAME.npz --measure

then

    python -m flow.dataset gen --avatars Hiyori,lafei,Mao,huangjiafangzhou_3,八千代辉夜姬,seed-san \
        --teacher-prompt beats --out data/datagen/r2 ...

(NAME entries take `NAME:model=PATH,rig=PATH,body=PATH` where the defaults
do not find them.) 八千代辉夜姬 and the Azur Lane rigs are for training and
tests only, never streamed.

### beats2 (r3 data)

q8r2 (trained on r2, served with `--body-prompt beats`) showed in its eval
and 30 demo clips: about 20% of replies open with exactly the one dense
example of the prompt (brows 0.7, pitch +5, yaw -5, dur 0.2, hold 0.4) and
about 40% with a brow lift + pitch + yaw; dances played in bust without
`[frame full]` while `[frame full]` went to a single wave; about 8% of the
teacher turns answered in another language than the user's latest line. The
`beats` prompt stays byte for byte (q8r2 is served with it; the digests in
`tests/test_facevocab.py` pin it). The changes are a new prompt version,
`beats2` (`pose.PROMPTS`), which is now the dataset default. After the
second live trial (left and right mirrored in 4 of 6 look requests, a wave
request without a wave) beats2 also has two request topics of its own in
`flow/datagen.personas.json`, `look` and `wave` (`"prompts": ["beats2"]`:
v1 and beats draw the old topic list, so r2 regenerates as before; `wave`
is in `dataexport.ASK_TOPICS`), and its look rule gives the sign: "look to
your right" is a negative `head.yaw` / `gaze.x`. The rest of the changes:
`--teacher-prompt beats2` (`beats` regenerates r2, `v1` r1), served with
`--body-prompt beats2` (`flow.director`, `flow.live`; `flow.train.evalsft`
replays the recorded user turns with the registry's prompt, so it needs no
flag). `dataexport` recognises it as beats data (the prompt says
"expression beats"): the same defaults, manners cap `--max-beats`, hands
cap and language filter.

- *Openings.* Six dense examples (`pose.dense_variants`, built per body from
  `pose.offered` like the old one), each eight beats in four to five
  sentences with its own numbers and order. Hiyori's first beats: an eye
  smile with a tilt ("Hehe, you remembered?"); a glance away (gaze.x -0.5,
  yaw -7); a laugh with closed eyes (eye.*.open 0, mouth 1); worried brows
  with a pout (brows -0.6, mouth -0.5); a blush with chin down; a calm slow
  tilt (dur 0.45, hold 1). The two short examples rotate with the variant
  too. The data picks one variant per session from the session's seed
  (`pose.variant_for`; reproducible, recorded as `avatar.variant` in each
  turn record, and every variant's prompt is in `avatars.json` under
  `PERSONA#K`); a served prompt is variant 0 (`system_prompt(variant=0)`),
  and the student saw all of them. One more rule: "Vary how your replies
  open: never start every reply with the same expression, and do not take
  the opening of an example; begin with what this line makes you feel."
- *Camera.* The `[frame]` entry now says: "Write [frame full] right before a
  dance or another whole-body move ([move dance] ... the ones
  `framing.WHOLE_BODY` has that this body does) and before showing an
  outfit, so the whole figure is in the picture; never for a wave or a
  single hand move, which stay in bust." The move example shows
  `[frame full][move ...]`. (The runtime follows whole-body moves anyway;
  this keeps the data consistent.)
- *Language.* Every user turn of beats2 ends with the reply language next
  to the camera line, from the user's latest line:
  `<line>\n(your body now: ...)\n(camera: ...)\n(reply in: English)`
  ("Japanese", "Chinese", "Korean"; left out when the line settles nothing,
  like "OK!"). One function writes it everywhere, `pose.turn_tail`
  (`pose.detect_lang`, which `dataset.detect_lang` is: the main language of
  a mixed line, so English with a kana word is English). The dataset
  (`user_content`), the director (`--say`, `--chat`, VRM too) and the live
  loop call it, so training and serving match exactly. The rule in the
  prompt points at it.
- *Fewer switches.* `datagen.personas.json` is not edited (r2 reproduces
  with it); `--lang-switch P` sets the chance a turn's language is drawn
  afresh, default `dataset.LANG_SWITCH` = 0.03 for beats2 (the personas'
  own 0.10-0.15 for `beats` and `v1`). `mix` is unchanged and each persona
  keeps its `languages` weights, so the balance stays about 40% en, 37% ja,
  23% zh over a run. `dataset.run_opts` reads a run that did not record
  `teacher_prompt` as `beats`.

- *Quiet chat.* The first live trial of q8r2 showed a student that only
  speaks when spoken to. Live now sends `(chat is quiet)` after a while
  without a message (below) and the simulated user does the same on 10% of
  the turns after the first (`--quiet-share P`, `dataset.QUIET_SHARE`; 0
  none; beats2 only, the older prompts regenerate as they were). One
  constant, `pose.QUIET_CHAT`, and one function, `pose.quiet_user`, write
  the line in both. The turn is not a user line: no simulator call, topic
  `quiet`, `quiet: true` on the record, `lang.asked` / `lang.user` the
  session's current language (the language of the last real line), and the
  `(reply in: ...)` line read from that line (`turn_tail(...,
  lang_text=)`). The draw comes from its own generator
  (`dataset.quiet_turn`, from the session seed and the turn), so nothing
  else of a session moves. `dataexport` keeps these turns: `dedupe` leaves
  them out (the line is the same by design) and `asked_for` says they asked
  for nothing, so the manners strip unasked hands as for any chat turn. The
  prompt gets, in its rules (beats2 only): "A user message that is just
  "(chat is quiet)" means nobody has written for a while: keep talking as a
  streamer does when the chat is quiet. Start a small topic of your own,
  react to something you said or saw, ask the viewers a question, or carry
  on with the last topic. Keep it short and natural, and never say that the
  chat is quiet."
- *Looks hold.* Two sentences under the head move: "Your head stays at the
  angle a marker gave it until your next marker sets that angle again (and
  settles back when your reply ends), so tilt from one side to the other as
  you talk and do not count on the idle to bring it back." and, from the
  variables the body has, "When the user asks you to look or turn somewhere
  (left, right, up, down, at the camera), do it in one marker that holds
  that direction: "head.yaw" and "gaze.x", with "hold" 2 to 4. The head stays
  turned until your next head move or the end of the reply, so do not glance
  back after a moment." They describe the playback of the next section; the
  data itself is still played by the dataset as before (it does not carry:
  the trajectories and the action head's targets are unchanged).
- *Longer sessions.* Nothing caps a session: `--turns` / `TURNS` is the
  session length, the personas and topics have no turn count, the
  simulated user sees the whole transcript, and the character sees every
  earlier turn. r3 runs 10 turns, 1600 sessions (16000 turns; r2 was about
  15000). The live window (`--history-turns 5`) is shorter than the
  longest history the character saw (9 exchanges) and as long as a
  6-turn session's, so a request to the student never holds more history
  than it trained on. A session example of 10 turns is about 1.7x a
  6-turn one: `train_sft --max-len auto` (99th percentile, 256-token
  steps, at most 16384) cuts the rest after the last reply that fits.

r3 generation, the r2 avatar list, 10 turns a session, 1600 sessions (about
the turns of r2):

    SESSIONS=1600 TURNS=10 CONC=8 BUDGET_USD=20 OUT=data/datagen/r3 \
    AVATARS=Hiyori,lafei,Mao,huangjiafangzhou_3,八千代辉夜姬,seed-san \
        bash scripts/datagen.sh

or `python -m flow.dataset gen --avatars Hiyori,lafei,Mao,huangjiafangzhou_3,八千代辉夜姬,seed-san
--sessions 1600 --turns 10 --teacher-prompt beats2 --out data/datagen/r3 ...`
(`--teacher-prompt beats2` is the default; the personas file is the same).
Export and train as for r2; serve the student with `--body-prompt beats2`.
八千代辉夜姬 and the Azur Lane rigs are for training and tests only.

### Live after the first q8r2 trial (2026-10)

The student (tsk-q8r2, Mao, `--body-prompt beats`) was talked to for 19
turns. From turn 12 every reply was the same three beats, requested hand
poses did nothing, the first-text latency grew from 0.9 to 2.7 s, the face
went blank between beats, Mao seemed to look left all the time, a requested
look right lasted under a second, the character never spoke unprompted, and
her head tilted in discrete steps. What changed, all in `flow.live` (and
the offline director where it plays the same tokens), none of it in what
the dataset and the action head learn from:

- **History window** (`--history-turns K`, default 5; 0 keeps all). The
  system prompt and the last K exchanges and the current line are sent: a
  request is at most 6 turns, as long as the sessions of r1/r2 (6) and shorter
  than r3's. The model copies its own earlier replies once the chat is
  longer than anything it trained on. `Performer.trim_history` drops the
  older messages for good and logs `history trimmed to the last K
  exchanges (n older messages not sent)`.
- **Head carry** (`--head-carry`, default on; `--no-head-carry`). A pose
  used to hold its target, then release to the idle's value, so each beat
  went out and back to centre. A beat's head and gaze angles now stay where
  it put them: `pose.tokens(..., carry=True)` names them in the token
  (`carry`) and `PoseLayer` holds them until the next token that sets that
  angle, which blends straight from there (min-jerk, the pose-ease times),
  or until an end token. Live sends the end token (`pose.release_token`) at
  the end of each reply's speech: the carried angles relax over
  `CARRY_RELEASE` = 1.5 s (never under 1 s); the baseline below starts to
  fade. Without an end (the reply failed, a quiet stream) they relax after
  `CARRY_MAX` = 20 s. HeadDrift stays under all of it. Carried: pitch and
  tilt always; yaw, gaze.x and gaze.y unless the beat is a reaction
  (`|yaw| <= 6` degrees, `|gaze| <= 0.4`, hold <= 0.6 s), which relaxes
  after its hold as before. The q8r2 replies open with a yaw of -4 or -5 (20
  of the 21 replies in `runs/demo_q8r2` that open with one; 11 -5, 9 -4,
  mean -4.1); carried, that would have been a lasting turn. A look
  (`head.yaw: 25, hold: 3`) stays to the end of the reply and relaxes over 1.5
  s. A reaction in the middle of a look relaxes to the idle, not back to the
  look. The offline director and the VRM render take the same flags
  (`--head-carry` is passed to `flow.runtime`, which writes the end token at
  the speech's end into the tokens); the dataset sets it off.
- **Expression baseline** (`--baseline-scale X`, default 1; 0 is off).
  Each face beat leaves a resting share of itself: `BASE_FRAC` 0.375 of
  how far it moved the face parameter from where the idle had it (`face_params`:
  brows and the unbound eye, mouth and cheek shapes; not angles, hands,
  switches or openings), 0.25 to 0.5 with the reply's mood (`pose.valence`
  of each beat, from its smiles, mouth form, blush and brows; a beat that
  agrees with the mean of the reply so far keeps more, one that does not
  keeps less). It is reached by the end of the beat's hold, kept to the end of the reply
  (`BASE_HOLD_MAX` = 8 s with no end token) and faded over `BASE_DECAY` = 5 s,
  or replaced by the next beat on that parameter, which starts from where
  the face is. It is added to the frame after the drift and before the
  micro-expressions and the beats (`PoseLayer.base_apply`), so both ride on
  it and a beat's release returns to the resting face, not to a blank one.
  Scale 0 is the frames of before, bit for bit.
- **Quiet chat** (`--quiet-after S`, default 20; 0 is never; `--quiet-max
  N`, default 3). S seconds after a reply's speech ends with nothing waiting
  the Performer sends the model `(chat is quiet)` (`pose.QUIET_CHAT`, the
  string the dataset's simulated user sends) with the usual body, camera and
  `(reply in: ...)` lines, the language being that of the last real message.
  At most `--quiet-max` in a row; any real message resets the count, and a
  message that arrives while a quiet turn is spoken waits its turn as
  ever. The voice keeps the conversation's language (a quiet turn does not
  re-detect it from the English line). Logged as `live: turn n: chat quiet
  S s after the last reply; quiet turn i/N`; the page's user event and the
  recorded turn carry `quiet`. The student of r2 (beats prompt) has never
  seen the line; it is only meaningful with the beats2 student.

Why Mao looks to the left. Measured with the real Mao rig (the Cubism
sample, `rigs/Mao.npz`, the calm idle, no GL) through `MotionStream`, 40
seeds of 60 s each, drift on: mean head yaw (ParamAngleX) +0.37 degrees
(the spread between seeds is 2.65, 22 of 40 negative), mean eye-ball X
(ParamEyeBallX) +0.002 (22 of 40 negative), pitch +0.40. The idle has no side.
Mao's own motions do have one (mean ParamAngleX: mtn_04 -4.1, special_01
-9.8, special_02 -20.4, special_03 -5.7; EyeBallX -0.1 to -0.22) but the
gesture library keeps out every parameter a role is bound to, head and
gaze included (checked: none of the two idle gestures writes either), so none of it
reaches the head. What does have a side is the student: in the 30 replies
of `runs/demo_q8r2`, 43 beats set a head yaw, 58% negative, mean -0.72
degrees; 20 of 21 first beats of a reply have -4 or -5. `head.yaw` is
positive toward the character's left (the viewer's right), so -5 is a turn
toward her right, the viewer's left: the same screen side every reply, a
brief turn that starts every answer. The scripted Mao_14 reply (yaw -4,
none, +5, 11 s of speech) averages -0.34 degrees of yaw over its speech
under either playback; in a reply with only the opening yaw -5 the 8 s after
the beat average -0.69 degrees (12 seeds, the idle's own spread) because the
reaction relaxes, where a carried -5 would have stayed at -5. A requested look right (yaw +25, hold 3) was at
+24 at 2 s and back at -0.8 by 5 s; it is now +25 until the end of the
reply at 9 s and back by 11 s.

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

#### Probes: why the grid came out as it did (2026-10)

The grid says no regression head beats a constant on the motion around
the posture, text adds nothing, and prosody heads tie hand speed to
loudness far more than people do. `flow.probe` asks why, with no
controller: two questions, each with the control that would show an
answer to be speaker or recording identity. Spec in the module docstring.
Code and tests ran on synthetic data only (CPU); none of the numbers
below exist yet.

    cd /mnt/e/Tsukuyomi
    bash scripts/beat2_probe.sh                    # prosody + match, ~/tsk-runs/probe
    MODE=prosody bash scripts/beat2_probe.sh       # numpy only, minutes
    LIMIT=2000 STEPS=500 bash scripts/beat2_probe.sh    # a quick look

or by hand (export `data/beat2/seed-san/action`, hidden states
`~/tsk-runs/hidden/beat2`, layers `[36, 29, 18]` projected to 1024):

    python -m flow.probe prosody --export data/beat2/seed-san/action \
        --out ~/tsk-runs/probe
    python -m flow.probe match --export data/beat2/seed-san/action \
        --hidden ~/tsk-runs/hidden/beat2 --out ~/tsk-runs/probe --layer 1
    python -m flow.probe match ... --split speaker --layer all   # new speakers
    python -m flow.probe all ... --sem ~/BEAT2/sem               # + semantic windows

Both write `report.txt` and `report.json` (`prosody`, `match`). `--layer`
takes an index into the stored layers (1 = layer 29, the default), a layer
number, or `all` (a learned mix, three times the RAM; estimate: the states of
one layer are ~3 GB). `--limit N` keeps whole clips until N segments are in.
The budget of `match` is `--steps 3000 x --batch 128` per variant, six
variants. Measured on CPU with random states (D = 1024, 128 windows of
~20 tokens): assembling a batch takes 10 ms for one layer, 20 ms for all
three, 60 ms at 48 tokens a window; scoring 256 windows 0.04-0.15 s. That
is the cost that stays on the CPU on a GPU machine, so a variant is about
2-6 minutes (3000 steps, 12 curve checks, the final scoring), the six of
them 15-25 minutes for one layer and up to ~40 with `--layer all`, plus
loading the shards. An estimate from those numbers, not a run on the 3090
Ti. `prosody` is numpy and takes minutes.

**prosody** (the ground truth only). Per segment, six voice features
(loudness, its range, voiced share, f0 and its range, speaking rate in
units per voiced second -- characters for zh / ja, space-separated words
for en / es --, pauses) against the hands' and head's amplitude (std and
p95 - p5 around the segment mean), speed (mean, p90) and the share of
frames with the hands moving; then 2 s windows with a lag scan, and the
frame-level accent correlation of the grid (reproduces the 0.03). Every
number has a 95% CI from resampling whole clips. Read the
`within speaker` tables (z-scored per speaker, then pooled), not
`overall`: where the two differ, speaker style was driving it. The
`no dur` table removes segment length (a longer segment has more spread by
itself). In the tables `*` marks a CI that excludes 0, the last two
columns average |r| over the four amplitude and the four speed columns.
The two hypotheses are at the end of the report, with the paired
difference of the two |r| and its CI:

- loudness moves the amplitude more than the speed: `loud` row, `hand_std`
  / `head_std` against `hand_v` / `head_v`; supported when the difference
  is positive with a CI above 0. A lag scan that peaks away from 0 says who
  follows whom (positive: motion after the voice).
- the speaking rate moves the speed more than the amplitude: `rate` row,
  the same columns the other way round; supported when `hand_v` beats
  `hand_std` with a CI above 0.

Amplitude and speed are not independent measures (a faster swing of the
same size is faster), so read the whole row, not one cell. If the within-
speaker numbers are small (|r| under 0.1-0.2) the voice carries little of
the motion in the mocap itself, and a head that reads loudness as speed
has learned something the data does not support.

**match** (text and prosody against motion). 3 s windows; the motion, the
tokens from 1 s before the window to its end, and the window's prosody are
encoded to 128-d and matched with a symmetric InfoNCE. Each variant is
scored on held-out clips (or speakers) as top-1 / top-5 retrieval among
K = 32 candidate motions for a text, under three ways of drawing the 31
wrong ones: `random` (another clip), `speaker` (the same speaker, another
clip) and `clip` (the same recording, a different time). Chance is 1/32 =
0.031 and 5/32 = 0.16.

| variant | trained on | tells you |
|---|---|---|
| text, prosody, text+prosody | batches of one speaker | what each carries about the motion once the speaker is no help |
| text_shuffled | the tokens of another window | the pipeline: must be at chance everywhere (a CI above chance is a leak) |
| text_random | random batches | how much of the score is speaker or language, which random negatives allow; compare its `random` column with `speaker` and `clip` |
| text_clip | batches from one recording | the hardest: speaker, recording and language the same |

How to read it:

- Text carries gesture information if `text` is clearly above chance in the
  `speaker` and `clip` columns (the CI's low end above 0.031, and more than
  `text_shuffled`) on held-out clips; the `clip` column is the strictest.
  Better still on `--split speaker` (speakers never seen in training).
- Text beyond the voice: `text+prosody` above `prosody` in the same column.
  `prosody` above `text` means the audio is the signal and the LLM states
  add nothing, as the grid's d - c suggested.
- Everything at chance in `speaker` / `clip` but high in `random` for
  `text_random` is the speaker-identity finding: the states say who is
  talking and in what language, not what the hands do.
- All variants near chance with train top-1 rising in the curves is
  overfitting of a probe that is too large for the signal (the text is
  1024-d per token): lower `--steps`, or `--dim 64`, before concluding.
  The report is the last step; nothing is chosen on the held-out windows.
- `--center` removes the window's mean pose, leaving only the gesture
  (posture identifies a recording and a speaker; it is what the MSE grid
  mostly scored).
- `--sem DIR` (BEAT2's semantic annotations; the format is detected, not
  known: a file `<clip>.txt|csv|tsv|json` with start and end seconds, and
  a gesture type or a relevance score) splits the accuracy into windows
  that overlap a semantic gesture and those that do not; if nothing parses
  it says so and goes on. Text helping only on the semantic windows would
  be the case for a text-driven gesture layer for the rare semantic ones.

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

#### Probe results (2026-10-28, runs/probe/report.txt)

- **Voice vs motion in people** (segment level, within speaker): loudness
  r ~0.21-0.28 with both gesture range and speed (speed slightly the
  stronger: the "loudness -> amplitude" hypothesis is not confirmed, both
  rise); pitch and pitch range ~0.15-0.20; speaking rate -> speed (0.17) >
  range (0.13), as hypothesised; pauses -> less motion. Chinese couples
  more strongly (loudness -> head speed 0.48). At the frame level the
  coupling is ~0.03-0.09: voice sets how much a phrase moves, not when
  each frame moves. Windowed lag scan is flat, hands slightly leading the
  audio (best -0.4 to -0.7 s), as gesture studies report.
- **Text carries motion information.** Contrastive matching, held-out
  clips, 32 candidates (chance 0.031): text top-1 0.153 against
  same-speaker candidates and 0.098 against the same recording; prosody
  0.149; text + prosody 0.204 (the two add up); wrong text 0.035
  (chance). The grid's "text adds nothing" was the per-frame regression,
  not the text: what text predicts is the kind and amount of motion over
  a few seconds, which per-frame error cannot score.
- So: condition on phrase-level voice features (not per-frame loudness),
  keep text, generate (sample) rather than regress, and score generators
  the way text-to-motion work does: retrieval precision and embedding
  distances with these contrastive encoders, not frame MSE.

### Gesture search (2026-10)

`flow/gesture_search.py`: between and under the model's explicit markers, a
VRM avatar plays real human co-speech motion retrieved from the BEAT2
library. It is the use the probe results point to ("condition on phrase-
level features, keep text, retrieve or sample, judge by retrieval"), built
on the contrastive encoders instead of a regression head. Built and tested
on synthetic BEAT2-shaped data (`tests/test_gesture_search.py`: tiny
encoders trained on a toy export where the text names the arm's swing); it
has not run on the real library, so the first run on the GPU machine is the
check (below).

How it works (the module docstring has the details):

- **Encoders kept.** `flow.probe match --save DIR` writes each trained
  variant (state dict, motion normalisation, columns, window / look-back /
  token settings, the layers the text encoder reads, the hidden folder's
  `meta.json`, the held-out and training clips). The report is unchanged
  without it. `flow.probe.load_encoders` embeds motion windows and
  (token states, prosody) from a saved folder.
- **Library** (`build`): every 3 s window of the export (hop 0.5 s) except
  the probe's held-out clips, embedded from the canonical `.traj` columns as
  the probe's motion encoder reads them; the 23 fitted joints (`.joints`)
  stored once per segment, a window is (segment, start); start / end poses,
  speaker, clip, language per window; the median pose and the recordings'
  median voiced RMS. About 0.35 KB per window (a 128-d fp16 embedding and
  two poses); the joints (46 bytes a frame) are stored once, not per
  window. Under `data/gesture_lib/` (gitignored).
- **Query** (per sentence of the TTS reply): text states as `flow.train.
  hidden` made them for the probe -- same model, `--context none`, the
  spoken text, the stored layers, the stored projection, fp16, then the
  probe's layer selection -- run in the director's process
  (`TextStates`; `tests/test_gesture_search.py` checks bit equality with
  what `hidden` stores); prosody from the sentence's TTS audio at 30 fps
  (`flow.beat2.prosody`) after its loudness is set to the library's (TTS and
  microphone levels differ by more than the encoder's input tolerates);
  the sentence itself is made to look like the export's texts first (BEAT2's
  words are the TextGrid's: `build` records whether they are lowercase and
  have punctuation -- `text_form` in the library's `meta.json` -- and a
  sentence is lowercased / stripped of punctuation accordingly, NFKC, the
  characters' times kept on their characters; a library without `text_form`
  leaves the text as it is);
  windows of 3 s every 2 s over the sentence plus 0.3 s either side, each
  embedded as the probe embeds a window (silence past the audio).
- **Choice**: cosine to the library's motion embeddings (optionally
  one speaker / language), the best 48 scored cosine - 0.25 x continuity -
  a small price for each recent pick from the same clip, one of the best 6
  drawn with a softmax at temperature 0.05 (`--gs-temperature 0`: the best;
  `--gs-seed`). Continuity: at the moment the cross-fade starts, the arm
  joints (share of their range) and the palms (hu, the target body's
  forward kinematics) of the candidate against the pose the track is in.
  A window overlapping a recent pick of the same recording is out.
- **Body**: the arms (upper arm, forearm, wrist, shoulders) are absolute,
  scaled about the body's rest pose (`--gs-amp`) and clamped to its limits;
  `--gs-parts full` adds the head and torso as the window's deviation from
  the library's median pose, added to the idle's head and trunk (so the
  idle's own head moves keep going). Each chosen window goes through the
  body's collision model; where an arm is deeper than 0.04 hu in the trunk,
  head or the other arm it is solved again with `beat2.fix_contacts` (the
  collision-aware IK); a window still deeper than 0.08 hu is skipped for
  the next best. The cross-faded track is checked once more (a blend of
  joint angles is not collision-free by itself). Retargeting to another
  VRM is applying the same joint names to its body (clamped, collisions
  re-checked): `tests` run a Seed-san window on the three-vrm-girl body.
  Live2D is out of scope (its arms are drawings).
- **Play**: one `"gsearch"` token per run of sentences closer than 1 s: the
  windows cross-faded over 0.4 s (minimum jerk, the later window taking
  over as it fades in), eased in and out over 0.4 s. `GestureLayer` sits in
  `MotionStream.frame` after the idle and the mouth, before moves and
  poses: a pose or move that plays an arm takes it exactly as it takes it
  from the idle, the gesture ducking out of those parameters before the
  marker starts (0.4 s) and back in after its hold; nothing of the
  gesture shows in a held pose (the layer reports the parameters it plays
  as `holding`, so the pose layer's "life" ignores them). Without
  `--gesture-search` none of this is imported: `tests` compare the plan's
  tokens, states, subtitles and actions byte for byte with the module made
  unimportable.

Director flags (`--avatar vrm`; they need `--tts`, the prosody is the
audio's): `--gesture-search LIB --gs-encoders DIR [--gs-model PATH]
[--gs-hidden DIR] [--gs-amp 1] [--gs-seed 0] [--gs-parts hands|full]
[--gs-speaker NAME] [--gs-temperature 0.05] [--gs-topk 6] [--gs-4bit]
[--gs-device auto|cuda|cpu] [--gs-baseline]`. `--gs-baseline` also writes
`states_markers.json`, the same turn without the retrieved motion
(`demo --say` uses it). The text model is loaded once per session, on the
first sentence. `flow.live` does not take the flags yet (its `Follower`
plans per sentence and has the audio: `GestureSearch.plan` takes queries,
so it is the place to hook it).

On the GPU machine (WSL, repo at `/mnt/e/Tsukuyomi`; the export
`data/beat2/seed-san/action`, the hidden folder `~/tsk-runs/hidden/beat2`
extracted with layers [36, 29, 18], `--proj 1024`,
`--text spoken --context none`, the merged model `~/tsk-runs/sft/q8/merged`):

    cd /mnt/e/Tsukuyomi
    # 1. the text+prosody encoders again, kept (the settings of the report
    #    you want to match: --layer 1 is the stored layer 29; the split
    #    follows from --seed / --eval-frac / --limit and is recorded; --out
    #    is another folder so the earlier report.txt / report.json stay)
    python -m flow.probe match --export data/beat2/seed-san/action \
        --hidden ~/tsk-runs/hidden/beat2 --out ~/tsk-runs/probe-gs \
        --layer 1 --variants text+prosody --save ~/tsk-runs/probe/gs
    # 2. the library (BEAT2-derived: research only, stays under data/)
    python -m flow.gesture_search build --export data/beat2/seed-san/action \
        --encoders ~/tsk-runs/probe/gs --out data/gesture_lib/seed-san
    # 3. retrieval precision and motion statistics on the held-out clips
    python -m flow.gesture_search eval --export data/beat2/seed-san/action \
        --encoders ~/tsk-runs/probe/gs --lib data/gesture_lib/seed-san
    # 4. held-out segments side by side: idle | random windows | retrieved |
    #    ground truth, the recorded audio on it
    python -m flow.gesture_search ids --export data/beat2/seed-san/action \
        --encoders ~/tsk-runs/probe/gs
    python -m flow.gesture_search demo --id beat2.<clip>/s03 \
        --export data/beat2/seed-san/action --encoders ~/tsk-runs/probe/gs \
        --lib data/gesture_lib/seed-san --beat2 ~/BEAT2 --out runs/gs/held
    python -m flow.gesture_search demo --id beat2.<clip>/s03 ... \
        --vrm vrm/toukai_teio.vrm --out runs/gs/toukai     # a body of yours
    # 5. a free sentence through the director: markers only | markers + search
    python -m flow.gesture_search demo --say "Come here, I want to show you something." \
        --export data/beat2/seed-san/action --encoders ~/tsk-runs/probe/gs \
        --lib data/gesture_lib/seed-san --out runs/gs/say-oguri \
        --gs-model ~/tsk-runs/sft/q8/merged --gs-hidden ~/tsk-runs/hidden/beat2 \
        --avatar vrm --vrm vrm/oguri_cap.vrm --mode body --backend ollama \
        --tts voicevox --size 384
    # the same inside a normal director run
    python -m flow.director --avatar vrm --vrm vrm/toukai_teio.vrm --mode body \
        --backend ollama --tts voicevox --say "..." --out runs/gs/turn --render \
        --gesture-search data/gesture_lib/seed-san \
        --gs-encoders ~/tsk-runs/probe/gs --gs-model ~/tsk-runs/sft/q8/merged

`demo --say` takes the director's own flags as they are (everything it does
not know, `--mode`, `--backend`, `--tts`, `--vrm`, ...) and `--gs-model`
for the text model (`--model` stays the director's language model).
The two VRMs of yours are not in the repo: the paths above are where they
are expected (`vrm/` is ignored); their body files are built on first use
(`rigs/<name>.body.json`).

What to check first on the real library: that step 1's held-out text+
prosody top-1 (32 candidates) is the old report's 0.204 -- if not, the
flags (`--layer`, `--limit`, `--seed`, `--eval-frac`) differ from that run
and the held-out clips are not the same ones; that `meta.json`'s
`text_form` is what you expect of the transcripts (lowercase, no
punctuation: otherwise TTS sentences are read as they are); that `eval` says retrieved
windows are closer to the true motion than random ones by clearly more than
chance (0.5; the probe's text+prosody top-1 of 32 was 0.204 against 0.031,
so expect a modest margin: what text and voice predict is the kind and
amount of motion over a few seconds, not the gesture), the speaker-
normalised amplitude and speed (retrieved close to the truth, a random
window's correlation with the truth near 0), then the demo by eye -- the
cross-fades and the arms against the body -- and the prosody gain (the
library's median voiced RMS is in `meta.json`; a TTS voice far from it is
the first suspect if every sentence retrieves the same still gesture).

Cautions:
- The library is BEAT2-derived (retargeted mocap): research use under
  BEAT2's licence, kept out of git (`build --out` must be under `data/`).
- Text states: the hidden folder does not record whether `--load-4bit` was
  used; query with the same precision (`--gs-4bit` only if extraction used
  it), or the states shift. `--context none` states are what the probe was
  trained on; a hidden folder made with `--context turn` is said to differ.
- Latency: one forward of the text model per sentence in the director's
  process (a sentence is ~10-40 tokens: tens of ms on a GPU for an 8B model
  in bf16, seconds on CPU -- an estimate, not measured on the real model,
  only on a tiny one here) plus the encoders (milliseconds) and the checks
  of the chosen windows (about 40 ms per window measured on Seed-san, 0.3-
  0.5 s when the contact fix runs). Offline this is paid once per reply,
  before the render; the model load (tens of seconds) once per session. The model is 16 GB in bf16: next to Ollama on one 24 GB card use
  `--gs-4bit` (and extract the hidden states the same way) or run the
  search on another GPU / the CPU (`--gs-device`); the encoders and the
  library (200-300 MB of embeddings for ~400k windows, in RAM) are small.
- Continuity artefacts: the cross-fade blends joint angles, so two windows
  that differ much at the fade can pass through a pose neither has (the
  track is re-checked for collisions and solved again where it goes into the
  body, but a bent wrist or an elbow swinging through is not a collision);
  the continuity weight (0.25) is a first guess, untested on real data.
  The first window of a run eases in from the rest pose, so a window that
  starts with the hands high is penalised by the same cost.
