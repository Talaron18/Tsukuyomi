# Research survey: motion control and beyond (2026-10)

What recent work (2025 to 2026-10) offers this project: a local Qwen3-8B on
one RTX 3090 Ti (24 GB, Ampere: no FP8) driving Live2D/VRM bodies, moving
from inline markers to an action head on the model's hidden states
(docs/embodiment.md, Stage 2), and later to video, games, driving and
hands.

**How far to trust this.** Four research passes, 2026-10-26. From their
sandbox arxiv.org and huggingface.co were mostly unreachable, so most paper
claims come from search snippets; GitHub pages were read directly. Marks:
**[repo]** the GitHub page was read; **[snippet]** search results only,
not read. Speeds are the authors' claims, mostly on newer GPUs; nothing
here was run on our card. arXiv ids from 2606 on could not be cross-checked
at all. Check a paper before building on it.

## 1. Motion control (the current focus)

### The interface: what reads the LLM

Our plan (per-token hidden states, aligned to the speech, to a small head)
has close published relatives:

| Work | How the LLM drives motion | Status |
|---|---|---|
| GestureFAR (arXiv 2609.21576, 2026-09) | AR backbone, a per-token flow-matching head on its hidden states, distilled to one step; streaming, body + face | [snippet]; repo linked but empty |
| LLaMo (CVPR 2026) | LLM + motion-side parameters, flow head on the motion hidden state; claims 30+ fps streaming | [snippet] |
| UniMotion (arXiv 2603.22282) | modality-specific flow heads from LLM hidden states to motion latents | [snippet] |
| MotionGPT3 (arXiv 2506.24086) | LLM hidden states condition a diffusion head over motion-VAE latents; body only, not streaming | [repo] MIT code, HF weights |
| ELLSA 2 (arXiv 2610.21452) | full-duplex dialogue model; speaking/listening hidden states drive gestures + face in 0.8 s chunks | [snippet]; HF tsinghua-ee/ELLSA2, licence unknown |
| OpenHelix (2505.03912), RationalVLA (2506.10826) | a learned `<ACT>` token; its hidden state is the goal for a low-level policy | [snippet]; OpenHelix code open |
| FLOWER (2509.04996) | VLM hidden states into a flow transformer by cross-attention | [snippet] |

Takeaways:
- **Per-token conditioning plus a flow head is the mainstream design.**
  flow/actionhead.py already has this shape (cross-attention over token
  states, regression or flow matching).
- **One-step distillation** (GestureFAR's head distillation, SnapFlow
  2604.05656) is the known way to keep a flow head cheap enough for 30 fps.
- **The `<ACT>` token is the alternative**: one learned token per control
  step instead of reading every speech token. It's worth an ablation once
  the per-token version has a baseline.
- **Layer choice matters and nobody agrees.** Try `-8`, `-16` and the middle
  layer; the extractor takes `--layers=-8,-16`.

### Latency and streaming

- **Real-Time Chunking** (Physical Intelligence, 2506.07339, [snippet]):
  frames the actions already committed during inference as an inpainting
  constraint, at inference time only. This applies to any chunked head
  here; it's the fix for the "audio waits on motion" gap.
- **FASTER** (2603.19199, [snippet]): a flow schedule that produces
  near-term actions first, so the first action comes after one step.
- **LiveGesture** (CVPR 2026, 2604.10927, [snippet]): a causal motion
  tokenizer with hierarchical AR over body regions, with no look-ahead.
  It claims under 50 ms per 200 ms audio chunk; a good latency budget to
  copy.
- **Cheap baseline first:** OpenVLA-OFT (2502.19645) reports an L1
  regression head matching diffusion heads. Our head's regression mode is
  that baseline; flow matching has to beat it to be worth its cost.

### Two layers: plan, then track

Helix (Figure: a 7B VLM at 7-9 Hz over a fast visuomotor policy at
200 Hz), Gemini Robotics 1.5 (planner then actor) and HOVER (one tracking
policy for several command types) all split slow intent from fast control.
That is what the marker path already is: the LLM gives targets, and the
solver and motion stream track them. The action head should keep that
split: predict targets or short chunks at token rate, and leave
smoothing, idle and breath to flow.stream.

### Data beyond the teacher

The head can only learn what its targets hold, and today those are the
trajectories DeepSeek's markers produced. Real motion would raise that
ceiling:

| Source | What | Licence notes |
|---|---|---|
| BEAT2 | conversational co-speech mocap (face, body, hands), 25-30 speakers, ~60-76 h | Apache-2.0 per one secondary source; check the HF tag |
| FloodDiffusion (CVPR 2026) | streaming text-to-motion, ~50 fps output | [repo] Apache-2.0, HF weights; trained on HumanML3D/BABEL |
| FloodDiffusion 2 (2609.33167) | adds root-path conditioning | [repo] MIT, HF weights, run on an RTX 4090 |
| SemTalk (ICCV 2025) | holistic face/body/hands, per-part RVQ tokenizers | [repo] weights on HF, code licence not stated |
| HumanML3D / AMASS | text-motion | the AMASS terms forbid redistribution: rebuild it locally, research use |
| Seamless Interaction (Meta, 2506.22554) | 4,000+ h of dyadic video with motion/expression models | licence not found |

The VRM body is a standard humanoid, so SMPL-X mocap retargets to it
directly; that's step E4 (GMR, 2510.02252, is a reference retargeter).
Live2D only takes what the canonical variables can express: head, torso,
arms and face.

### Front ends (for comparison)

Open-LLM-VTuber ([repo] MIT, Live2D, works with any OpenAI-compatible
server) and Project AIRI ([repo] MIT, Live2D + VRM) are the popular open
VTuber stacks. Neither has body control beyond blinks, look-at and idle.
Our solver, poses, moves and data engine are further along than both; at
most, borrow their UI or plugins.

### RPent

github.com/RLinf/RPent, "Recursive Physical Agent" ([repo] per the
research pass, Apache-2.0, updated 2026-09): an agentic planner (Claude
Code, Codex or a custom planner) calls VLAs such as pi0.5 as tools, in
simulators (LIBERO-PRO, RoboCasa, RoboTwin) and on a Franka arm. It's a
planner above the VLAs, not hidden states driving a body: the reference
for a future planning layer on top of our controller, not for the head.

### Read in full (papers/, 2026-10-26): what changes for flow/actionhead.py

The nine priority-1 papers are in papers/; all the arXiv ids above for them
are correct. What they say, against the head as built (frames cross-attend
to the token states up to a look-ahead, chunks with half overlap
cross-faded, MSE or flow matching, no motion history, no audio):

- **Condition on past motion.** GestureFAR (a causal transformer over its
  own past motion latents + audio so far, an MLP flow head per latent at
  7.5 Hz) and LiveGesture (region-expert AR transformers over causal motion
  tokens) both autoregress on the motion already emitted. Our head sees
  only text, so continuity comes from the cross-fade. LiveGesture's
  **autoregressive masking** (training on corrupted history) is the fix for
  the drift that history feeding brings.
- **Replace the cross-fade.** RTC shows that temporal ensembling (what the
  cross-fade is) is less smooth than **inpainting**. The next chunk is
  generated with the frames already committed frozen, and a soft mask over
  the overlap. It's training-free for any flow head, so it applies to
  `--objective fm` as is.
- **Timing comes from speech, not the LLM.** OpenHelix's probe found the
  MLLM latent carries task semantics and is insensitive to fast changes.
  GestureFAR conditions on audio (mHuBERT), and ELLSA 2 on the speech
  synthesiser's own hidden states. Our head gets per-character speech times
  but no prosody. r1 was generated with `--tts none` (rate timing), so
  adding audio features (energy and pitch from the TTS clip, which the lip
  sync already computes) needs a TTS pass over the records.
- **Which layer:** ELLSA 2 conditions on the final layer and mixes layers
  16/20/24 for its call head; OpenHelix reports both final and middle layers
  in use. So learn a weighted mix of a few layers rather than picking one.
- **Freeze, align, then (maybe) tune.** OpenHelix: joining the LLM and the
  policy without first training the projector with the LLM frozen fails
  outright (0% in their Table 6). Prompt-tuning one `<ACT>` token
  generalised best. Our order (frozen extraction, head trained on cached
  states) is the right first stage. MotionGPT3 adds that a separate motion
  branch keeps the language ability intact.
- **Keep explicit gestures explicit.** ELLSA 2 separates *whether* to make
  a semantic gesture (a binary call head on layers 16/20/24, which can run
  ahead of the speech) from *how* (a retrieved exemplar guiding the
  generator). That is our split already: markers / [move] / gestures are
  the exemplars, and the head does the co-speech motion. A call head could
  later replace the textual markers for gestures without losing precision.
- **Listening.** ELLSA 2 keeps generating listener motion (nods,
  expressions) while the user talks. We only have the idle there.
- **Making flow cheap:** first check OpenVLA-OFT's result (L1 regression ≈
  diffusion, faster) against our MSE mode. If flow wins, distil the head
  alone: GestureFAR distils only the per-token head on cached conditions
  (consistency warm-up + DMD); SnapFlow self-distils to one step with no
  separate teacher or critic, and beats its 10-step teacher on LIBERO.
  FASTER's horizon-aware schedule makes the first frames one step, the
  lever if time-to-first-motion matters more than chunk quality.
- **Runtime:** GestureFAR's stateful runtime (persistent caches, body
  anchors against drift, a short playback buffer, slerp blends between
  speaking, listening and idle) matches what flow.stream already does for
  markers. The head should feed the same stream, not replace it.

## 2. Later directions

### Live co-watching and long video

| Work | Why | Notes |
|---|---|---|
| MiniCPM-o 4.5 (2604.27393) | full-duplex video/audio/speech; **its LLM core is Qwen3-8B**, the same as ours | [snippet]; GGUF builds exist; check the licence |
| JoyAI-VL-Interaction (JD, 2026-06) | each second: speak, stay silent, or hand off; initialised from Qwen3-8B | [snippet]; INT4 variant |
| LiveCC-7B (CVPR 2025) | continuous commentary at ~2 fps; the cheapest working baseline | [repo] code + HF checkpoints, licence not stated |
| Proact-VL (2603.03447) | a per-second "speak or stay silent" head; gaming commentary benchmark | [snippet] |
| StreamingVLM (2510.09608) | an attention-sink + recent-window KV cache for hours-long streams | [snippet]; code not confirmed |

First experiment: LiveCC-7B (or MiniCPM-o 4.5) on a 1-2 fps screen capture
of a recorded stream. Log time to first token, and how often it speaks or
stays silent; pass what it says to the character as a user line.

### Games from the screen

- **Labelling first.** No open keyboard-and-mouse FPS agent turned up.
  Record your own play with an input log and train an inverse dynamics
  model, as in "Pixels to Keys" (2609.37907, [snippet]: score each key by
  macro-F1, not accuracy).
- **NitroGen** (NVIDIA, CVPR 2026, [repo]): a 500M DiT that sees only the
  last frame and outputs gamepad actions; runs on Windows games only, for
  research use. A behaviour prior for gamepad games, not an FPS policy.
- **JARVIS-VLA** (Qwen2-VL-7B, Minecraft keyboard and mouse) and **Lumine**
  (30 Hz keyboard and mouse, reasons only when needed) are the
  architecture references. Lumine's "think only when needed" loop is the
  same idea as a slow planner over a fast body.
- **Matrix-Game 4.0** (Skywork, [repo] Apache-2.0, 2026-10) and 2.0 are
  playable world models (FPS and driving scenes) usable as a sandbox. The
  quoted speeds are FP8 on RTX 50-series cards, so measure on ours.

### World models

V-JEPA 2 ([repo] MIT code, encoders 300M-1B, an action-conditioned
predictor released) is the most runnable option. Freeze the encoder and
train a small action-conditioned predictor on your own frames plus inputs,
then compare it against copying the last frame. Genie 3, SIMA 2 and Cosmos
3 Super are closed or too large; Cosmos 3 Nano's real size needs checking.

### Driving with a pan-tilt camera

- **CARLA + Bench2Drive** (closed-loop, 220 routes) as the test harness.
  **SimLingo** (CVPR 2025, camera-only VLM driver, code and data open) as
  the first model to run on one GPU.
- **EyeVLA**, "The Robotic Eyeball" (2511.15279, [snippet]): a fine-tuned
  Qwen2.5-VL-7B aims a pan-tilt-zoom camera. It's the closest precedent for
  treating the camera's direction as an action. First experiment: SimLingo
  on a few Bench2Drive routes, then add camera pan as an extra action.
- Alpamayo 1 (NVIDIA, 10B) needs the whole 24 GB card for inference; use it
  as a reference only. A Donkey Car is the cheapest real camera-only car
  (about US$500 plus a Pi; a listing price).

### Robot hands

- **Simulation first, at no cost:** MuJoCo Playground has a LEAP-hand
  in-hand reorientation task with sim-to-real.
- **Cheapest real training loop:** an SO-101 arm (LeRobot, about US$100
  DIY base) with **SmolVLA** (450M, a flow-matching action expert, trains
  on one GPU from about 50 demos). SmolVLA's action expert is also a good
  reference for our own head.
- **Hands:** LEAP Hand (v1 about US$2,000 in parts; a cheaper v2), ORCA
  (tactile, under 2,000 CHF). DexGraspVLA (VLM planner over a diffusion
  controller) is the grasping template.
- pi0/pi0.5 LoRA needs more than 22.5 GB (openpi README, read directly),
  so it can't share the card with the LLM; use them as teachers or
  references.

## 3. What to do with this, in order

1. **The scheduled offline action-head test**, with the regression head
   first as the OpenVLA-OFT-style baseline, then flow matching, and three
   layers (`-8`, `-16`, middle).
2. **If flow wins:** distil it to one step (GestureFAR / SnapFlow) and add
   Real-Time Chunking in flow.live, so motion never holds back the audio.
3. **Raise the data ceiling** for VRM avatars: retarget BEAT2 (once its
   licence is confirmed) and use FloodDiffusion 2 as a body prior.
4. **Serving hidden states** (not done yet): llama-server doesn't stream
   intermediate layers. The options are a llama.cpp evaluation callback,
   llama-cpp-python, or transformers with hooks; GestureFAR and ELLSA 2 are
   the designs to read first.
5. **Later, in any order:** a co-watching baseline on MiniCPM-o 4.5 or
   LiveCC (MiniCPM-o shares our Qwen3-8B core); an inverse dynamics model
   from your own game recordings; SimLingo in CARLA with camera pan as an
   action; the LEAP hand in MuJoCo Playground before buying hardware.
