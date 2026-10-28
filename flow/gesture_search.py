#!/usr/bin/env python3
"""
Gesture search (motion matching) for VRM avatars: between and under the
language model's explicit markers, real human co-speech motion retrieved
from the BEAT2 library (docs/embodiment.md, "Gesture search").

flow.probe's contrastive encoders say which 3 s of motion goes with which 3
s of speech: motion <-> (text states, prosody). Here the trained
text+prosody encoder embeds a spoken sentence, the motion encoder's
embeddings of the library's windows are searched by cosine, and the chosen
windows' joint parameters (the 23 vrmbody.JOINTS the retargeter fitted) are
cross-faded into a track the stream plays under the poses and moves.

  library   `build`: every 3 s window (hop 0.5 s) of the export's
            segments, but the probe's held-out clips (the encoders' config
            lists them: the split is not recomputed), embedded from the
            canonical `.traj` columns exactly as the probe's motion
            encoder reads them; the joints (`.joints`) are stored once per
            segment, a window is (segment, start). Per window: the
            embedding, the start and end poses, speaker / clip / language.
            Beside: the median pose, the median voiced RMS of the
            recordings (the loudness a sentence's audio is brought to),
            the encoders it was built with. data/gesture_lib/<avatar>
            (BEAT2-derived: research only, gitignored, never committed).
  query     a sentence of the reply: its spoken text, the characters'
            times and its stretch of the TTS audio. Text states as
            flow.train.hidden made them for the probe (the same model,
            --context none, the spoken text, the stored layers, the same
            projection), run in-process (TextStates; 4-bit optional -- the
            hidden folder does not record how it was extracted, so use the
            same precision); prosody as flow.beat2 computes it (30 fps, the
            audio's loudness set to the library's median first: TTS and
            microphones differ by far more than the encoder tolerates).
            Windows of 3 s every --gs-hop s over the sentence (+ 0.3 s
            either side, as the export's segments), each embedded as the
            probe embeds a window: the tokens in [start - look-back, end]
            with their times, the prosody frames (silence past the audio).
  choose    per window: cosine to the library's motion embeddings (a
            speaker / language filter first); the best --gs-pool; each
            scored cosine - continuity - repeat, then one of the best
            --gs-topk drawn with a softmax at --gs-temperature (0:
            the best). Continuity is the distance, at the moment the two
            windows cross-fade, between the pose the track is in and the
            candidate's: the arm joints (as a share of their range) and the
            palms (hu, on the target body's forward kinematics). Repeats:
            a candidate overlapping a recent pick of the same recording is
            out, one from the same clip pays for each recent pick. Random
            from --gs-seed, so a reply never plays twice the same.
  body      the target VRM body (flow.vrmbody): joints are the same names
            on every VRM body, so a window is played on any of them. The
            arms (upper arm, forearm, wrist, shoulders) are absolute,
            scaled about the body's rest pose by --gs-amp and clamped to
            its limits; --gs-parts full adds the head and torso, as the
            window's deviation from the library's median pose added to the
            idle's own head and trunk. Each chosen window is checked with
            the body's collision model (arm in the trunk, head or the other
            arm); past flow.beat2's contact threshold the arm is solved
            again where it penetrates (beat2.fix_contacts, the collision-
            aware IK); a window still deep after that is not played and the
            next best is tried. The blended track is checked again (a
            cross-fade of joint angles is not collision-free by itself).
            Live2D is out of scope: its arms are drawings, not joints.
  play      GestureLayer, MoveLayer's sibling: per sentence-chain one
            "gsearch" token holding the whole track (windows cross-faded
            over --gs-fade s, minimum jerk, the later window taking over as
            it fades in), eased in and out (0.4 s) at its ends, chained
            across sentences closer than 1 s. The layer sits under the
            moves and poses: an explicit pose or move that plays an arm
            (or the head where it sets it) takes those parameters exactly
            as it takes them from the idle, the gesture ducking out of
            them before the marker starts and back in after it ends (the
            layer is told of the markers when they are directed); nothing
            the gesture does shows through a held pose. No marker: nothing
            of this runs, and the stream's bytes are those of before.

Commands (flow.director's flags: --gesture-search and --gs-*; `ids` lists
held-out segments to demo):
    python -m flow.gesture_search build --export data/beat2/seed-san/action \\
        --encoders ~/tsk-runs/probe/gs --out data/gesture_lib/seed-san
    python -m flow.gesture_search eval --export data/beat2/seed-san/action \\
        --encoders ~/tsk-runs/probe/gs --lib data/gesture_lib/seed-san
    python -m flow.gesture_search demo --id beat2.2_scott_0_1_1/s03 \\
        --export ... --encoders ... --lib ... --beat2 ~/BEAT2 --out runs/gs
    python -m flow.gesture_search demo --say "..." --export ... (a free
        sentence through the director: markers only beside markers + search)

  eval      the held-out segments' windows, queried with their real text
            states and prosody (the hidden folder, the recorded audio's
            features): per query the retrieved window (best by cosine) and
            a random library window against the true motion; reported
            (a) the share of queries where the retrieved window's motion
            embedding is closer (cosine) to the true motion's than the
            random one's (chance 0.5; clip-bootstrap CI), (b) the probe's
            own question on the library (the true motion's rank by
            cosine to the query among K - 1 random library windows), (c)
            amplitude and speed of the hands and head (flow.probe's
            motion_features) of ground truth, retrieved and random,
            speaker-normalised against the speaker's own held-out
            distribution, and their within-speaker correlation with the
            truth's.
  demo      held-out segment: four VRM panels with the recorded audio --
            idle only, random library windows, retrieved, ground truth --
            rendered in headless Chromium (flow.vrmrender); a free
            sentence: the director plans the turn once (markers, TTS) and
            writes the markers-only states beside the markers + search
            ones, rendered side by side.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

try:
    from . import actionhead as ah
    from . import beat2 as b2
    from . import body as fb
    from . import probe as pr
    from . import speech as sp
    from . import vrmbody as vb
    from .behaviors import min_jerk
    from .train import hidden as fh
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import actionhead as ah
    from flow import beat2 as b2
    from flow import body as fb
    from flow import probe as pr
    from flow import speech as sp
    from flow import vrmbody as vb
    from flow.behaviors import min_jerk
    from flow.train import hidden as fh

FORMAT = "tsukuyomi.gesturelib/1"
JOINTS = list(vb.JOINTS)
ARM_KEYS = [k for k in JOINTS if k.startswith(("upper_arm.", "forearm.",
                                               "hand.", "shoulder"))]
HEAD_KEYS = [k for k in JOINTS if k.startswith(("head.", "torso."))]
TYPE = "gsearch"
PARTS = ("hands", "full")


def log(s):
    print(s, flush=True)


@dataclass
class Params:
    """Everything a search is tuned by (the director's --gs-* flags)."""
    amp: float = 1.0            # arm amplitude about the rest pose
    parts: str = "hands"        # hands | full (adds head and torso)
    hop: float = 2.0            # s between query windows
    fade: float = 0.4           # s a window takes to take over from the last
    ramp: float = 0.4           # s a track eases in and out over
    chain_gap: float = 1.0      # s between sentences that still chain
    pad: float = 0.3            # s either side of a sentence (the export's)
    tail: float = 0.2           # s a track runs past the last word
    topk: int = 6
    pool: int = 48
    temperature: float = 0.05   # 0: always the best
    cont_w: float = 0.25        # continuity cost against cosine
    joint_w: float = 0.5        # ... share of the arm joints (rest: palms)
    memory: int = 8             # recent picks the no-repeat rule remembers
    clip_pen: float = 0.03      # per recent pick from the same clip
    nudge_pen: float = 0.04     # hu: where the contact fix starts
    skip_pen: float = 0.08      # hu: a window still this deep is skipped
    speakers: tuple = ()        # restrict to these speakers (names)
    langs: tuple = ()           # ... and these languages
    seed: int = 0


# ------------------------------------------------------------- library

class Library:
    """The searchable windows (module docstring, library). Arrays: emb [N,
    dim], win_seg / win_start [N] (frames), start_pose / end_pose [N, 23],
    joints [frames, 23] with seg_off [S + 1], seg_clip / seg_spk /
    seg_lang [S]; meta: names, seg_ids, rest, rms_ref, window, fps."""

    def __init__(self, arrays, meta):
        self.meta = meta
        self.emb = np.asarray(arrays["emb"], np.float32)
        self.win_seg = np.asarray(arrays["win_seg"], np.int64)
        self.win_start = np.asarray(arrays["win_start"], np.int64)
        self.start_pose = np.asarray(arrays["start_pose"], np.float32)
        self.end_pose = np.asarray(arrays["end_pose"], np.float32)
        self.joints = np.asarray(arrays["joints"], np.float16)
        self.seg_off = np.asarray(arrays["seg_off"], np.int64)
        self.seg_clip = np.asarray(arrays["seg_clip"], np.int64)
        self.seg_spk = np.asarray(arrays["seg_spk"], np.int64)
        self.seg_lang = np.asarray(arrays["seg_lang"], np.int64)
        self.W = int(meta["window"])
        self.fps = float(meta["fps"])
        self.rest = np.asarray(meta["rest"], np.float32)
        self.rms_ref = meta.get("rms_ref")
        self.win_clip = self.seg_clip[self.win_seg]
        self.win_spk = self.seg_spk[self.win_seg]
        self.win_lang = self.seg_lang[self.win_seg]

    def __len__(self):
        return len(self.emb)

    def joints_of(self, w):
        """[W, 23] float32 joints of window w."""
        a = self.seg_off[self.win_seg[w]] + self.win_start[w]
        return self.joints[a:a + self.W].astype(np.float32)

    def pose_at(self, w, off):
        """[len(w), 23] pose of windows w at frame `off` into them."""
        w = np.atleast_1d(w)
        a = self.seg_off[self.win_seg[w]] + self.win_start[w] + int(off)
        return self.joints[a].astype(np.float32)

    def describe(self, w):
        """{segment id, start (s into the segment), clip, speaker}."""
        m = self.meta
        s = int(self.win_seg[w])
        return {"segment": m["seg_ids"][s],
                "start": round(float(self.win_start[w]) / self.fps, 3),
                "clip": m["clips"][int(self.seg_clip[s])],
                "speaker": m["speakers"][int(self.seg_spk[s])]}

    def mask(self, speakers=(), langs=()):
        """[N] bool: windows of these speakers / languages (names, case
        ignored; empty: all)."""
        ok = np.ones(len(self), bool)
        for what, want, table, per in (
                ("speaker", speakers, self.meta["speakers"], self.win_spk),
                ("language", langs, self.meta["langs"], self.win_lang)):
            if not want:
                continue
            low = {n.lower(): i for i, n in enumerate(table)}
            ids = [low[w.lower()] for w in want if w.lower() in low]
            if not ids:
                raise SystemExit(f"gesture search: no {what} {list(want)} in "
                                 f"the library (it has {table})")
            ok &= np.isin(per, ids)
        return ok

    def save(self, out):
        os.makedirs(out, exist_ok=True)
        np.savez(os.path.join(out, "library.npz"),
                 emb=self.emb.astype(np.float16), win_seg=self.win_seg,
                 win_start=self.win_start,
                 start_pose=self.start_pose.astype(np.float16),
                 end_pose=self.end_pose.astype(np.float16),
                 joints=self.joints, seg_off=self.seg_off,
                 seg_clip=self.seg_clip, seg_spk=self.seg_spk,
                 seg_lang=self.seg_lang)
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1)

    @classmethod
    def load(cls, path):
        try:
            with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            z = np.load(os.path.join(path, "library.npz"))
        except OSError as exc:
            raise SystemExit(f"{path}: not a gesture library ({exc}); build "
                             "one: python -m flow.gesture_search build")
        if meta.get("format") != FORMAT:
            raise SystemExit(f"{path}: format {meta.get('format')}, "
                             f"expected {FORMAT}")
        if meta["joints"] != JOINTS:
            raise SystemExit(f"{path}: built for other joints than "
                             "flow.vrmbody.JOINTS; rebuild it")
        return cls({k: z[k] for k in z.files}, meta)


def text_form(rows, n=5000):
    """{"lower", "punct"} of the export's spoken texts: the share of cased
    letters that are lowercase, the share of non-space characters that are
    punctuation -- what a sentence is made to look like before the text
    model reads it (normalise_text)."""
    step = max(len(rows) // n, 1)
    low = cased = punct = chars = 0
    for r in rows[::step]:
        for ch in unicodedata.normalize("NFKC", r.get("spoken") or ""):
            if ch.isspace():
                continue
            chars += 1
            punct += unicodedata.category(ch).startswith("P")
            if ch.lower() != ch.upper():
                cased += 1
                low += ch.islower()
    return {"lower": round(low / cased, 4) if cased else 0.0,
            "punct": round(punct / chars, 4) if chars else 1.0}


def build(export, encoders, out, hop=15, variant="text+prosody", device=None,
          limit=0, allow_outside=False, log=log):
    """The library of the export's windows (module docstring, library) in
    `out`. Returns the Library."""
    b2.check_out(out, allow_outside)
    enc = pr.load_encoders(encoders, variant, device)
    cfg = enc.cfg
    held = set(cfg.get("eval_clips") or ())
    if not held:
        log("gesture search: the encoders have no held-out clips; the "
            "library will hold everything")
    rows, npz = fh.read_export(export)
    z = np.load(npz)
    files = set(z.files)
    W = enc.W
    if int(cfg["fps"]) != 30:
        log(f"gesture search: the encoders were trained at {cfg['fps']} fps")
    skipped = Counter()
    seg_ids, seg_keys = [], []
    clips, spks, langs = [], [], []
    seg_clip, seg_spk, seg_lang, seg_off = [], [], [], [0]
    jparts, emb, wseg, wstart, sp_, ep_ = [], [], [], [], [], []
    queue = []
    rms = []

    def index(table, name):
        if name not in table:
            table.append(name)
        return table.index(name)

    def flush():
        if queue:
            emb.append(enc.embed_motion(np.stack(queue)))
            queue.clear()
    for r in rows:
        if limit and len(seg_ids) >= limit:
            break
        clip = pr.clip_of(r)
        key = r["key"]
        if clip in held:
            skipped["held-out clip"] += 1
            continue
        if key + ".traj" not in files or key + ".joints" not in files:
            skipped["no .traj / .joints (export with flow.beat2)"] += 1
            continue
        cm = pr.columns(r["vars"], enc.cols)
        if len(cm) != len(enc.cols):
            skipped["missing canonical columns"] += 1
            continue
        y = z[key + ".traj"].astype(np.float32)[:, [cm[c] for c in enc.cols]]
        J = z[key + ".joints"].astype(np.float32)
        if len(J) != len(y) or len(y) < W:
            skipped["shorter than a window"] += 1
            continue
        s = len(seg_ids)
        seg_ids.append(r["id"])
        seg_keys.append(key)
        seg_clip.append(index(clips, clip))
        seg_spk.append(index(spks, pr.speaker_name(r)))
        seg_lang.append(index(langs, pr.lang_of(r)))
        jparts.append(J.astype(np.float16))
        seg_off.append(seg_off[-1] + len(J))
        if key + ".prosody" in files:
            p = z[key + ".prosody"].astype(np.float32)
            v = p[:, 2] > 0.5
            if v.sum() > 10:
                rms.append(float(np.median(p[v, 0])))
        for a in range(0, len(y) - W + 1, int(hop)):
            if not (np.isfinite(y[a:a + W]).all()
                    and np.isfinite(J[a:a + W]).all()):
                continue
            queue.append(y[a:a + W])
            wseg.append(s)
            wstart.append(a)
            sp_.append(J[a])
            ep_.append(J[a + W - 1])
            if len(queue) >= 2048:
                flush()
        if len(seg_ids) % 200 == 0:
            log(f"gesture search: {len(seg_ids)} segments, "
                f"{len(wseg)} windows")
    flush()
    if not wseg:
        raise SystemExit("gesture search: no window in the library (skipped: "
                         f"{dict(skipped)})")
    joints = np.concatenate(jparts)
    sample = joints[::max(len(joints) // 200000, 1)].astype(np.float32)
    meta = {"format": FORMAT, "joints": JOINTS, "window": W, "fps": enc.fps,
            "hop": int(hop), "dim": int(cfg["dim"]),
            "speakers": spks, "clips": clips, "langs": langs,
            "seg_ids": seg_ids, "seg_keys": seg_keys,
            "rest": [float(x) for x in np.median(sample, 0)],
            "rms_ref": float(np.median(rms)) if rms else None,
            "encoders": os.path.abspath(encoders), "variant": variant,
            "export": os.path.abspath(export),
            "held_out_clips": sorted(held), "skipped": dict(skipped),
            "text_form": text_form(rows),
            "windows": len(wseg), "segments": len(seg_ids),
            "licence": "derived from BEAT2: research only, never commit"}
    lib = Library({"emb": np.concatenate(emb), "win_seg": np.array(wseg),
                   "win_start": np.array(wstart),
                   "start_pose": np.stack(sp_), "end_pose": np.stack(ep_),
                   "joints": joints, "seg_off": np.array(seg_off),
                   "seg_clip": np.array(seg_clip),
                   "seg_spk": np.array(seg_spk),
                   "seg_lang": np.array(seg_lang)}, meta)
    lib.save(out)
    log(f"gesture search: library of {len(wseg)} windows from "
        f"{len(seg_ids)} segments ({len(clips)} clips, {len(spks)} speakers)"
        f", {len(held)} held-out clips left out -> {out}"
        + (f"; skipped {dict(skipped)}" if skipped else ""))
    return lib


# ------------------------------------------------------- text states

class TextStates:
    """Token states of a sentence as flow.train.hidden extracts them for
    the probe: `--context none` (the text alone, no template), the layers
    of the hidden folder's meta.json, their fixed projection (proj.npy),
    stored as fp16, then the probe's layer selection. The model is loaded
    on first use and kept for the session."""

    def __init__(self, model, hidden, sel, load_4bit=False, device="auto",
                 log=log):
        self.model, self.sel, self.log = model, list(sel), log
        self.load_4bit, self.device = load_4bit, device
        with open(os.path.join(hidden, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("context") not in (None, "none"):
            log(f"gesture search: the hidden states were extracted with "
                f"--context {self.meta.get('context')}; sentences are "
                "queried with the text alone, which is a different "
                "distribution")
        self.layers = list(self.meta.get("layers") or [])
        self.proj = None
        if self.meta.get("proj"):
            self.proj = np.load(os.path.join(hidden, "proj.npy"))
        self.lm = self.tok = None
        self.cache = {}

    def _load(self):
        try:
            from .train import hf as fhf
        except ImportError:
            from flow.train import hf as fhf
        self.lm, self.tok = fhf.load(self.model, None, self.load_4bit,
                                     self.device, self.log)

    def __call__(self, text):
        """(states [n, len(sel), D] float16, spans [n, 2] int32) of text."""
        if text in self.cache:
            return self.cache[text]
        import torch
        if self.lm is None:
            self._load()
        ids, raw = fh.token_offsets(self.tok, text)
        dev = next(self.lm.parameters()).device
        t0 = time.monotonic()
        with torch.no_grad():
            out = self.lm(input_ids=torch.tensor([ids], device=dev),
                          output_hidden_states=True, use_cache=False)
            h = torch.stack([out.hidden_states[k][0] for k in self.layers],
                            1).float()
            if self.proj is not None:
                h = h @ torch.from_numpy(self.proj).to(h.device)
        h = h.cpu().numpy().astype(np.float16)[:, self.sel]
        self.log(f"gesture search: text states of {len(ids)} tokens in "
                 f"{time.monotonic() - t0:.2f}s")
        self.cache[text] = (h, raw.astype(np.int32))
        return self.cache[text]


# ---------------------------------------------------------- queries

@dataclass
class Query:
    """One stretch of speech to find motion for: a sentence (or a held-out
    BEAT2 segment). t0: its first frame's stream time (a multiple of 1 /
    fps); n frames; token states h [m, L, D] with times tt (s from t0);
    raw prosody [n, 3] (rms, f0, voiced); speech: (first, last) word times
    from t0."""
    t0: float
    n: int
    h: np.ndarray
    tt: np.ndarray
    pros: np.ndarray
    speech: tuple
    text: str = ""
    lang: str = ""


def reply_prosody(audio, sr, fps, rms_ref=None):
    """[frames, 3] prosody of a reply's audio (flow.beat2.prosody at the
    frame rate), its loudness brought to the library's: the voiced frames'
    median RMS becomes rms_ref."""
    n = int(math.ceil(len(audio) / sr * fps)) + 2
    p = b2.prosody(np.asarray(audio, np.float64), sr, fps, n)
    if rms_ref:
        v = p[:, 2] > 0.5
        if v.any():
            med = float(np.median(p[v, 0]))
            if med > 1e-9:
                p[:, 0] *= float(np.clip(rms_ref / med, 0.05, 20.0))
    return p


def normalise_text(text, form):
    """(text', keep): a TTS sentence as the encoder's text looked --
    `form` is the library's text_form: lowercased when the export's texts
    are (the TextGrid's words), punctuation dropped when they have next to
    none (not an apostrophe or hyphen inside a word), NFKC, single spaces.
    keep[i] is the index in `text` of text'[i], so the characters' times
    stay on their characters. No form: the text as it is."""
    if not form:
        return text, np.arange(len(text))
    lower = form.get("lower", 0.0) > 0.95
    bare = form.get("punct", 1.0) < 0.01
    out, keep = [], []
    n = len(text)
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if bare and unicodedata.category(c).startswith("P") and not (
                    c in "'\u2019-" and 0 < i < n - 1
                    and text[i - 1].isalnum() and text[i + 1].isalnum()):
                continue
            if c.isspace():
                if not out or out[-1] == " ":
                    continue
                c = " "
            for x in (c.lower() if lower else c):
                out.append(x)
                keep.append(i)
    while out and out[-1] == " ":
        out.pop()
        keep.pop()
    return "".join(out), np.array(keep, int)


def sentence_queries(spoken, spans, audio, sr, text_states, fps=30.0,
                     pad=0.3, rms_ref=None, t_offset=0.0, form=None):
    """[Query] one per sentence of a reply: `spans` the characters' (start,
    end) seconds in the audio (speech.speak), text_states(sentence) ->
    (states, spans); the sentence is made to look like the library's texts
    first (normalise_text)."""
    pros = reply_prosody(audio, sr, fps, rms_ref)
    out = []
    for a, b in sp.sentences(spoken):
        text = spoken[a:b]
        if not any(ch.isalnum() for ch in text):
            continue
        sp_ = np.asarray(spans[a:b], np.float64)
        first, last = float(sp_[:, 0].min()), float(sp_[:, 1].max())
        f0 = max(int(math.floor((first - pad) * fps)), 0)
        f1 = int(math.ceil((last + pad) * fps))
        t0 = f0 / fps
        chars = (sp_ - t0).astype(np.float32)
        norm, keep = normalise_text(text, form)
        if not any(ch.isalnum() for ch in norm):
            continue
        h, tok = text_states(norm)
        tt = ah.token_times(tok, chars[keep], last - t0)
        p = np.zeros((f1 - f0, 3), np.float32)
        seg = pros[f0:f1]
        p[:len(seg)] = seg
        out.append(Query(t0 + t_offset, f1 - f0, h, tt, p,
                         (first - t0, last - t0), text,
                         sp.language(text)))
    return out


def window_starts(n, W, hop, max_gap=None):
    """Query window starts (frames) over n frames: about every `hop`,
    spread evenly so that the last one ends with the stretch (a remainder
    under half a hop adds no window: the last is held for it),
    never further apart than max_gap (the cross-fade needs both windows
    around it); a short stretch: one window."""
    if n <= W:
        return [0]
    k = int(round((n - W) / hop)) + 1
    if k > 1 or n - W > (max_gap or W):
        k = max(k, int(math.ceil((n - W) / (max_gap or W))) + 1)
    if k == 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n - W, k)})


def embed_query(enc, q, starts):
    """[len(starts), dim] conditioning embeddings of q's windows, as the
    probe embeds a window (flow.probe.make_batch)."""
    c = enc.cfg
    W, fps = enc.W, enc.fps
    toks, tms, pros = [], [], []
    for a in starts:
        h, tm = pr.window_tokens(q.h, q.tt, a / fps - c["lookback"],
                                 (a + W) / fps, a / fps, c["max_tokens"])
        toks.append(h)
        tms.append(tm)
        p = np.zeros((W, 3), np.float32)
        seg = q.pros[a:a + W]
        p[:len(seg)] = seg
        pros.append(ah.pros_feat(p))
    return enc.embed_cond(toks, tms, np.stack(pros))


# ----------------------------------------------------------- the body

def smooth(u):
    return min_jerk(np.clip(u, 0.0, 1.0))


class Retarget:
    """Library joints on a target body: the driven parameters (arms
    absolute, scaled about the body's rest and clamped; head and torso, in
    --gs-parts full, as the deviation from the library's median pose),
    the body's collision check and contact fix."""

    def __init__(self, bm, rest, parts="hands", amp=1.0, nudge_pen=0.04,
                 fps=30.0):
        if parts not in PARTS:
            raise SystemExit(f"--gs-parts {parts}: one of {PARTS}")
        self.bm, self.amp, self.fps = bm, float(amp), float(fps)
        self.parts = parts
        self.keys = ARM_KEYS + (HEAD_KEYS if parts == "full" else [])
        self.kinds = ["abs"] * len(ARM_KEYS) + ["add"] * (
            len(self.keys) - len(ARM_KEYS))
        self.idx = np.array([JOINTS.index(k) for k in self.keys])
        self.nA = len(ARM_KEYS)
        p = bm.params
        miss = [k for k in JOINTS if k not in p]
        if miss:
            raise SystemExit(f"the body lacks joint parameters {miss}")
        self.lo = np.array([p[k]["min"] for k in self.keys], float)
        self.hi = np.array([p[k]["max"] for k in self.keys], float)
        self.dflt = np.array([p[k]["default"] for k in self.keys], float)
        self.span = np.maximum(self.hi - self.lo, 1e-6)
        self.rest = np.asarray(rest, float)[self.idx]
        self.base = bm.default_values()
        self.cfg = dict(b2.CFG, contact_pen=float(nudge_pen))

    def apply(self, J):
        """[..., K] driven values of library joints J [..., 23]."""
        g = np.asarray(J, float)[..., self.idx]
        v = np.empty_like(g)
        a = self.nA
        v[..., :a] = np.clip(self.dflt[:a] + self.amp * (g[..., :a]
                                                         - self.dflt[:a]),
                             self.lo[:a], self.hi[:a])
        v[..., a:] = self.amp * (g[..., a:] - self.rest[a:])
        return v

    def rest_values(self):
        v = np.zeros(len(self.keys))
        v[:self.nA] = self.dflt[:self.nA]
        return v

    def full(self, V):
        """{joint: [N]} of all 23 for driven values V [N, K] (the body's
        rest elsewhere; deltas added to the rest)."""
        V = np.atleast_2d(V)
        val = {k: np.full(len(V), float(self.base[k])) for k in JOINTS}
        for c, (k, kind) in enumerate(zip(self.keys, self.kinds)):
            val[k] = (V[:, c] if kind == "abs"
                      else float(self.base[k]) + V[:, c]).astype(float)
        return val

    def palms(self, v):
        """[2, 3] palm centres (hu) of one frame's driven values."""
        val = {k: float(x[0]) for k, x in self.full(v).items()}
        pal = self.bm.palms({**self.base, **val})
        return np.stack([pal["L"], pal["R"]])

    def _measure(self, V):
        n = len(V)
        hips = {"hips.x": np.zeros(n), "hips.y": np.zeros(n)}
        val = self.full(V)
        return val, hips, b2.measure(self.bm, val, hips, [], self.fps)

    def pen(self, V):
        """Deepest arm-in-body / arm-in-arm overlap (hu) over the frames."""
        if not getattr(self.bm, "coll", None):
            return 0.0
        return float(self._measure(V)[2]["pen"].max())

    def nudge(self, V):
        """(V', info): the arms solved again where they go deeper than the
        contact threshold (beat2.fix_contacts); V' = V when none does."""
        info = {"pen": 0.0, "fixed": 0}
        if not getattr(self.bm, "coll", None):
            return V, info
        val, hips, meas = self._measure(V)
        info["pen"] = float(meas["pen"].max())
        if info["pen"] <= self.cfg["contact_pen"]:
            return V, info
        fx = b2.fix_contacts(self.bm, val, hips, [], self.fps, meas,
                             self.cfg)
        out = np.array(V, float)
        for c, (k, kind) in enumerate(zip(self.keys, self.kinds)):
            if kind == "abs":
                out[:, c] = val[k]
        info.update(fixed=int(fx["frames"]),
                    pen_after=float(self._measure(out)[2]["pen"].max()))
        return out, info

    def limits(self):
        return {k: (float(a), float(b)) for k, a, b in zip(self.keys,
                                                          self.lo, self.hi)}


# ------------------------------------------------------------ choose

class Chooser:
    """Which library window for a query embedding (module docstring,
    choose). numpy only: `lib` and the embedding, a Retarget for the
    palms, no model."""

    def __init__(self, lib, rt, params, rng):
        self.lib, self.rt, self.p, self.rng = lib, rt, params, rng
        self.recent = deque(maxlen=max(int(params.memory), 0))
        self.ok = lib.mask(params.speakers, params.langs)

    def forget(self):
        self.recent.clear()

    def remember(self, w):
        self.recent.append(int(w))

    def banned(self):
        """[N] bool: windows overlapping a recent pick of the same
        recording."""
        lib = self.lib
        bad = np.zeros(len(lib), bool)
        for w in self.recent:
            bad |= (lib.win_seg == lib.win_seg[w]) & (
                np.abs(lib.win_start - lib.win_start[w]) < lib.W)
        return bad

    def repeat_cost(self, idx):
        lib = self.lib
        n = np.zeros(len(idx))
        for w in self.recent:
            n += lib.win_clip[idx] == lib.win_clip[w]
        return self.p.clip_pen * n

    def continuity(self, idx, off, cur, cur_palms):
        """[len(idx)] cost of entering windows idx (frame `off` into them)
        from the pose cur [K] (palms cur_palms [2, 3])."""
        rt = self.rt
        V = rt.apply(self.lib.pose_at(idx, off))
        dj = np.abs(V[:, :rt.nA] - cur[:rt.nA]) / rt.span[:rt.nA]
        jc = np.sqrt((dj ** 2).mean(1))
        pc = np.array([np.linalg.norm(rt.palms(v) - cur_palms, axis=1).mean()
                       for v in V])
        return self.p.joint_w * jc + (1 - self.p.joint_w) * pc

    def rank(self, q, off=0, cur=None, cur_palms=None, random=False):
        """The candidates for a query embedding q [dim], best first:
        (idx, score, sim, cost) arrays of at most `topk`. random: ignore q
        and the costs (the baseline of the demo and the eval)."""
        lib, p = self.lib, self.p
        ok = self.ok & ~self.banned()
        if not ok.any():
            ok = self.ok
        cand = np.flatnonzero(ok)
        if random:
            k = min(max(p.topk, 1), len(cand))
            idx = self.rng.choice(cand, k, replace=False)
            z = np.zeros(k)
            return idx, z, z, z
        sims = lib.emb @ np.asarray(q, np.float32)
        sims = sims[cand]
        n = min(p.pool, len(cand))
        top = np.argpartition(-sims, n - 1)[:n] if n < len(cand) \
            else np.arange(len(cand))
        idx, sim = cand[top], sims[top]
        cost = np.zeros(len(idx))
        if cur is not None and p.cont_w > 0:
            cost = self.continuity(idx, off, cur, cur_palms)
        score = sim - p.cont_w * cost - self.repeat_cost(idx)
        o = np.argsort(-score)[:max(p.topk, 1)]
        return idx[o], score[o], sim[o], cost[o]

    def draw(self, score):
        """Index into a ranked list: the best at temperature 0, else a
        softmax draw."""
        if self.p.temperature <= 0 or len(score) == 1:
            return 0
        z = (score - score.max()) / self.p.temperature
        pr_ = np.exp(z)
        return int(self.rng.choice(len(score), p=pr_ / pr_.sum()))


# -------------------------------------------------------------- blend

def blend_items(items, W, fade, f0, f1):
    """[f1 - f0, K] track over frames f0..f1 of the windows `items` [(first
    frame, values [W, K])] in order: window i plays from its first frame
    (its end and start held where a fade reaches past them); the next one
    takes over around the middle of what the two have in common, over
    `fade` frames, minimum jerk, each fade shifting what came before
    toward the new window."""
    f = np.arange(f0, f1)
    out = None
    prev = None
    for i, (fi, V) in enumerate(items):
        x = V[np.clip(f - fi, 0, W - 1)]
        if out is None:
            out = x.astype(float)
        else:
            b = 0.5 * (fi + prev + W)
            s = smooth((f - (b - fade / 2.0)) / max(fade, 1e-9))[:, None]
            out = out + s * (x - out)
        prev = fi
    return out if out is not None else np.zeros((len(f), 0))


def token_of(rt, V, f0, fps, p, windows, hold=0):
    """The "gsearch" token of a track V [F, K] starting at frame f0."""
    lim = rt.limits()
    return {"t": round(f0 / fps, 4), "type": TYPE, "role": TYPE, "src":
            "director", "fps": float(fps), "keys": list(rt.keys),
            "kinds": list(rt.kinds),
            "lo": [lim[k][0] for k in rt.keys],
            "hi": [lim[k][1] for k in rt.keys],
            "values": np.round(V, 3).tolist(),
            "duration": round(len(V) / fps, 4), "ramp_in": p.ramp,
            "ramp_out": p.ramp, "groups": [], "windows": windows}


# ----------------------------------------------------------- planner

class GestureSearch:
    """Library + encoders + (for free sentences) the text-state model on a
    target body: plan() turns queries into "gsearch" tokens."""

    def __init__(self, lib, enc, bm, text=None, params=None, fps=30.0,
                 log=log):
        self.lib, self.enc, self.bm, self.text = lib, enc, bm, text
        self.p = params or Params()
        self.fps, self.log = float(fps), log
        if abs(self.fps - lib.fps) > 1e-6:
            raise SystemExit(f"gesture search: the library is {lib.fps:g} "
                             f"fps, the stream {self.fps:g}")
        self.rt = Retarget(bm, lib.rest, self.p.parts, self.p.amp,
                           self.p.nudge_pen, self.fps)
        self.rng = np.random.default_rng([int(self.p.seed), 5])
        self.chooser = Chooser(lib, self.rt, self.p, self.rng)
        self.cache = {}
        self.stats = Counter()

    @classmethod
    def from_args(cls, a, bm, log=log):
        """From flow.director's --gesture-search flags (gs_args)."""
        lib = Library.load(a.gesture_search)
        enc = pr.load_encoders(a.gs_encoders, a.gs_variant,
                               None if a.gs_device == "auto" else a.gs_device)
        cfg = enc.cfg
        meta = cfg["hidden"]["meta"]
        hid = a.gs_hidden or cfg["hidden"]["dir"]
        model = a.gs_model or meta.get("model")
        if not model:
            raise SystemExit("--gs-model: the language model the hidden "
                             "states came from")
        if not os.path.exists(os.path.join(hid, "meta.json")):
            raise SystemExit(f"--gs-hidden {hid}: no meta.json (the folder "
                             "of flow.train.hidden the probe was trained on)")
        text = TextStates(model, hid, cfg["sel"], a.gs_4bit, a.gs_device,
                          log)
        p = Params(amp=a.gs_amp, parts=a.gs_parts, seed=a.gs_seed,
                   temperature=a.gs_temperature, topk=a.gs_topk,
                   speakers=tuple(x for x in (a.gs_speaker or "").split(",")
                                  if x))
        return cls(lib, enc, bm, text, p, a.fps, log)

    def layer(self, defaults=None):
        return GestureLayer(self.fps, defaults, self.rt.limits())

    # ---- one window

    def prepare(self, w):
        """(values [W, K], info) of library window w on the body, or None
        when it penetrates past the skip threshold after the contact
        fix."""
        if w in self.cache:
            return self.cache[w]
        V = self.rt.apply(self.lib.joints_of(w))
        V, info = self.rt.nudge(V)
        deep = info.get("pen_after", info["pen"])
        res = (V, info) if deep <= self.p.skip_pen else None
        self.stats["fixed" if info["fixed"] else "clean"] += 1
        if res is None:
            self.stats["skipped"] += 1
        self.cache[w] = res
        return res

    def choose(self, q, f_new, prev, random=False):
        """(window, values, record) for a query embedding q entering at
        frame f_new after `prev` ((first frame, values) of the last window
        of this track) or from the idle (None); None when every candidate
        is out."""
        W, rt, p = self.lib.W, self.rt, self.p
        fade = p.fade * self.fps
        if prev is None:
            off = int(round(p.ramp * self.fps / 2))
            cur = rt.rest_values()
        else:
            fp, Vp = prev
            fs = 0.5 * (f_new + fp + W) - fade / 2.0
            off = int(np.clip(round(fs - f_new), 0, W - 1))
            cur = Vp[int(np.clip(round(fs - fp), 0, W - 1))]
        palms = None if random else rt.palms(cur)
        idx, score, sim, cost = self.chooser.rank(
            q, off, cur, palms, random=random)
        order = list(range(len(idx)))
        while order:
            j = self.chooser.draw(score[order]) if len(order) > 1 else 0
            c = order.pop(j)
            got = self.prepare(int(idx[c]))
            if got is None:
                continue
            self.chooser.remember(idx[c])
            rec = dict(self.lib.describe(int(idx[c])), sim=round(
                float(sim[c]), 4), cost=round(float(cost[c]), 4),
                nudged=int(got[1]["fixed"]), frame=int(f_new))
            return int(idx[c]), got[0], rec
        return None

    # ---- a reply

    def plan(self, queries, random=False):
        """([tokens], report) for queries (sentence_queries): one token
        per chain of sentences."""
        p, W, fps = self.p, self.lib.W, self.fps
        hop = max(int(round(min(p.hop, W / fps - p.fade - 0.1) * fps)), 1)
        fade = p.fade * fps
        tokens, report = [], []
        track = None            # {"items", "recs", "f0", "end"}

        def close():
            nonlocal track
            if track is None:
                return
            f1 = max(int(math.ceil(track["end"] * fps)),
                     track["items"][-1][0] + 1)
            V = blend_items(track["items"], W, fade, track["f0"], f1)
            V, info = self.rt.nudge(V)
            tok = token_of(self.rt, V, track["f0"], fps, p, track["recs"])
            tokens.append(tok)
            report.append({"t": tok["t"], "seconds": tok["duration"],
                           "windows": track["recs"],
                           "track_pen": round(info["pen"], 4),
                           "track_fixed": info["fixed"]})
            track = None
        for q in queries:
            f_q = int(round(q.t0 * fps))
            t_first = q.t0 + q.speech[0]
            t_last = q.t0 + q.speech[1]
            if track is not None and t_first - track["speech_end"] > \
                    p.chain_gap:
                close()
            starts = window_starts(q.n, W, hop,
                                   int(W - math.ceil(fade)) - 1)
            emb = None if random else embed_query(self.enc, q, starts)
            for k, a in enumerate(starts):
                f_new = f_q + a
                prev = track["items"][-1] if track else None
                got = self.choose(None if random else emb[k], f_new, prev,
                                  random)
                if got is None:
                    self.stats["no window"] += 1
                    close()
                    continue
                w, V, rec = got
                if track is None:
                    track = {"items": [], "recs": [], "f0": f_new,
                             "end": 0.0, "speech_end": 0.0}
                track["items"].append((f_new, V))
                track["recs"].append(rec)
            if track is not None:
                track["end"] = t_last + p.tail
                track["speech_end"] = t_last
        close()
        return tokens, report

    def plan_reply(self, spoken, said, random=False, t_offset=0.0):
        """plan() for a reply spoken by flow.speech.speak (`said`)."""
        if self.text is None:
            raise SystemExit("gesture search: no text-state model "
                             "(--gs-model)")
        qs = sentence_queries(spoken, said["spans"], said["audio"],
                              said["sr"], self.text, self.fps, self.p.pad,
                              self.lib.rms_ref, t_offset,
                              self.lib.meta.get("text_form"))
        return self.plan(qs, random)


# -------------------------------------------------------------- layer

class GestureLayer:
    """"gsearch" tokens played over the frame the idle built: MoveLayer's
    sibling (flow.moves). commit(tok) queues a token; apply(f, over,
    parts) is frame f. Per driven parameter the track's value is blended
    in with the token's envelope (eased in and out) as an absolute
    target (arms: over + e * (value - over)) or an offset (head and torso:
    over + e * value); a later token takes over a shared parameter as it
    eases in (e_i = w_i * prod over later j of (1 - w_j)), so a chain of
    sentences never jumps. Values are clamped to the body's limits.

    note(tokens) is told of the pose and move tokens of the stream when
    they are directed: a parameter such a token plays (a pose's `params`,
    a move's absolute ones) is ducked -- the gesture's weight on it falls
    to 0 over `ramp` before the marker starts and rises again after its
    hold -- so the marker takes the arm from a gesture that has already
    gone, and `holding` (the parameters the gesture plays now) keeps the
    pose layer from showing it through a held pose. `written` /
    `holding` as MoveLayer's."""

    def __init__(self, fps, defaults=None, limits=None, ramp=0.4):
        self.fps = float(fps)
        self.defaults = dict(defaults or {})
        self.limits = dict(limits or {})
        self.ramp = float(ramp)
        self.queue = []
        self.active = []
        self.ducks = []              # (param, start, end, down, up)
        self.written = set()
        self.holding = set()

    def commit(self, tok):
        self.queue.append(tok)

    def busy(self):
        return bool(self.queue or self.active)

    def duck(self, params, start, end, down=None, up=None):
        """Duck `params` over [start, end] s (ramps `down` before, `up`
        after)."""
        down = self.ramp if down is None else down
        up = self.ramp if up is None else up
        for p in params:
            self.ducks.append((p, float(start), float(end), float(down),
                               float(up)))

    def note(self, tokens):
        """Duck for the directed pose and move tokens."""
        for k in tokens:
            typ = k.get("type")
            if typ == "pose":
                a = float(k["t"])
                b = a + float(k.get("duration", 0.0)) + float(
                    k.get("hold", 0.0))
                self.duck(k.get("params", {}), a, b, self.ramp,
                          max(float(k.get("release", self.ramp)), 1 / self.fps))
            elif typ == "move" and k.get("abs"):
                a = float(k["t"]) + float(k.get("ramp_in", 0.3))
                b = float(k["t"]) + float(k["duration"]) - float(
                    k.get("ramp_out", 0.3))
                self.duck(k["abs"], a, max(b, a), self.ramp,
                          float(k.get("ramp_out", 0.3)))

    def gate(self, p, tf):
        """The gesture's share of parameter p at time tf (1: free)."""
        g = 1.0
        for q, a, b, down, up in self.ducks:
            if q != p:
                continue
            if tf <= a - down:
                continue
            if tf < a:
                g *= 1.0 - float(smooth((tf - (a - down)) / down))
            elif tf <= b:
                g = 0.0
            elif tf < b + up:
                g *= float(smooth((tf - b) / up))
        return g

    def _start(self, tok):
        V = np.asarray(tok["values"], float)
        keys = list(tok["keys"])
        lim = {k: (a, b) for k, a, b in zip(keys, tok["lo"], tok["hi"])}
        self.limits.update(lim)
        self.active.append({
            "tok": tok, "f0": int(round(float(tok["t"]) * self.fps)),
            "V": V, "keys": keys, "kinds": list(tok["kinds"]),
            "dur": (len(V) - 1) / self.fps,
            "rin": max(float(tok.get("ramp_in", self.ramp)), 1 / self.fps),
            "rout": max(float(tok.get("ramp_out", self.ramp)),
                        1 / self.fps)})

    def apply(self, f, over, parts):
        self.written, self.holding = set(), set()
        if not self.busy():
            return over, parts
        while self.queue and int(round(float(self.queue[0]["t"])
                                       * self.fps)) <= f:
            self._start(self.queue.pop(0))
        over = dict(over)
        tf = f / self.fps
        live = []
        for a in self.active:
            j = f - a["f0"]
            if j >= len(a["V"]):
                continue
            w = float(env(j / self.fps, a["dur"], a["rin"], a["rout"]))
            live.append((a, w, a["V"][max(j, 0)]))
        self.active = [a for a in self.active if f - a["f0"] < len(a["V"])]
        self.ducks = [d for d in self.ducks if tf < d[2] + d[4] + 1.0]
        rem = {}
        for a, w, row in reversed(live):
            for c, k in enumerate(a["keys"]):
                r = rem.setdefault(k, [1.0, []])
                r[1].append((a["kinds"][c], float(row[c]), w * r[0]))
                r[0] *= 1.0 - w
        for k, (_r, items) in rem.items():
            u = float(over.get(k, self.defaults.get(k, 0.0)))
            g = self.gate(k, tf)
            v = u
            for kind, x, e in items:
                e *= g
                v += e * (x - u) if kind == "abs" else e * x
                if kind == "abs" and e > 0:
                    self.holding.add(k)
            lo, hi = self.limits.get(k, (-math.inf, math.inf))
            over[k] = min(max(v, lo), hi) if v != u else v
            self.written.add(k)
        return over, parts


def env(tau, dur, rin, rout):
    """The envelope of a token (flow.moves.envelope): 0 outside, eased in
    over rin and out over rout."""
    if tau <= 0 or tau >= dur:
        return 0.0
    w = 1.0
    if tau < rin:
        w = float(smooth(tau / rin))
    if tau > dur - rout:
        w = min(w, float(smooth((dur - tau) / rout)))
    return w


# --------------------------------------------------------------- eval

STATS = ("hand_std", "hand_v", "head_std", "head_v", "hand_act")


def _canon(lib, z, w, cols, cache):
    """[W, C] canonical columns of library window w, from the export it
    was built from (the library stores joints, the statistics read the
    palms and head angles)."""
    s = int(lib.win_seg[w])
    if s not in cache:
        key = lib.meta["seg_keys"][s]
        cache.clear()
        cache[s] = z[key + ".traj"].astype(np.float32)
    a = int(lib.win_start[w])
    return cache[s][a:a + lib.W]


def embed_windows(enc, corpus, win, o, bs=256):
    """(motion, conditioning) unit embeddings [n, dim] of corpus windows
    with their own text (the probe's evaluation)."""
    import torch
    n = len(win["seg"])
    src = np.arange(n)
    norm = (enc.norm[0], enc.norm[1])
    zm, zc = [], []
    enc.model.eval()
    with torch.no_grad():
        for a in range(0, n, bs):
            b = pr.make_batch(corpus, win, np.arange(a, min(a + bs, n)), src,
                              norm, o, torch, enc.device)
            for z_, f in ((zm, enc.model.em), (zc, enc.model.ec)):
                z_.append(torch.nn.functional.normalize(f(b), dim=-1)
                          .cpu().numpy())
    return np.concatenate(zm), np.concatenate(zc)


def evaluate(export, encoders, lib_dir, hidden=None, out=None, hop=30, k=32,
             n_rand=8, limit=0, seed=0, boot=1000, variant="text+prosody",
             device=None, params=None, log=log):
    """Retrieval and motion statistics on the held-out windows (module
    docstring, eval). Returns the report; writes report.json / .txt."""
    enc = pr.load_encoders(encoders, variant, device)
    cfg = enc.cfg
    lib = Library.load(lib_dir)
    held = set(cfg.get("eval_clips") or ())
    if not held:
        raise SystemExit("gesture search eval: the encoders recorded no "
                         "held-out clips (probe match --save)")
    hid = hidden or cfg["hidden"]["dir"]
    rows, npz = pr.read_rows(export, limit, seed, hid)
    rows = [r for r in rows if pr.clip_of(r) in held]
    if not rows:
        raise SystemExit("gesture search eval: none of the held-out clips "
                         f"has hidden states in {hid}")
    o = argparse.Namespace(
        window=cfg["window"], layer=cfg["layer_arg"], hop=hop,
        center=cfg["center"], lookback=cfg["lookback"],
        max_tokens=cfg["max_tokens"], variants_list=[variant],
        active_speed=0.3)
    corpus = pr.Corpus(rows, npz, hid, o, log=lambda s: None)
    if list(corpus.cols) != list(cfg["cols"]) or list(corpus.sel) != list(
            cfg["sel"]):
        raise SystemExit("gesture search eval: the corpus columns / layers "
                         "differ from the encoders'")
    win = corpus.windows(set(corpus.clips) & held)
    n = len(win["seg"])
    if not n:
        raise SystemExit("gesture search eval: no held-out window")
    log(f"gesture search eval: {n} held-out windows from "
        f"{len(set(win['clip'].tolist()))} clips, library of {len(lib)}")
    zm, zc = embed_windows(enc, corpus, win, o)
    rng = np.random.default_rng(seed)
    p = params or Params(seed=seed, memory=0, cont_w=0.0)
    p = Params(**{**p.__dict__, "memory": 0, "cont_w": 0.0})
    chooser = Chooser(lib, None, p, rng)
    clip_id = win["clip"]
    spk = win["spk"]
    best = np.zeros(n, int)
    drawn = np.zeros(n, int)
    for i in range(n):
        idx, score, _s, _c = chooser.rank(zc[i])      # no cost: by cosine
        best[i] = int(idx[0])
        drawn[i] = int(idx[chooser.draw(score)])
    rand = rng.integers(0, len(lib), (n, n_rand))
    rep = {"queries": n, "library": len(lib),
           "clips": int(len(set(clip_id.tolist()))), "chance": 0.5}

    def beats(sel):
        c = np.einsum("nd,nd->n", zm, lib.emb[sel])
        r = np.einsum("nd,nkd->nk", zm, lib.emb[rand])
        return ((c[:, None] > r) + 0.5 * (c[:, None] == r)).mean(1), c, r
    for name, sel in (("best", best), ("sampled", drawn)):
        win_, c, r = beats(sel)
        m, lo, hi = pr.acc_ci(win_, clip_id, boot, seed)
        rep[f"retrieved_closer_{name}"] = {
            "share": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "mean_cos_retrieved": round(float(c.mean()), 4),
            "mean_cos_random": round(float(r.mean()), 4)}
    # the probe's own question, on the library: the true motion's rank
    # by cosine to the query among K - 1 random library windows
    neg = rng.integers(0, len(lib), (n, k - 1))
    tc = np.einsum("nd,nd->n", zc, zm)
    nc = np.einsum("nd,nkd->nk", zc, lib.emb[neg])
    rk = (nc >= tc[:, None]).sum(1)
    for key, hit in (("top1", rk == 0), ("top5", rk < 5)):
        m, lo, hi = pr.acc_ci(hit, clip_id, boot, seed)
        rep[f"truth_{key}_of_{k}"] = {"acc": round(m, 4), "lo": round(lo, 4),
                                      "hi": round(hi, 4),
                                      "chance": round(min(
                                          (1 if key == "top1" else 5) / k,
                                          1.0), 4)}
    # motion statistics
    z = np.load(npz)
    cmap = {c: i for i, c in enumerate(cfg["cols"])}
    fps = enc.fps
    feats = {"truth": [], "retrieved": [], "random": []}
    cache = {}
    for i in range(n):
        s = corpus.segs[win["seg"][i]]
        a = int(win["start"][i])
        ys = {"truth": s["y"][a:a + corpus.W],
              "retrieved": _canon(lib, z, best[i], cfg["cols"], cache),
              "random": _canon(lib, z, int(rand[i, 0]), cfg["cols"], cache)}
        for nme, y in ys.items():
            f, _h, _k = pr.motion_features(y, cmap, fps, o)
            feats[nme].append([f[t] for t in STATS])
    F = {k_: np.array(v, float) for k_, v in feats.items()}
    stats = {}
    for j, t in enumerate(STATS):
        g = F["truth"][:, j]
        z_ = {}
        for nme in F:
            x = F[nme][:, j]
            ok = np.isfinite(x)
            row = {"mean": (round(float(np.nanmean(x)), 4)
                            if ok.any() else None)}
            # speaker-normalised: against the speaker's own held-out
            # truth (mean 0, sd 1 for the truth itself)
            zs = np.full(n, np.nan)
            for sp_i in np.unique(spk):
                m_ = (spk == sp_i) & np.isfinite(g)
                if m_.sum() < 2:
                    continue
                gm, gs_ = g[m_].mean(), g[m_].std()
                if gs_ > 1e-9:
                    zs[spk == sp_i] = (x[spk == sp_i] - gm) / gs_
            row["speaker_z"] = (round(float(np.nanmean(zs)), 4)
                                if np.isfinite(zs).any() else None)
            if nme != "truth" and ok.sum() > 10:
                c = pr.corr(g[ok], x[ok], spk[ok], clip_id[ok], "within",
                            B=boot, seed=seed)
                row["r_with_truth"] = {kk: (None if c.get(kk) is None
                                            else round(float(c[kk]), 3))
                                       for kk in ("r", "r_lo", "r_hi")
                                       if kk in c}
            z_[nme] = row
        stats[t] = z_
    rep["motion"] = stats
    rep["text"] = eval_lines(rep, k)
    out = out or os.path.join(lib_dir, "eval")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump({k_: v for k_, v in rep.items() if k_ != "text"}, f,
                  indent=1)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep["text"]) + "\n")
    log("\n".join(rep["text"]))
    return rep


def eval_lines(rep, k):
    L = [f"gesture search eval: {rep['queries']} held-out windows "
         f"({rep['clips']} clips) against a library of {rep['library']}",
         "", "retrieved window vs a random library window, by closeness to "
         "the true motion's embedding (share of queries where it is closer; "
         "chance 0.5; [95% clip bootstrap]):"]
    for nme in ("best", "sampled"):
        d = rep[f"retrieved_closer_{nme}"]
        L.append(f"  {nme:8s} {d['share']:.3f} [{d['lo']:.3f}, "
                 f"{d['hi']:.3f}]   mean cos {d['mean_cos_retrieved']:.3f} "
                 f"vs {d['mean_cos_random']:.3f}")
    L += ["", f"the true motion among K - 1 = {k - 1} random library windows,"
          " by cosine to the query:"]
    for key in ("top1", "top5"):
        d = rep[f"truth_{key}_of_{k}"]
        L.append(f"  {key}: {d['acc']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] "
                 f"(chance {d['chance']:.3f})")
    L += ["", "motion statistics per 3 s window (mean; speaker z: against "
          "the speaker's own truth; r: within-speaker correlation with the "
          "truth's, [CI]):"]
    L.append("  " + "".ljust(10) + "".join(f"{n:^38}" for n in (
        "truth", "retrieved", "random")))
    for t, d in rep["motion"].items():
        cells = []
        for nme in ("truth", "retrieved", "random"):
            e = d[nme]
            c = ("-" if e["mean"] is None else f"{e['mean']:.3f}") + (
                "  z -" if e["speaker_z"] is None
                else f"  z {e['speaker_z']:+.2f}")
            r = e.get("r_with_truth")
            if r and r.get("r") is not None:
                c += f"  r {r['r']:+.2f}"
                if r.get("r_lo") is not None:
                    c += f" [{r['r_lo']:+.2f},{r['r_hi']:+.2f}]"
            cells.append(c.center(38))
        L.append("  " + t.ljust(10) + "".join(cells))
    return L


# --------------------------------------------------------------- demo

def segment_query(row, z, hidden_index, hidden_dir, sel):
    """The Query of a stored BEAT2 segment: its recorded prosody, its
    stored text states and word times (what eval reads)."""
    key = row["key"]
    e = hidden_index[row["id"]]
    sh = np.load(os.path.join(hidden_dir, e["shard"]))
    h = sh[e["key"] + ".h"]
    h = (h if h.ndim == 3 else h[:, None])[:, sel]
    chars = z[key + ".chars"].astype(np.float32)
    tt = ah.token_times(sh[e["key"] + ".spoken"], chars,
                        row.get("speech_seconds") or 0.0)
    pros = z[key + ".prosody"].astype(np.float32)
    return Query(0.0, len(pros), h, tt, pros,
                 (float(chars[0, 0]), float(chars[-1, 1])), row["spoken"],
                 pr.lang_of(row))


def mouth_of(pros):
    """[n] mouth opening from a prosody's RMS (flow.beat2.states_of)."""
    ref = max(float(np.percentile(pros[:, 0], 95)), 1e-4)
    return np.clip(pros[:, 0] / ref, 0, 1) * 0.8


def truth_states(av, z, row, fingers=True):
    """[state] of the segment's recorded joints on `av`
    (beat2.segment_states); the stored fingers only for the avatar the
    export was made for (their bones are that avatar's)."""
    if fingers:
        return b2.segment_states(av, z, row)[0]
    key = row["key"]
    joints = z[key + ".joints"].astype(np.float32)
    traj = z[key + ".traj"].astype(np.float32)
    hips = np.zeros((len(joints), 2), np.float32)
    for c, v in enumerate(row["vars"]):
        if v in ("hips.x", "hips.y"):
            hips[:, ("hips.x", "hips.y").index(v)] = np.nan_to_num(traj[:, c])
    return b2.states_of(av, joints, hips, z[key + ".prosody"].astype(
        np.float32))


def play_states(bm, rig, tokens, n, mouth, seed=0, gs=None):
    """[state] of n frames of the VRM idle with these gesture tokens (and
    no markers), the mouth following `mouth` [n]."""
    try:
        from . import stream as fs
    except ImportError:
        from flow import stream as fs
    st = fs.MotionStream(rig, seed=seed, idle=True, log=lambda s: None)
    if tokens:
        st.gsearch = gs.layer(dict(zip(rig.pids, rig.dflt.tolist())))
        st.direct(tokens)
    out = []
    for i in range(n):
        over, _p = st.frame()
        over = dict(over)
        over["mouth.open"] = float(mouth[min(i, len(mouth) - 1)])
        out.append(bm.state(over))
    return out


def stack_videos(vids, dest, audio=None, log=log):
    """The videos side by side (ffmpeg hstack) with `audio` muxed."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed for the video")
    side = dest[:-4] + ".silent.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for v in vids:
        cmd += ["-i", v]
    cmd += ["-filter_complex", f"hstack=inputs={len(vids)}", "-c:v",
            "libx264", "-pix_fmt", "yuv420p", side]
    subprocess.run(cmd, check=True)
    if audio and os.path.exists(audio):
        try:
            from .director import mux_cmd
        except ImportError:
            from flow.director import mux_cmd
        c = mux_cmd(side, audio, "", dest)
        subprocess.run(c, check=True, capture_output=True)
        os.remove(side)
    else:
        os.replace(side, dest)
    log(f"gesture search: {dest}")
    return dest


def demo_segment(a, log=log):
    """Four panels of a stored (held-out) segment (module docstring,
    demo): idle only | random windows | retrieved | ground truth."""
    try:
        from . import vrm as fv
        from . import vrmrender as vr
    except ImportError:
        from flow import vrm as fv
        from flow import vrmrender as vr
    enc = pr.load_encoders(a.encoders, a.variant, a.device)
    cfg = enc.cfg
    lib = Library.load(a.lib)
    rows, npz = fh.read_export(a.export)
    row = next((r for r in rows if r["id"] == a.id), None)
    if row is None:
        raise SystemExit(f"gesture search: {a.id} not in {a.export}.jsonl "
                         f"({len(rows)} rows, e.g. {[r['id'] for r in rows[:3]]})")
    if pr.clip_of(row) not in set(cfg.get("eval_clips") or ()):
        log(f"gesture search: {a.id} is not a held-out clip of the "
            "encoders: the library holds its own motion; pick a held-out "
            "one (the probe report lists them)")
    meta_path = os.path.join(os.path.dirname(a.export), "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    own = not a.vrm
    if own:
        av = b2.Avatar(meta.get("vrm"), a.body or meta.get("body"))
    else:
        # Another avatar: its own body file (built beside the others in
        # rigs/ when there is none), as the director resolves it.
        _bm, bpath = fv.body_for(a.vrm, a.body, log=lambda s: None)
        av = b2.Avatar(a.vrm, bpath)
    bm = av.bm
    hid = a.hidden or cfg["hidden"]["dir"]
    z = np.load(npz)
    q = segment_query(row, z, fh.load_index(hid), hid, cfg["sel"])
    p = Params(amp=a.amp, parts=a.parts, seed=a.seed,
               temperature=a.temperature, topk=a.topk,
               speakers=tuple(x for x in (a.speaker or "").split(",") if x))
    gs = GestureSearch(lib, enc, bm, None, p, row["fps"], log)
    fps = float(row["fps"])
    n = int(row["frames"])
    if a.seconds:
        n = min(n, int(round(a.seconds * fps)))
    rig = fv.VRMRig(bm, a.events, fps, "calm", log=lambda s: None)
    mouth = mouth_of(q.pros)
    tok_r, rep_r = gs.plan([q], random=True)
    gs.chooser.forget()
    gs.rng = np.random.default_rng([a.seed, 5])
    gs.chooser.rng = gs.rng
    tok_s, rep_s = gs.plan([q])
    panels = {
        "idle": play_states(bm, rig, [], n, mouth, a.seed),
        "random": play_states(bm, rig, tok_r, n, mouth, a.seed, gs),
        "retrieved": play_states(bm, rig, tok_s, n, mouth, a.seed, gs),
        "truth": truth_states(av, z, row, own)[:n]}
    d = os.path.join(a.out, b2.key_of(a.id))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"id": a.id, "spoken": row["spoken"], "panels": list(
            panels), "random": rep_r, "retrieved": rep_s,
            "truth": {"clip": row.get("clip"), "t0": row.get("t0")}}, f,
            indent=1, ensure_ascii=False)
    for name, st in panels.items():
        fv.save_states(os.path.join(d, name + ".states.json"), st, fps,
                       av.body_path, av.vrm_path)
    wav = None
    if a.beat2:
        info = next((c for c in b2.list_clips(a.beat2)
                     if c["stem"] == row["clip"]), None)
        if info:
            x, sr = b2.read_wav(info["wav"])
            wav = os.path.join(d, "speech.wav")
            s0 = int(row["t0"] * sr)
            b2.write_wav(wav, x[s0:s0 + int(n / fps * sr)], sr)
    if a.no_render:
        return {"dir": d, "panels": list(panels)}
    vids = []
    for name, st in panels.items():
        mp4 = os.path.join(d, name + ".mp4")
        vr.render(st, bm, mp4, fps, a.size, av.vrm_path, log=None)
        vids.append(mp4)
    dest = stack_videos(vids, os.path.join(d, "demo.mp4"), wav, log)
    return {"dir": d, "demo": dest, "panels": list(panels)}


def demo_say(a, rest, log=log):
    """A free sentence through the director: the turn planned once, the
    markers-only states beside the markers + search ones."""
    try:
        from . import director as dr
        from . import vrmrender as vr
        from . import body as fb_
    except ImportError:
        from flow import director as dr
        from flow import vrmrender as vr
        from flow import body as fb_
    d = os.path.join(a.out, "say")
    argv = list(rest) + ["--avatar", "vrm", "--say", a.say, "--out", d,
                         "--gesture-search", a.lib, "--gs-encoders",
                         a.encoders, "--gs-baseline", "--gs-amp",
                         str(a.amp), "--gs-seed", str(a.seed),
                         "--gs-parts", a.parts, "--gs-variant", a.variant]
    for flag, v in (("--gs-model", a.gs_model), ("--gs-hidden", a.hidden),
                    ("--gs-speaker", a.speaker), ("--vrm", a.vrm),
                    ("--vrm-body", a.body),
                    ("--gs-device", a.device)):
        if v:
            argv += [flag, v]
    rc = dr.main(argv)
    if rc:
        return rc
    base = os.path.join(d, "states_markers.json")
    if not os.path.exists(base):
        raise SystemExit("gesture search demo --say: the turn planned no "
                         "search (it needs --tts and a reply with speech)")
    vids = []
    for name in ("states_markers", "states"):
        with open(os.path.join(d, name + ".json"), encoding="utf-8") as f:
            rec = json.load(f)
        bm = fb_.load(rec["body"])
        mp4 = os.path.join(d, name + ".mp4")
        vr.render(rec["states"], bm, mp4, rec.get("fps", 30.0), a.size,
                  rec.get("vrm") if rec.get("vrm") and os.path.exists(
                      rec["vrm"]) else None, log=None)
        vids.append(mp4)
    stack_videos(vids, os.path.join(d, "demo.mp4"),
                 os.path.join(d, "speech.wav"), log)
    return 0


def held_out_ids(export, encoders, n=20, variant="text+prosody", log=log):
    """Print the ids of held-out segments (the ones `demo --id` should
    take): id, seconds, the first words."""
    d = encoders if os.path.exists(os.path.join(encoders, "config.json")) \
        else os.path.join(encoders, variant)
    with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
        held = set(json.load(f).get("eval_clips") or ())
    rows, _ = fh.read_export(export)
    out = [r for r in rows if pr.clip_of(r) in held]
    for r in out[:n]:
        log(f"{r['id']}  {r['frames'] / r['fps']:5.1f}s  "
            f"{r['spoken'][:60]}")
    log(f"gesture search: {len(out)} held-out segments of {len(rows)}")
    return [r["id"] for r in out]


# ---------------------------------------------------------------- cli

def parse(argv):
    # No abbreviations: what `demo --say` does not know goes to the
    # director, where --mode must not become --model.
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=lambda
                            **kw: argparse.ArgumentParser(
                                allow_abbrev=False, **kw))

    def common(s, lib=True, enc=True):
        s.add_argument("--export", required=True,
                       help="flow.beat2's <out>/action stem")
        if enc:
            s.add_argument("--encoders", required=True,
                           help="flow.probe match --save DIR")
            s.add_argument("--variant", default="text+prosody")
        if lib:
            s.add_argument("--lib", required=True)
        s.add_argument("--device", default=None)
        s.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("build", help="embed the export's windows")
    common(b, lib=False)
    b.add_argument("--out", required=True)
    b.add_argument("--hop", type=int, default=15, help="frames [15]")
    b.add_argument("--limit", type=int, default=0, help="segments")
    b.add_argument("--allow-outside-data", action="store_true")
    e = sub.add_parser("eval", help="retrieval and statistics on held-out "
                       "windows")
    common(e)
    e.add_argument("--hidden", default=None)
    e.add_argument("--out", default=None)
    e.add_argument("--hop", type=int, default=30)
    e.add_argument("--k", type=int, default=32)
    e.add_argument("--random", type=int, default=8,
                   help="random library windows per query [8]")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--boot", type=int, default=1000)
    i = sub.add_parser("ids", help="held-out segment ids for demo --id")
    i.add_argument("--export", required=True)
    i.add_argument("--encoders", required=True)
    i.add_argument("--variant", default="text+prosody")
    i.add_argument("--n", type=int, default=20)
    d = sub.add_parser("demo", help="side-by-side videos")
    common(d)
    d.add_argument("--id", default=None, help="a held-out segment id")
    d.add_argument("--say", default=None, help="a free sentence (the "
                   "director's flags go with it)")
    d.add_argument("--out", required=True)
    d.add_argument("--hidden", default=None)
    d.add_argument("--beat2", default=None, help="the BEAT2 folder (the "
                   "segment's audio)")
    d.add_argument("--vrm", default=None)
    d.add_argument("--body", default=None)
    d.add_argument("--gs-model", default=None,
                   help="--say: the local model of the hidden states")
    d.add_argument("--events", default="default")
    d.add_argument("--amp", type=float, default=1.0)
    d.add_argument("--parts", default="hands", choices=PARTS)
    d.add_argument("--speaker", default=None)
    d.add_argument("--temperature", type=float, default=0.05)
    d.add_argument("--topk", type=int, default=6)
    d.add_argument("--size", type=int, default=384)
    d.add_argument("--seconds", type=float, default=0.0,
                   help="only the first seconds of the segment [all]")
    d.add_argument("--no-render", action="store_true")
    a, rest = ap.parse_known_args(argv)
    if a.cmd != "demo" and rest:
        ap.error("unrecognized arguments: " + " ".join(rest))
    if a.cmd == "demo" and bool(a.id) == bool(a.say):
        ap.error("demo: exactly one of --id and --say")
    if a.cmd == "demo" and a.id and rest:
        ap.error("unrecognized arguments: " + " ".join(rest))
    return a, rest


def main(argv=None) -> int:
    a, rest = parse(sys.argv[1:] if argv is None else list(argv))
    if a.cmd == "build":
        build(a.export, a.encoders, a.out, a.hop, a.variant, a.device,
              a.limit, a.allow_outside_data)
    elif a.cmd == "ids":
        held_out_ids(a.export, a.encoders, a.n, a.variant)
    elif a.cmd == "eval":
        evaluate(a.export, a.encoders, a.lib, a.hidden, a.out, a.hop, a.k,
                 a.random, a.limit, a.seed, a.boot, a.variant, a.device)
    elif a.say:
        return demo_say(a, rest)
    else:
        demo_segment(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
