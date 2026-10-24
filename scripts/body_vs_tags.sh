#!/usr/bin/env bash
# E2 (docs/embodiment.md): does a language model direct its own body from a
# description (and a picture) better than it picks tags? The same user
# lines, the same seed and backend, through the director in tag mode and in
# body mode (flow.pose), rendered, scored and put on a blind A/B page.
#
# Per rig and condition, per line: <OUT>/<rig>/<cond>/<i>/ holds the
# director's run (reply.txt, actions.json with the reach report in body
# mode, tokens.json, plan.npz, demo.mp4) and score.json (flow.score on
# plan.npz against the rig's clips, when clips2/<rig>.npz exists).
#
# Conditions:
#   tags       BACKEND, the tag vocabulary (the baseline)
#   body       BACKEND, body targets from the text description
#   with VISION=backend:model, also on that model (a VLM):
#   vtags      the tags
#   vbody      body targets, text only
#   vbody_img  body targets plus a picture of itself (--body-image)
# CHECK=1 adds --body-check to the body conditions: one self-correction
# round per line, offline (check/check.json: reach before and after).
#
# Then per rig: `python -m flow.pose summary` (markers or tags per reply,
# reach, MISS, Core-checked reach, model latency, score checks) into
# summary.txt, and one blind A/B page over all rigs (flow.legacy.abtest):
# each line's clips pairwise, sources hidden.
#
# Usage, from the repo root (on the server: GL for the renders, keys for
# the backends):
#   bash scripts/body_vs_tags.sh
#   RIGS=Hiyori VISION=siliconflow:Qwen/Qwen3-VL-32B-Instruct bash scripts/body_vs_tags.sh
# The VLM name is a flag, not a promise: check what the service lists.
#
# Environment knobs (defaults in brackets):
#   RIGS     [Hiyori lafei]
#   BACKEND  [deepseek]; MODEL its model [the director's default]
#   VISION   backend:model for the image condition [none]
#   TTS      director --tts [auto: the rig's voice config]
#   CHECK    1 = --body-check on body conditions [0]
#   RENDER   1 = render (needs GL; xvfb-run is used) [1]
#   EVENTS   event stats [ev.json, else the built-in defaults]
#   MODELS   folders holding the .model3.json files [live2d-models:CubismWebSamples]
#   SEED     [0]
#   OUT      [runs/e2]
#   REPLIES  with BACKEND=file: <REPLIES>/<cond>/<i>.txt as the replies
#            (a dry run without a model)
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

RIGS=${RIGS:-"Hiyori lafei"}
BACKEND=${BACKEND:-deepseek}
MODEL=${MODEL:-}
VISION=${VISION:-}
TTS=${TTS:-auto}
CHECK=${CHECK:-0}
RENDER=${RENDER:-1}
EVENTS=${EVENTS:-ev.json}
[[ -f "$EVENTS" ]] || EVENTS=default
MODELS=${MODELS:-live2d-models:CubismWebSamples}
SEED=${SEED:-0}
OUT=${OUT:-runs/e2}
REPLIES=${REPLIES:-}

# The user lines: greeting, invitation, apology, surprise, compliment, and a
# request to show, point and wave -- the one a body should do better.
SAYS=(
    "Hi! Did you miss me?"
    "Want to go to the beach with me tomorrow?"
    "I ate the cake you were saving. Sorry."
    "I got promoted to fleet admiral today!"
    "You look really beautiful today."
    "Can you wave to my friend over there, and point to where the door is?"
)

persona_of() {
    case $1 in
        Hiyori) echo "Hiyori, a cheerful, curious high-school girl." ;;
        lafei)  echo "Laffey, a sleepy, gentle destroyer from Azur Lane." ;;
        *)      echo "${PERSONA:-a friendly, expressive character.}" ;;
    esac
}

IFS=: read -ra model_roots <<< "$MODELS"
roots=()
for r in "${model_roots[@]}"; do [[ -e "$r" ]] && roots+=("$r"); done
model_of() {
    find -L "${roots[@]}" -name "$1.model3.json" -not -name '*.probe.*' \
        -print -quit
}
key_of() {
    case $1 in
        siliconflow) echo SILICONFLOW_API_KEY ;;
        deepseek)    echo DEEPSEEK_API_KEY ;;
        anthropic)   echo ANTHROPIC_API_KEY ;;
        *)           echo "" ;;
    esac
}

conds=(tags body)
[[ -n "$VISION" ]] && conds+=(vtags vbody vbody_img)

run() {  # rig, cond, i
    local R=$1 cond=$2 i=$3
    local M; M=$(model_of "$R")
    [[ -z "$M" ]] && { echo "!! $R: no model file" >&2; return 1; }
    local d="$OUT/$R/$cond/$i"
    local be=$BACKEND md=$MODEL
    if [[ $cond == v* ]]; then be=${VISION%%:*}; md=${VISION#*:}; fi
    local args=(--model-json "$M" --rig "rigs/$R.npz" --events "$EVENTS"
                --seed "$SEED" --persona "$(persona_of "$R")" --out "$d"
                --tts "$TTS")
    [[ -f "clips2/$R.npz" ]] && args+=(--clips "clips2/$R.npz")
    if [[ $be == file ]]; then
        args+=(--backend file --reply-file "$REPLIES/$cond/$i.txt")
    else
        args+=(--backend "$be" --say "${SAYS[$i]}")
        [[ -n "$md" ]] && args+=(--model "$md")
    fi
    case $cond in
        *body_img) args+=(--mode body --body-image) ;;
        *body)     args+=(--mode body) ;;
    esac
    [[ $CHECK == 1 && $cond == *body* ]] && args+=(--body-check)
    [[ $RENDER == 1 ]] && args+=(--render)
    mkdir -p "$d"
    local t0; t0=$(date +%s.%N)
    if [[ $RENDER == 1 ]]; then
        xvfb-run -a -s "-screen 0 1024x1024x24" \
            python -m flow.director "${args[@]}" > "$d.log" 2>&1
    else
        python -m flow.director "${args[@]}" > "$d.log" 2>&1
    fi
    local rc=$?
    python -c "import sys; print(float(sys.argv[2]) - float(sys.argv[1]))" \
        "$t0" "$(date +%s.%N)" > "$d/wall_seconds.txt"
    if [[ $rc != 0 ]]; then echo "!! $R $cond #$i failed (see $d.log)" >&2; return 1; fi
    if [[ -f "clips2/$R.npz" && -f "$d/plan.npz" ]]; then
        python -m flow.score --traj "$d/plan.npz" --clips "clips2/$R.npz" \
            --rig "rigs/$R.npz" --model "$M" --json "$d/score.json" \
            > "$d/score.txt" 2>&1      # exits 1 when a check fails
        [[ -f "$d/score.json" ]] || echo "!! score $d (see score.txt)" >&2
    fi
}

for R in $RIGS; do
    for cond in "${conds[@]}"; do
        be=$BACKEND; [[ $cond == v* ]] && be=${VISION%%:*}
        key=$(key_of "$be")
        if [[ -n "$key" && -z "${!key:-}" ]]; then
            echo "-- $R $cond: $key not set, skipping"; continue
        fi
        for i in "${!SAYS[@]}"; do
            run "$R" "$cond" "$i"
        done
        echo "-- $R $cond: done"
    done
    python -m flow.pose summary "$OUT/$R" | tee "$OUT/$R/summary.txt"
done

# Blind A/B: per rig and line, every pair of conditions (abtest pairs the
# sources of one "rig" key). The clip with speech and subtitles when there
# is one.
items=()
for R in $RIGS; do
    for i in "${!SAYS[@]}"; do
        for cond in "${conds[@]}"; do
            d="$OUT/$R/$cond/$i"
            v="$d/demo.mp4"; [[ -f "$v" ]] || v="$d/render/preview.mp4"
            [[ -f "$v" ]] && items+=("$R-$i/$cond=$v")
        done
    done
done
if (( ${#items[@]} >= 2 )); then
    python -m flow.legacy.abtest make "$OUT/ab" "${items[@]}" --seed "$SEED"
fi

# Not the frames (render/*.png) and not the big wavs.
echo "Push: find $OUT \\( -name '*.json' -o -name '*.txt' -o -name '*.mp4' -o -name '*.log' -o -name '*.html' -o -name '*.png' \\) -not -path '*/render/*' -size -20M -print0 | xargs -0 git add -f"
