#!/usr/bin/env bash
# Stage 1 director checks across rigs and language models.
#
#   signs: the hand-written direction test (runs/demo/signs2.txt) on each rig
#          with --no-idle, plus a 2x head crop -- does each tag go the way it
#          says? On aidang_2 the idle sway hid the tags entirely.
#   llms:  the same user lines to each language model on one rig; a table of
#          how many tags each used and of which kinds. Both models tried first
#          used three tags a reply, never a nod, shake or gesture.
#
# Usage, from the repo root:
#   bash scripts/director_matrix.sh                 # both steps
#   STEPS=llms RENDER=0 bash scripts/director_matrix.sh   # tags only, no video
#
# Environment knobs (defaults in brackets):
#   RIGS     rigs for the signs step [aidang_2 Hiyori Natori lafei xuefeng]
#   LLM_RIG  rig for the llms step [aidang_2]
#   LLMS     backend:model pairs, space separated; model may contain '/'
#            [siliconflow:Qwen/Qwen3.6-35B-A3B siliconflow:zai-org/GLM-5.3-Flash
#             siliconflow:moonshotai/Kimi-K3 deepseek:deepseek-chat
#             ollama:gemma4:e4b ollama:gemma4:e2b]
#            Pairs whose key is not set (SILICONFLOW_API_KEY, DEEPSEEK_API_KEY,
#            ANTHROPIC_API_KEY) are skipped; a failing pair does not stop
#            the rest.
#   PERSONA  persona line ["Atago, a gentle, teasing heavy cruiser from Azur Lane."]
#   MODELS   folders holding the .model3.json files [live2d-models:CubismWebSamples]
#   EVENTS   event stats [ev.json]
#   RENDER   1 = render video too [1]
# The table's "wall s" is per reply, model call plus planning (plus the render
# when RENDER=1, so compare models with RENDER=0 for latency).
#   CROP     ffmpeg crop for the head close-up, w:h:x:y at --size 512
#            [130:110:225:20, aidang_2's face]; other rigs need their own
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

RIGS=${RIGS:-"aidang_2 Hiyori Natori lafei xuefeng"}
LLM_RIG=${LLM_RIG:-aidang_2}
LLMS=${LLMS:-"siliconflow:Qwen/Qwen3.6-35B-A3B siliconflow:zai-org/GLM-5.3-Flash siliconflow:moonshotai/Kimi-K3 deepseek:deepseek-chat ollama:gemma4:e4b ollama:gemma4:e2b"}
PERSONA=${PERSONA:-"Atago, a gentle, teasing heavy cruiser from Azur Lane."}
MODELS=${MODELS:-live2d-models:CubismWebSamples}
EVENTS=${EVENTS:-ev.json}
RENDER=${RENDER:-1}
CROP=${CROP:-130:110:225:20}
STEPS=${STEPS:-"signs llms"}
OUT=runs/demo/matrix

has() { [[ " $STEPS " == *" $1 "* ]]; }

IFS=: read -ra model_roots <<< "$MODELS"
roots=()
for r in "${model_roots[@]}"; do [[ -e "$r" ]] && roots+=("$r"); done
model_of() {
    find -L "${roots[@]}" -name "$1.model3.json" -not -name '*.probe.*' \
        -print -quit
}

# The user lines: a greeting, agreement, refusal, surprise, a shy moment,
# and Japanese -- each should call for different tags.
SAYS=(
    "Hi Atago, did you miss me?"
    "Want to go to the beach with me tomorrow?"
    "I ate the cake you were saving. Sorry."
    "I got promoted to fleet admiral today!"
    "You look really beautiful today."
    "こんにちは、愛宕。今日は何をしたい？"
)

run() {  # rig, out dir, extra director args...
    local R=$1 dir=$2; shift 2
    local M; M=$(model_of "$R")
    if [[ -z "$M" ]]; then echo "!! $R: no model file, skipping" >&2; return 1; fi
    local render=(); [[ $RENDER == 1 ]] && render=(--render)
    local cmd=(python -m flow.director --model-json "$M" --rig "rigs/$R.npz"
               --clips "clips2/$R.npz" --events "$EVENTS" --seed 0
               --persona "$PERSONA" --out "$dir" "${render[@]}" "$@")
    mkdir -p "$dir"
    if [[ $RENDER == 1 ]]; then
        xvfb-run -a -s "-screen 0 1024x1024x24" "${cmd[@]}" > "$dir.log" 2>&1
    else
        "${cmd[@]}" > "$dir.log" 2>&1
    fi
}

if has signs; then
    for R in $RIGS; do
        d="$OUT/signs/$R"
        if run "$R" "$d" --backend file --reply-file runs/demo/signs2.txt \
                --no-idle; then
            [[ $RENDER == 1 && -f "$d/demo.mp4" ]] && ffmpeg -v error -y \
                -i "$d/demo.mp4" -vf "crop=$CROP,scale=520:440" "$d/head.mp4"
            echo "-- signs $R: ok"
        else
            echo "!! signs $R: failed (see $d.log)" >&2
        fi
    done
fi

if has llms; then
    for pair in $LLMS; do
        backend=${pair%%:*}; model=${pair#*:}
        case $backend in
            siliconflow) key=SILICONFLOW_API_KEY ;;
            deepseek)    key=DEEPSEEK_API_KEY ;;
            anthropic)   key=ANTHROPIC_API_KEY ;;
            *)           key= ;;
        esac
        if [[ -n "$key" && -z "${!key:-}" ]]; then
            echo "-- $pair: $key not set, skipping"; continue
        fi
        tag=$(echo "$pair" | tr '/:' '__')
        for i in "${!SAYS[@]}"; do
            d="$OUT/llms/$tag/$i"
            t0=$(date +%s.%N)
            if run "$LLM_RIG" "$d" --backend "$backend" --model "$model" \
                    --say "${SAYS[$i]}"; then
                python -c "import sys; print(float(sys.argv[2]) - float(sys.argv[1]))" \
                    "$t0" "$(date +%s.%N)" > "$d/wall_seconds.txt"
            else
                echo "!! $pair #$i: failed (see $d.log)" >&2
            fi
        done
        echo "-- $pair: done"
    done
    python - "$OUT/llms" <<'EOF'
import collections, json, os, sys
root = sys.argv[1]
if not os.path.isdir(root):
    sys.exit()
print(f"\n{'model':<38} {'ok':>3} {'tags':>5} {'/reply':>6} {'kinds':>5} "
      f"{'warn':>4} {'wall s':>6}  tag counts")
for tag in sorted(os.listdir(root)):
    d = os.path.join(root, tag)
    if not os.path.isdir(d):
        continue
    ok = n = warn = 0; wall = []; kinds = collections.Counter()
    for i in sorted(os.listdir(d)):
        f = os.path.join(d, i, "actions.json")
        if not os.path.exists(f):
            continue
        a = json.load(open(f))
        ok += 1
        acts = a.get("actions", [])
        n += len(acts); warn += len(a.get("warnings", []))
        kinds.update(x["tag"] + (":" + x["arg"] if x.get("arg") and
                                 x["tag"] != "pause" else "") for x in acts)
        w = os.path.join(d, i, "wall_seconds.txt")
        if os.path.exists(w):
            wall.append(float(open(w).read() or 0))
    per = n / ok if ok else 0
    ws = sum(wall) / len(wall) if wall else 0
    top = ", ".join(f"{k} {v}" for k, v in kinds.most_common())
    print(f"{tag:<38} {ok:>3} {n:>5} {per:>6.1f} "
          f"{len({k.split(':')[0] for k in kinds}):>5} {warn:>4} {ws:>6.1f}  {top}")
EOF
fi
echo "Push: git add -f $OUT && git commit -m 'director matrix' && git push"
