#!/usr/bin/env bash
# Render one demo clip per (prompt, avatar) with flow.director: the model's
# reply, its voice, the body and the subtitles in runs/<OUT>/<avatar>_<n>/demo.mp4.
# Clips already rendered are skipped, so a rerun picks up where it stopped.
#
#   MODEL=tsk-q8m OUT=demo_q8m_v2 bash scripts/demo.sh
#
#   MODEL     the Ollama model (required)
#   OUT       folder under runs/ [demo_$MODEL]
#   AVATARS   space-separated [Hiyori lafei]
#   PROMPTS   one user line per line [scripts/demo_prompts.txt]
#   FRAME     bust | full [bust]
#   EXTRA     more flow.director flags, e.g. "--no-micro" or "--body-prompt beats"
set -u
MODEL=${MODEL:?set MODEL}
OUT=runs/${OUT:-demo_$MODEL}
AVATARS=${AVATARS:-Hiyori lafei}
PROMPTS=${PROMPTS:-scripts/demo_prompts.txt}
FRAME=${FRAME:-bust}
EXTRA=${EXTRA:-}
declare -A MJ=(
  [Hiyori]="CubismWebSamples/Samples/Resources/Hiyori/Hiyori.model3.json"
  [lafei]="live2d-models/live2d_3/model/Azue Lane(JP)/lafei/lafei.model3.json"
  [Natori]="CubismWebSamples/Samples/Resources/Natori/Natori.model3.json"
  [Mao]="CubismWebSamples/Samples/Resources/Mao/Mao.model3.json"
)
unset DISPLAY
mkdir -p "$OUT"
i=0
while IFS= read -r say || [[ -n "$say" ]]; do
  [[ -z "$say" ]] && continue
  i=$((i+1))
  for av in $AVATARS; do
    d="$OUT/${av}_$i"
    [[ -f "$d/demo.mp4" ]] && continue
    mj=${MJ[$av]:-}
    # Not at the usual place (or not listed): the first <avatar>.model3.json
    # under live2d-models/ or CubismWebSamples/.
    if [[ -z "$mj" || ! -f "$mj" ]]; then
      mj=$(find live2d-models CubismWebSamples -name "$av.model3.json" 2>/dev/null | head -1)
    fi
    [[ -z "$mj" ]] && { echo "${av}: no $av.model3.json under live2d-models/ or CubismWebSamples/"; continue; }
    xvfb-run -a -s '-screen 0 1024x1024x24' python -m flow.director \
      --model-json "$mj" --rig "rigs/$av.npz" --mode body --backend ollama \
      --model "$MODEL" --tts auto --say "$say" --render --frame "$FRAME" \
      $EXTRA --out "$d" < /dev/null > "$d.log" 2>&1
    echo "${av}_$i: $(tail -1 "$d.log" | cut -c1-100)"
  done
done < "$PROMPTS"
echo "push: git add -f $OUT/*/demo.mp4 $OUT/*/reply.txt $OUT/*.log"
