#!/usr/bin/env bash
# The tuned director against its teacher (flow.train.evalsft): the
# held-out sessions of a training run (RUN/split.json), user line by user
# line, through the student (as it will be served: Ollama by default),
# the teacher live (DeepSeek, the same prompts) and the teacher's recorded
# replies; each reply parsed and played on the avatar (poses solved, moves
# built), scored: parse-warning rate, refusals, markers per turn, unplayed
# markers, reach / MISS, first-token latency and tokens/s. Then the table,
# and with AB=1 a blind A/B page of rendered first turns (the
# scripts/body_vs_tags.sh layout, flow.legacy.abtest).
#
# Usage, from the repo root on the server (Ollama serving the student,
# DEEPSEEK_API_KEY for the live teacher):
#   RUN=runs/sft/q8 DATA=data/datagen/r1 STUDENT_MODEL=tsk-q8 \
#       bash scripts/eval_sft.sh
#   STUDENT_BACKEND=hf STUDENT_MODEL=Qwen/Qwen3-8B \
#       HF_ADAPTER=runs/sft/q8/adapter bash scripts/eval_sft.sh
#
# Environment knobs (defaults in brackets):
#   RUN              [runs/sft/q8] flow.train.sft's --out (split.json)
#   DATA             [the data folder in RUN/data.json] flow.dataset --out
#   OUT              [$RUN/eval]
#   STUDENT_BACKEND  [ollama] ollama | hf
#   STUDENT_MODEL    [tsk] the Ollama name (or the hf base / merged folder)
#   HF_ADAPTER       [] --backend hf: the adapter on STUDENT_MODEL
#   HF_4BIT          1 = the hf base in 4 bit [0]
#   TEACHER_BACKEND  [deepseek]; TEACHER_MODEL [its default]
#   TEACHER          1 = run the live teacher [1 when its key is set]
#   RECORDED         1 = score the recorded teacher replies (free) [1]
#   BASE_MODEL       an Ollama name for the untuned base too, e.g.
#                    qwen3:8b (sent num_ctx 8192) []
#   SESSIONS         [all] the first N held-out sessions; TURNS [all]
#   AB               1 = the A/B pack (renders: GL / Chromium) [0]
#   AB_N             [6] first turns per avatar; TTS [auto] for the renders
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

RUN=${RUN:-runs/sft/q8}
if [[ -z "${DATA:-}" && -f "$RUN/data.json" ]]; then
    DATA=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['data'])" "$RUN/data.json")
fi
DATA=${DATA:?set DATA to the flow.dataset folder}
OUT=${OUT:-$RUN/eval}
STUDENT_BACKEND=${STUDENT_BACKEND:-ollama}
STUDENT_MODEL=${STUDENT_MODEL:-tsk}
HF_ADAPTER=${HF_ADAPTER:-}
HF_4BIT=${HF_4BIT:-0}
TEACHER_BACKEND=${TEACHER_BACKEND:-deepseek}
TEACHER_MODEL=${TEACHER_MODEL:-}
RECORDED=${RECORDED:-1}
BASE_MODEL=${BASE_MODEL:-}
AB=${AB:-0}
AB_N=${AB_N:-6}
TTS=${TTS:-auto}
if [[ -z "${TEACHER:-}" ]]; then
    TEACHER=0
    [[ $TEACHER_BACKEND == deepseek && -n "${DEEPSEEK_API_KEY:-}" ]] && TEACHER=1
    [[ $TEACHER_BACKEND == siliconflow && -n "${SILICONFLOW_API_KEY:-}" ]] && TEACHER=1
fi

common=(--data "$DATA" --split "$RUN/split.json" --out "$OUT")
[[ -n "${SESSIONS:-}" ]] && common+=(--sessions "$SESSIONS")
[[ -n "${TURNS:-}" ]] && common+=(--turns "$TURNS")
mkdir -p "$OUT"

student=(--name student --backend "$STUDENT_BACKEND" --model "$STUDENT_MODEL")
[[ -n "$HF_ADAPTER" ]] && student+=(--hf-adapter "$HF_ADAPTER")
[[ $HF_4BIT == 1 ]] && student+=(--hf-4bit)
python -m flow.train.evalsft run "${common[@]}" "${student[@]}" \
    2>&1 | tee "$OUT/student.log"

if [[ $RECORDED == 1 ]]; then
    python -m flow.train.evalsft run "${common[@]}" --name recorded \
        --recorded 2>&1 | tee "$OUT/recorded.log"
fi
if [[ $TEACHER == 1 ]]; then
    teacher=(--name teacher --backend "$TEACHER_BACKEND")
    [[ -n "$TEACHER_MODEL" ]] && teacher+=(--model "$TEACHER_MODEL")
    python -m flow.train.evalsft run "${common[@]}" "${teacher[@]}" \
        2>&1 | tee "$OUT/teacher.log"
fi
if [[ -n "$BASE_MODEL" ]]; then
    python -m flow.train.evalsft run "${common[@]}" --name base \
        --backend ollama --model "$BASE_MODEL" --ollama-ctx 8192 \
        2>&1 | tee "$OUT/base.log"
fi

python -m flow.train.evalsft compare "$OUT" | tee "$OUT/compare.txt"

if [[ $AB == 1 ]]; then
    names=student
    [[ $TEACHER == 1 ]] && names="teacher,$names"
    [[ $TEACHER != 1 && $RECORDED == 1 ]] && names="recorded,$names"
    python -m flow.train.evalsft ab --data "$DATA" --out "$OUT" \
        --names "$names" --n "$AB_N" --tts "$TTS"
fi
