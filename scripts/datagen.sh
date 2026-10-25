#!/usr/bin/env bash
# Stage 2 data engine (flow.dataset, flow.dataexport): simulated users talk
# to the body-mode director on each avatar; every turn is recorded with its
# markers, solved targets, reach and canonical body trajectory, then
# filtered, summarised and exported for a LoRA fine-tune (sft.jsonl) and an
# action head (action.jsonl + action.npz).
#
# Safe to leave running for hours: sessions are written whole, so an
# interruption (Ctrl-C, a crash, a lost connection, the budget) loses at
# most the sessions in flight; run the same command again to resume (the
# spend so far counts toward the budget). Rate limits and 5xx are retried
# with backoff; a rejected key or an empty balance (401/402/403) stops the
# run; MAX_FAILS sessions failing in a row stop it too.
#
# Usage, from the repo root (on the server: the Live2D models under
# MODELS, DEEPSEEK_API_KEY in the environment):
#   bash scripts/datagen.sh                       # the pilot below
#   SESSIONS=2000 TURNS=8 BUDGET_USD=20 CONC=8 OUT=data/datagen/r1 \
#       nohup bash scripts/datagen.sh > datagen.log 2>&1 &
#   ESTIMATE=1 bash scripts/datagen.sh            # the cost, nothing called
#
# Environment knobs (defaults in brackets):
#   AVATARS     [Hiyori,seed-san,lafei] rigs/<name>.body.json each; Live2D
#               ones need their .model3.json under MODELS and rigs/<name>.npz
#   PERSONAS    [cheerful,sleepy,tsundere] ids in CONFIG
#   CONFIG      [flow/datagen.personas.json] personas, users, topics
#   SESSIONS    [20] sessions in all, round-robin over avatar x persona
#   TURNS       [6] turns per session
#   CONC        [4] sessions at once
#   BACKEND     [deepseek] the character; MODEL its model [director default]
#   USER_BACKEND [deepseek] the user simulator; USER_MODEL
#   TTS         [none] the rate model; auto/edge/voicevox: real word timings
#   BUDGET_USD  [] stop starting sessions at this spend (all runs into OUT)
#   BUDGET_TOKENS [] the same in tokens
#   PRICE_IN PRICE_IN_HIT PRICE_OUT  US$ per M tokens [flow.dataset.PRICES:
#               DeepSeek list prices as last checked; check them]
#   MODELS      [live2d-models:CubismWebSamples]
#   LITE        1 = Live2D avatars without their model (body file only;
#               flagged, out of the SFT export) [0]
#   MAX_FAILS   [5] consecutive failed sessions before stopping
#   SEED        [0]
#   OUT         [data/datagen/pilot]
#   ESTIMATE    1 = print the estimate and stop [0]
#   EXPORT      0 = skip stats and exports [1]
#   EXTRA       more flow.dataset flags, e.g. "--traj-vars all --no-gestures"
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

AVATARS=${AVATARS:-Hiyori,seed-san,lafei}
PERSONAS=${PERSONAS:-cheerful,sleepy,tsundere}
CONFIG=${CONFIG:-flow/datagen.personas.json}
SESSIONS=${SESSIONS:-20}
TURNS=${TURNS:-6}
CONC=${CONC:-4}
BACKEND=${BACKEND:-deepseek}
MODEL=${MODEL:-}
USER_BACKEND=${USER_BACKEND:-deepseek}
USER_MODEL=${USER_MODEL:-}
TTS=${TTS:-none}
BUDGET_USD=${BUDGET_USD:-}
BUDGET_TOKENS=${BUDGET_TOKENS:-}
MODELS=${MODELS:-live2d-models:CubismWebSamples}
LITE=${LITE:-0}
MAX_FAILS=${MAX_FAILS:-5}
SEED=${SEED:-0}
OUT=${OUT:-data/datagen/pilot}
ESTIMATE=${ESTIMATE:-0}
EXPORT=${EXPORT:-1}
EXTRA=${EXTRA:-}

args=(--avatars "$AVATARS" --personas "$PERSONAS" --config "$CONFIG"
      --sessions "$SESSIONS" --turns "$TURNS" --concurrency "$CONC"
      --backend "$BACKEND" --user-backend "$USER_BACKEND" --tts "$TTS"
      --models "$MODELS" --max-consecutive-failures "$MAX_FAILS"
      --seed "$SEED" --out "$OUT")
[[ -n "$MODEL" ]] && args+=(--model "$MODEL")
[[ -n "$USER_MODEL" ]] && args+=(--user-model "$USER_MODEL")
[[ -n "$BUDGET_USD" ]] && args+=(--budget-usd "$BUDGET_USD")
[[ -n "$BUDGET_TOKENS" ]] && args+=(--budget-tokens "$BUDGET_TOKENS")
[[ -n "${PRICE_IN:-}" ]] && args+=(--price-in "$PRICE_IN")
[[ -n "${PRICE_IN_HIT:-}" ]] && args+=(--price-in-hit "$PRICE_IN_HIT")
[[ -n "${PRICE_OUT:-}" ]] && args+=(--price-out "$PRICE_OUT")
[[ $LITE == 1 ]] && args+=(--live2d-lite)
# shellcheck disable=SC2206
[[ -n "$EXTRA" ]] && args+=($EXTRA)

mkdir -p "$OUT"
if [[ $ESTIMATE == 1 ]]; then
    exec python -m flow.dataset estimate "${args[@]}"
fi

# The estimate first (prepares every avatar, calls nothing), then the run.
python -m flow.dataset estimate "${args[@]}" || exit $?
echo "-- starting in 5 s (Ctrl-C to stop; the same command resumes)"
sleep 5
python -m flow.dataset gen "${args[@]}" 2>&1 | tee -a "$OUT/datagen.log"
rc=${PIPESTATUS[0]}

if [[ $EXPORT == 1 ]]; then
    python -m flow.dataexport stats "$OUT" --json "$OUT/stats.json" \
        | tee "$OUT/stats.txt"
    python -m flow.dataexport sft "$OUT" --out "$OUT/export/sft.jsonl"
    python -m flow.dataexport sft "$OUT" --out "$OUT/export/sft_turns.jsonl" \
        --per turn
    python -m flow.dataexport action "$OUT" --out "$OUT/export/action"
fi
exit "$rc"
