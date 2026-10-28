#!/usr/bin/env bash
# The action head's ablation grid on BEAT2 (docs/embodiment.md, Stage 2 (b)):
# five runs, each trained with a best-checkpoint, evaluated on the held-out
# turns with the same frames scored, a log per run and a table of the
# summary metrics side by side at the end.
#
#   a_text         text only (the head as it was)
#   b_hist         + 1 s of motion history
#   c_hist_pros    + prosody
#   d_no_text      history + prosody, no text (what the text adds: c - d)
#   e_fm           c with flow matching
#
# Usage, from the repo root (the numbers are in docs/embodiment.md):
#   bash scripts/ah_grid.sh
#   STEPS=5000 ONLY="b_hist c_hist_pros" EXTRA="--layer-mix" bash scripts/ah_grid.sh
# Variables: EXPORT, HIDDEN, OUT (runs go in $OUT/<name>/: train.log,
# eval.log, eval.json, head.best.pt), STEPS, EVAL_EVERY, PATIENCE (evaluations
# without improvement before a run stops, 0 = never), SKIP (seconds at the
# start of every clip that no run is scored on: the history's, the same for
# all runs), EXTRA (flags added to every train), ONLY (names to run), PY,
# FORCE=1 (redo runs that have an eval.json). A run that failed does not stop
# the others; the table prints what finished.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PY=${PY:-python}
EXPORT=${EXPORT:-data/beat2/seed-san/action}
HIDDEN=${HIDDEN:-$HOME/tsk-runs/hidden/beat2}
OUT=${OUT:-$HOME/tsk-runs/ah/grid}
STEPS=${STEPS:-4000}
EVAL_EVERY=${EVAL_EVERY:-100}
PATIENCE=${PATIENCE:-0}
SKIP=${SKIP:-1.0}
EXTRA=${EXTRA:-}
ONLY=${ONLY:-"a_text b_hist c_hist_pros d_no_text e_fm"}

flags() {
    case "$1" in
        a_text)      echo "" ;;
        b_hist)      echo "--history 1" ;;
        c_hist_pros) echo "--history 1 --prosody" ;;
        d_no_text)   echo "--history 1 --prosody --no-text" ;;
        e_fm)        echo "--history 1 --prosody --objective fm" ;;
        *) echo "unknown run $1" >&2; return 1 ;;
    esac
}

mkdir -p "$OUT"
done_runs=()
for name in $ONLY; do
    f=$(flags "$name") || continue
    d="$OUT/$name"
    mkdir -p "$d"
    if [ -f "$d/eval.json" ] && [ -z "${FORCE:-}" ]; then
        echo "== $name: done (FORCE=1 to redo)"
        done_runs+=("$d/eval.json")
        continue
    fi
    echo "== $name: train $f $EXTRA"
    # shellcheck disable=SC2086
    $PY -m flow.actionhead train --export "$EXPORT" --hidden "$HIDDEN" \
        --out "$d" --steps "$STEPS" --eval-every "$EVAL_EVERY" \
        --patience "$PATIENCE" $f $EXTRA > "$d/train.log" 2>&1 \
        || { echo "   train failed, see $d/train.log"; continue; }
    tail -n 2 "$d/train.log"
    echo "== $name: eval (best checkpoint, first $SKIP s not scored)"
    $PY -m flow.actionhead eval --ckpt "$d" --export "$EXPORT" \
        --hidden "$HIDDEN" --skip "$SKIP" --json "$d/eval.json" \
        > "$d/eval.log" 2>&1 \
        || { echo "   eval failed, see $d/eval.log"; continue; }
    done_runs+=("$d/eval.json")
done

echo
if [ ${#done_runs[@]} -gt 0 ]; then
    $PY -m flow.actionhead table "${done_runs[@]}" | tee "$OUT/table.txt"
else
    echo "no run finished"
fi
