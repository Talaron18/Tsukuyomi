#!/usr/bin/env bash
# The probes of docs/embodiment.md ("Probes", under the BEAT2 action head
# ablations): does the voice predict the motion in the ground truth
# (prosody), does the text or the prosody carry information about it
# (match, a contrastive probe), both (all). Writes report.txt and
# report.json in $OUT.
#
# Usage, from the repo root:
#   bash scripts/beat2_probe.sh                  # all, the default budget
#   MODE=prosody bash scripts/beat2_probe.sh     # numpy only, no hidden states
#   LIMIT=2000 STEPS=500 bash scripts/beat2_probe.sh     # a quick look
#   EXTRA="--split speaker --layer all" bash scripts/beat2_probe.sh
# Variables: MODE (prosody | match | all), EXPORT, HIDDEN, OUT, LIMIT
# (segments, whole clips; 0 = all), STEPS (match: per variant), SEM (the
# folder of BEAT2's semantic annotations, if you have it), PY, EXTRA (flags
# added to the command).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PY=${PY:-python}
MODE=${MODE:-all}
EXPORT=${EXPORT:-data/beat2/seed-san/action}
HIDDEN=${HIDDEN:-$HOME/tsk-runs/hidden/beat2}
OUT=${OUT:-$HOME/tsk-runs/probe}
LIMIT=${LIMIT:-0}
STEPS=${STEPS:-3000}
SEM=${SEM:-}

args=(--export "$EXPORT" --out "$OUT" --limit "$LIMIT")
if [ "$MODE" != prosody ]; then
    args+=(--hidden "$HIDDEN" --steps "$STEPS")
fi
if [ -n "$SEM" ]; then
    args+=(--sem "$SEM")
fi
# shellcheck disable=SC2086
exec $PY -m flow.probe "$MODE" "${args[@]}" ${EXTRA:-}
