#!/usr/bin/env bash
# Find the flow-prior training settings that use this machine best.
#
# The model is small (~1M parameters) and trains on batches of 16, too little
# work per step to spread over many cores: past some thread count each step
# gets no faster. Larger batches give each step more work, but also change
# training (fewer, larger steps per epoch), so choose them by measurement.
#
# Times a short slice of training (no evaluation) for each THREADS x BATCH
# pair and prints samples per second. Pick the fastest THREADS for the batch
# you train with; compare batches by samples/s, not seconds.
#
# Usage, from the repo root:
#   bash scripts/bench_train.sh                     # default grid
#   THREADS_LIST="8 16 36" BATCHES="16 64" bash scripts/bench_train.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

CLIPS=${CLIPS:-clips2}
RIGS=${RIGS:-rigs}
STEPS=${STEPS:-60}
THREADS_LIST=${THREADS_LIST:-"4 8 12 18 36"}
BATCHES=${BATCHES:-"16 64 128"}

printf "%-6s %-8s %8s %12s\n" batch threads seconds samples/s
for b in $BATCHES; do
    for t in $THREADS_LIST; do
        s=$(date +%s.%N)
        # --holdout NONE matches no rig, so no held-out evaluation runs.
        python -m flow.flowprior "$CLIPS" --rigs "$RIGS" --holdout NONE \
            --epochs 1 --steps "$STEPS" --batch "$b" --threads "$t" \
            --base hold > /dev/null 2>&1
        e=$(date +%s.%N)
        python3 -c "d=$e-$s; print(f'{$b:<6} {$t:<8} {d:8.1f} {$STEPS*$b/d:12.0f}')"
    done
done
