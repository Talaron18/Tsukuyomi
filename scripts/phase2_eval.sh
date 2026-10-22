#!/usr/bin/env bash
# Phase 2 on the full rig set: fit the behaviour baseline, train the flow
# prior, then dump, score and render every source on the held-out rigs.
#
# Model paths are found with `find` and always quoted: rig folders such as
# "Azue Lane(JP)" contain spaces and brackets, which broke unquoted globs.
#
# Usage, from the repo root:
#   bash scripts/phase2_eval.sh                  # everything
#   STEPS="score render" bash scripts/phase2_eval.sh   # skip fit/train
#
# Environment knobs (defaults in brackets):
#   HOLDOUT   held-out rigs, comma separated [aidang_2,Natori,Hiyori,lafei,xuefeng]
#   MODELS    root holding the .model3.json files [live2d-models]
#   DET_CKPT  deterministic prior checkpoint, or empty to skip that source [prior.pt]
#   EPOCHS    flow-prior epochs [40]
#   SECONDS_  seconds per scored run [60]
set -euo pipefail

HOLDOUT=${HOLDOUT:-aidang_2,Natori,Hiyori,lafei,xuefeng}
MODELS=${MODELS:-live2d-models}
DET_CKPT=${DET_CKPT-prior.pt}
EPOCHS=${EPOCHS:-40}
SECS=${SECONDS_:-60}
STEPS=${STEPS:-"clips fit train score render ab"}

has() { [[ " $STEPS " == *" $1 "* ]]; }

model_of() {
    # The rig file is named after the model file, so search by that name.
    find "$MODELS" -name "$1.model3.json" -not -name '*.probe.*' -print -quit
}

mkdir -p runs vids

if has clips; then
    # Clips built before the vowel-band fix carry ParamA/I/U/E/O as motion.
    python -m l2dprobe.motions --models "$MODELS/**/*.model3.json" \
        --rigs rigs --out clips2
fi
if has fit; then
    python -m flow.behaviors --fit clips2 --holdout "$HOLDOUT" --out beh.json
fi
if has train; then
    python -m flow.flowprior clips2 --rigs rigs --holdout "$HOLDOUT" \
        --epochs "$EPOCHS" --steps 200 --threads "$(nproc)" --device auto \
        --save fp.pt
fi

sources=(beh flow static)
[[ -n "$DET_CKPT" && -f "$DET_CKPT" ]] && sources+=(det)

src_args() {
    case $1 in
        beh)    echo --behaviors beh.json ;;
        flow)   echo --ckpt fp.pt ;;
        det)    echo --ckpt "$DET_CKPT" ;;
        static) echo --static ;;
    esac
}

IFS=, read -ra rigs_held <<< "$HOLDOUT"
for R in "${rigs_held[@]}"; do
    M=$(model_of "$R")
    if [[ -z "$M" ]]; then
        echo "!! $R: no $R.model3.json under $MODELS, skipping" >&2
        continue
    fi
    base=(--model "$M" --rig "rigs/$R.npz" --clips "clips2/$R.npz")
    for x in "${sources[@]}"; do
        read -ra extra <<< "$(src_args "$x")"
        if has score; then
            for s in 0 1 2; do
                python -m flow.runtime "${base[@]}" "${extra[@]}" --seed "$s" \
                    --seconds "$SECS" --dump "runs/$R.$x.$s.npz" --dump-only
            done
            # Exit code 2 is a FAIL verdict, not an error; keep going.
            python -m flow.score --traj "runs/$R.$x.0.npz" \
                --seeds "runs/$R.$x.1.npz" "runs/$R.$x.2.npz" \
                --clips "clips2/$R.npz" --rig "rigs/$R.npz" --model "$M" \
                --novelty-ref clips2 --holdout "$HOLDOUT" \
                --json "runs/$R.$x.json" | tee "runs/$R.$x.txt" || true
        fi
        if has render; then
            xvfb-run -a -s "-screen 0 1024x1024x24" python -m flow.runtime \
                "${base[@]}" "${extra[@]}" --seconds 20 --out "vids/$R/$x"
        fi
    done
done

if has ab; then
    items=()
    for R in "${rigs_held[@]}"; do
        for x in "${sources[@]}"; do
            v="vids/$R/$x/preview.mp4"
            [[ -f "$v" ]] && items+=("$R/$x=$v")
        done
    done
    python -m flow.abtest make ab "${items[@]}"
    echo "Judge ab/index.html, write ab/judgments.txt, then:"
    echo "  python -m flow.abtest tally ab"
fi
