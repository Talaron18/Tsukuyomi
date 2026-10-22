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
#   MODELS    folders holding the .model3.json files, colon separated
#             [live2d-models:CubismWebSamples] (missing ones are ignored)
#   DET_CKPT  deterministic prior checkpoint, or empty to skip that source [prior.pt]
#   EPOCHS    flow-prior epochs [40]
#   SECONDS_  seconds per scored run [60]
set -euo pipefail

# python -m flow.X resolves against the current directory, so a run started
# anywhere but the repo root found no `flow` package. Point Python at the repo
# this script lives in instead.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

HOLDOUT=${HOLDOUT:-aidang_2,Natori,Hiyori,lafei,xuefeng}
MODELS=${MODELS:-live2d-models:CubismWebSamples}
IFS=: read -ra model_roots <<< "$MODELS"
roots=()
for r in "${model_roots[@]}"; do [[ -e "$r" ]] && roots+=("$r"); done
if (( ${#roots[@]} == 0 )); then
    echo "no model folder found among: $MODELS" >&2; exit 1
fi
DET_CKPT=${DET_CKPT-prior.pt}
EPOCHS=${EPOCHS:-40}
SECS=${SECONDS_:-60}
STEPS=${STEPS:-"clips fit train score render ab"}

has() { [[ " $STEPS " == *" $1 "* ]]; }

model_of() {
    # The rig file is named after the model file, so search by that name.
    # -L: MODELS may itself be a symlink (e.g. in a side directory for one
    # source), and find does not descend into a symlinked start point.
    find -L "${roots[@]}" -name "$1.model3.json" -not -name '*.probe.*' \
        -print -quit
}

mkdir -p runs vids

if has clips; then
    # Clips built before the vowel-band fix carry ParamA/I/U/E/O as motion.
    globs=()
    for r in "${roots[@]}"; do globs+=(--models "$r/**/*.model3.json"); done
    python -m l2dprobe.motions "${globs[@]}" --rigs rigs --out clips2
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
        echo "!! $R: no $R.model3.json under ${roots[*]}, skipping" >&2
        continue
    fi
    base=(--model "$M" --rig "rigs/$R.npz" --clips "clips2/$R.npz")
    for x in "${sources[@]}"; do
        read -ra extra <<< "$(src_args "$x")"
        if has score; then
            # One broken source (say, a checkpoint that no longer matches the
            # rig files) must not abort every other rig and source.
            ok=1
            for s in 0 1 2; do
                if ! python -m flow.runtime "${base[@]}" "${extra[@]}" \
                        --seed "$s" --seconds "$SECS" \
                        --dump "runs/$R.$x.$s.npz" --dump-only; then
                    echo "!! $R/$x: runtime failed, skipping this source" >&2
                    echo "$R/$x" >> runs/FAILED.txt
                    ok=0; break
                fi
            done
            [[ $ok == 1 ]] || continue
            # Exit code 2 is a FAIL verdict, not an error; keep going.
            python -m flow.score --traj "runs/$R.$x.0.npz" \
                --seeds "runs/$R.$x.1.npz" "runs/$R.$x.2.npz" \
                --clips "clips2/$R.npz" --rig "rigs/$R.npz" --model "$M" \
                --novelty-ref clips2 --holdout "$HOLDOUT" \
                --json "runs/$R.$x.json" | tee "runs/$R.$x.txt" || true
        fi
        if has render; then
            xvfb-run -a -s "-screen 0 1024x1024x24" python -m flow.runtime \
                "${base[@]}" "${extra[@]}" --seconds 20 --out "vids/$R/$x" \
                || echo "!! $R/$x: render failed" >&2
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
