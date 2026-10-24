#!/usr/bin/env bash
# Remove local leftovers of finished experiments from a working copy.
#
# Lists what it would delete (with sizes) unless --apply is given. Nothing
# here is needed by the current pipeline (flow.events / flow.director /
# flow.live): the rig files (rigs/), clips (clips2/), probe output (out/),
# event statistics (ev.json), models and the latest runs stay.
#
# Usage, from the repo root:
#   bash scripts/tidy.sh            # dry run
#   bash scripts/tidy.sh --apply    # delete
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
apply=0; [[ "${1:-}" == --apply ]] && apply=1

# Old preview renders, checks and debug output, each a folder of frame PNGs.
targets=(preview_* chk_* dbg rigcheck_* tmp
         # Results of the first eval rounds, superseded by runs/ and vids/.
         runs_v1 runs_v2_runtimefix vids_v1
         # Cross-validation logs of the flow prior (abandoned for events).
         fold_*.log
         # Leftover listings and an archive of out/ (out/ itself is kept).
         tree_output.txt project_structure.txt out.tar.gz
         # First clip set (clips2/ replaced it) and the synthetic render
         # dataset of the original phase 1 (flow/legacy/render_dataset.py).
         clips data synth manifest*.json
         # Checkpoints of the abandoned priors; fp.pt stays, phase2_eval.sh
         # still renders the flow source from it.
         prior_*.pt fp_damped.pt flowprior_log.json)

# Rendered frames: every render/ folder keeps its preview.mp4; the PNGs it
# was encoded from are 20k+ files under runs/ and 12k under vids/.
mapfile -t frames < <(find runs vids -path '*/render/*.png' -o \
                           -path 'vids/*/*/f*.png' 2>/dev/null)

total=0
for t in "${targets[@]}"; do
    [[ -e "$t" ]] || continue
    kb=$(du -sk "$t" | cut -f1); total=$((total + kb))
    printf '%8s  %s\n' "$(du -sh "$t" | cut -f1)" "$t"
done
if (( ${#frames[@]} )); then
    kb=$(printf '%s\0' "${frames[@]}" | du -ck --files0-from=- | tail -1 | cut -f1)
    total=$((total + kb))
    printf '%8s  %d rendered frame PNGs under runs/ and vids/\n' \
        "$((kb / 1024))M" "${#frames[@]}"
fi
echo "total: $((total / 1024)) MB"

if (( apply )); then
    for t in "${targets[@]}"; do [[ -e "$t" ]] && rm -rf -- "$t"; done
    (( ${#frames[@]} )) && printf '%s\0' "${frames[@]}" | xargs -0 rm -f
    echo "deleted"
else
    echo "dry run; rerun with --apply to delete"
fi
