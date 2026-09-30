#!/bin/bash
# SheafStain inference. inference.py reads the YAML as --config (paths, run
# selectors, and the tiling / FARD protocol all live there).
#
# The images are independent, so the test set is split across the GPUs in
# `gpu_ids` and the shards run at once, one process per GPU. inference.py takes
# `--num_workers N --worker_id i` and keeps images[i::N], and the stitched files
# are named after the image id, so the shards write into one directory without
# colliding. With a single GPU this behaves exactly as it did before.
#
# Usage:
#   bash script/run_inference.sh [CONFIG]                  # default config.yaml
#   DRYRUN=1 bash script/run_inference.sh                  # print the commands, run nothing
#   GPUS=0,1,2,3 bash script/run_inference.sh              # override the GPUs to use
#   bash script/run_inference.sh config.yaml --epoch 300   # extra flags override
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config.yaml}"
EXTRA=("${@:2}")
eval "$(python script/config_env.py "$CONFIG")"

# GPUS wins, then gpu_ids from the config, then a single device. gpu_ids is
# "-1" for CPU, which leaves one process on the default device.
if [ -n "${GPUS:-}" ]; then
    IFS=',' read -ra DEVS <<< "$GPUS"
elif [ -n "${gpu_ids:-}" ] && [ "${gpu_ids}" != "-1" ]; then
    IFS=',' read -ra DEVS <<< "$gpu_ids"
else
    DEVS=(0)
fi
NDEV=${#DEVS[@]}

# Single device: keep the original shape, straight to the terminal. The config
# already carries gpu_ids, so no --gpu_ids override is needed here.
if [ "$NDEV" -le 1 ]; then
    CMD=(python inference.py --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"})
    echo "+ ${CMD[*]}"
    [ "${DRYRUN:-0}" = "1" ] && exit 0
    exec "${CMD[@]}"
fi

# Several devices: one worker per GPU. Logs go beside the results, since the
# processes would otherwise interleave on the terminal. EXTRA comes last so an
# explicit --num_workers / --worker_id on the command line still wins.
#
# The log directory carries the epoch, matching where inference.py writes the
# images. Without it every epoch shares one set of infer_gpu<id>.log files and
# each run truncates the previous one.
LOG_DIR="${LOG_DIR:-${results_dir:-./results}/${name:-sheafstain}/test_${epoch:-latest}_new}"
mkdir -p "$LOG_DIR"

echo "inference over ${NDEV} GPUs (${DEVS[*]}), images split ${NDEV} ways"
PIDS=(); TAGS=()
for i in $(seq 0 $((NDEV - 1))); do
    dev="${DEVS[$i]}"
    CMD=(python inference.py --config "$CONFIG"
         --gpu_ids "$dev" --num_workers "$NDEV" --worker_id "$i"
         ${EXTRA[@]+"${EXTRA[@]}"})
    echo "+ [gpu $dev] ${CMD[*]}"
    [ "${DRYRUN:-0}" = "1" ] && continue
    "${CMD[@]}" > "$LOG_DIR/infer_gpu${dev}.log" 2>&1 &
    PIDS+=($!); TAGS+=("gpu $dev: worker $i/$NDEV")
done
[ "${DRYRUN:-0}" = "1" ] && exit 0

echo "logs: $LOG_DIR/infer_gpu*.log"
FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "done  ${TAGS[$i]}"
    else
        echo "FAILED ${TAGS[$i]} (see $LOG_DIR/infer_gpu*.log)"
        FAIL=1
    fi
done
exit $FAIL
