#!/bin/bash
# Precompute the spatial-sheaf presets that training reads. util/compute_vfm_presets.py
# is a standalone script, so this runner pulls the values it needs out of the YAML
# (via script/config_env.py) and passes them as flags.
#
# One preset takes hours, and the ids are independent, so the range is split
# across the GPUs in `gpu_ids` and the shards run at once, one process per GPU.
# With a single GPU this behaves exactly as it did before.
#
# Usage:
#   bash script/run_presets.sh [CONFIG]              # default config.yaml
#   DRYRUN=1 bash script/run_presets.sh              # print the commands, run nothing
#   PRESET_START=0 PRESET_END=8 bash script/run_presets.sh   # override the id range
#   GPUS=0,1 bash script/run_presets.sh              # override the GPUs to use
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config.yaml}"
EXTRA=("${@:2}")
eval "$(python script/config_env.py "$CONFIG")"

: "${vfm_model_path:?set vfm_model_path in $CONFIG}"
: "${dataroot:?set dataroot in $CONFIG}"
: "${sheaf_preset_dir:?set sheaf_preset_dir in $CONFIG}"

START="${PRESET_START:-${preset_start:-0}}"
END="${PRESET_END:-${preset_end:-1}}"
[ "$END" -ge "$START" ] || { echo "preset_end ($END) is below preset_start ($START)"; exit 1; }

# GPUS wins, then gpu_ids from the config, then a single device. gpu_ids is
# "-1" for CPU, which leaves one process on device 0's default.
if [ -n "${GPUS:-}" ]; then
    IFS=',' read -ra DEVS <<< "$GPUS"
elif [ -n "${gpu_ids:-}" ] && [ "${gpu_ids}" != "-1" ]; then
    IFS=',' read -ra DEVS <<< "$gpu_ids"
else
    DEVS=(0)
fi
NDEV=${#DEVS[@]}

preset_cmd() {   # $1 = gpu id, $2 = first preset, $3 = last preset
    echo python util/compute_vfm_presets.py \
        --vfm_name "${vfm_name:-gigapath}" \
        --vfm_model_path "$vfm_model_path" \
        --dataroot "$dataroot" \
        --stain "${stain:-her2}" \
        --img_ext "${img_ext:-.png}" \
        --train_split_mode "${train_split_mode:-bci}" \
        --output_dir "$sheaf_preset_dir" \
        --preset_start "$2" --preset_end "$3" \
        --gpu "$1" ${EXTRA[@]+"${EXTRA[@]}"}
}

# Single device: keep the original shape, straight to the terminal.
if [ "$NDEV" -le 1 ]; then
    CMD=$(preset_cmd "${DEVS[0]}" "$START" "$END")
    echo "+ $CMD"
    [ "${DRYRUN:-0}" = "1" ] && exit 0
    exec $CMD
fi

# Several devices: contiguous blocks, one process each. Logs go beside the
# presets, since the processes would otherwise interleave on the terminal.
TOTAL=$((END - START + 1))
PER=$(( (TOTAL + NDEV - 1) / NDEV ))
LOG_DIR="${LOG_DIR:-$sheaf_preset_dir}"
mkdir -p "$LOG_DIR"

echo "presets $START..$END ($TOTAL) over ${NDEV} GPUs (${DEVS[*]}), up to $PER each"
PIDS=(); TAGS=()
for i in $(seq 0 $((NDEV - 1))); do
    s=$((START + i * PER))
    [ "$s" -gt "$END" ] && break
    e=$((s + PER - 1)); [ "$e" -gt "$END" ] && e=$END
    dev="${DEVS[$i]}"
    CMD=$(preset_cmd "$dev" "$s" "$e")
    echo "+ [gpu $dev] $CMD"
    [ "${DRYRUN:-0}" = "1" ] && continue
    $CMD > "$LOG_DIR/preset_gpu${dev}.log" 2>&1 &
    PIDS+=($!); TAGS+=("gpu $dev: presets $s..$e")
done
[ "${DRYRUN:-0}" = "1" ] && exit 0

echo "logs: $LOG_DIR/preset_gpu*.log"
FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "done  ${TAGS[$i]}"
    else
        echo "FAILED ${TAGS[$i]} (see $LOG_DIR/preset_gpu*.log)"
        FAIL=1
    fi
done
exit $FAIL
