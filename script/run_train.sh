#!/bin/bash
# SheafStain training. All parameters come from the YAML config (train.py reads
# it as --config; the recipe, paths, and run selectors live there).
#
# Usage:
#   bash script/run_train.sh [CONFIG]                 # default CONFIG=config.yaml
#   NGPU=8 bash script/run_train.sh config.yaml       # multi-GPU via torchrun
#   DRYRUN=1 bash script/run_train.sh                 # print the command, run nothing
#   bash script/run_train.sh config.yaml --stain er   # extra flags override the YAML
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config.yaml}"
eval "$(python script/config_env.py "$CONFIG")"
NGPU="${NGPU:-${num_gpus:-1}}"

if [ "$NGPU" -gt 1 ]; then
  CMD=(torchrun --nproc_per_node="$NGPU" train.py --config "$CONFIG")
else
  CMD=(python train.py --config "$CONFIG")
fi
CMD+=("${@:2}")

echo "+ ${CMD[*]}"
[ "${DRYRUN:-0}" = "1" ] && exit 0
exec "${CMD[@]}"
