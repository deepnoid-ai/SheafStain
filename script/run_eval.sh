#!/bin/bash
# Evaluate generated IHC against the ground truth. The eval scripts are standalone,
# so this runner derives the prediction and ground-truth directories from the YAML
# (results_dir / name / epoch / dataroot / stain) and runs both metric scripts.
#
# The CSVs land in the run directory the images came from,
# <results_dir>/<name>/test_<epoch>_new/, next to stitched/ and the inference
# logs, so one epoch's images, logs and metrics stay together.
#
# Usage:
#   bash script/run_eval.sh [CONFIG]                 # default config.yaml
#   DRYRUN=1 bash script/run_eval.sh
#   PRED_DIR=... GT_DIR=... bash script/run_eval.sh  # override the derived dirs
#   OUT_DIR=... bash script/run_eval.sh              # override where the CSVs go
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config.yaml}"
eval "$(python script/config_env.py "$CONFIG")"

: "${dataroot:?set dataroot in $CONFIG}"
: "${name:?set name in $CONFIG}"

RUN_DIR="${results_dir:-./results}/${name}/test_${epoch:-latest}_new"
PRED="${PRED_DIR:-$RUN_DIR/stitched}"
GT="${GT_DIR:-${dataroot}/image/psi/ihc/${stain:-her2}}"
OUT="${OUT_DIR:-$RUN_DIR}"

Q=(python util/eval_quantitative.py --pred_dir "$PRED" --gt_dir "$GT" --output_csv "$OUT/quant_${name}.csv")
B=(python util/eval_biological.py   --pred_dir "$PRED" --gt_dir "$GT" --output_csv "$OUT/bio_${name}.csv")

echo "+ ${Q[*]}"
echo "+ ${B[*]}"
[ "${DRYRUN:-0}" = "1" ] && exit 0
mkdir -p "$OUT"
"${Q[@]}"
"${B[@]}"
