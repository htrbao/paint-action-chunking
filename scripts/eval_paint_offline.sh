#!/usr/bin/env bash
# Offline PAINT evaluation against a finetuned checkpoint and a dataset.
#
# Runs three arms over the same trajectories so the numbers are comparable:
#   none          plain sampler, synchronous execution (accuracy reference)
#   repaint-euler PAINT, asynchronous execution
#   rtc           guidance baseline, asynchronous execution
#
# The async arms keep executing the in-flight chunk for INFERENCE_DELAY steps
# before switching to the new one, which is the seam prefix-consistent sampling
# targets. Look at "prefix inconsistency" (lower is better) for continuity, and
# MSE/MAE for accuracy. The 'none' arm has no prefix metric by construction.
#
# Usage:
#   bash scripts/eval_paint_offline.sh <model_path> <dataset_path> <embodiment_tag>
# Example:
#   bash scripts/eval_paint_offline.sh \
#       /path/to/checkpoint-10000 demo_data/libero_demo/ libero_sim

set -euo pipefail

MODEL_PATH="${1:?usage: $0 <model_path> <dataset_path> <embodiment_tag>}"
DATASET_PATH="${2:?usage: $0 <model_path> <dataset_path> <embodiment_tag>}"
EMBODIMENT_TAG="${3:?usage: $0 <model_path> <dataset_path> <embodiment_tag>}"

# Tune these. EXECUTION_HORIZON is how many steps you execute per inference;
# INFERENCE_DELAY is how many are already committed when the new chunk lands.
# Both must satisfy EXECUTION_HORIZON + INFERENCE_DELAY <= model action horizon.
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
INFERENCE_DELAY="${INFERENCE_DELAY:-2}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
STEPS="${STEPS:-200}"
TRAJ_IDS="${TRAJ_IDS:-0 1 2}"
OUT_DIR="${OUT_DIR:-/tmp/paint_eval}"

mkdir -p "$OUT_DIR"

echo "model=$MODEL_PATH dataset=$DATASET_PATH tag=$EMBODIMENT_TAG"
echo "execution_horizon=$EXECUTION_HORIZON inference_delay=$INFERENCE_DELAY denoising_steps=$DENOISING_STEPS"
echo "Keep denoising_steps identical across arms: the PAINT prefix residual is"
echo "O(dt), so it changes with the step count and arms would not be comparable."
echo

run_arm() {
  local name="$1"; shift
  local log="$OUT_DIR/${name}.log"
  echo "=== arm: $name ==="
  python gr00t/eval/open_loop_eval.py \
    --model-path "$MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --denoising-steps "$DENOISING_STEPS" \
    --steps "$STEPS" \
    --traj-ids $TRAJ_IDS \
    --save-plot-path "$OUT_DIR/${name}.jpeg" \
    "$@" 2>&1 | tee "$log"
  echo
}

run_arm none
run_arm repaint-euler --smooth-option repaint-euler --inference-delay "$INFERENCE_DELAY"
run_arm rtc          --smooth-option rtc           --inference-delay "$INFERENCE_DELAY"

echo "================ summary ================"
for name in none repaint-euler rtc; do
  echo "--- $name ---"
  grep -E "Average (MSE|MAE|prefix inconsistency)" "$OUT_DIR/${name}.log" || echo "  (no summary; check $OUT_DIR/${name}.log)"
done
echo
echo "Logs and plots in $OUT_DIR"
