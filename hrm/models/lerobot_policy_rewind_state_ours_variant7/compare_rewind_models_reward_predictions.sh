#!/bin/bash
source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Overlay the per-frame predicted-reward curves from all six rewind checkpoints
# on the same set of success/failure validation demos.
#
# Pick which eval directory to read from via SUFFIX:
#   SUFFIX=""        -> eval_val_last/        (uniform sampling, default)
#   SUFFIX="_hybrid" -> eval_val_last_hybrid/ (hybrid sampling: 8 uniform + 8 latest)
#
# The pro6000 baseline has eval_val_last50/ from an older run; for
# apples-to-apples results, prefer SUFFIX=_hybrid after running
# eval_all_six_hybrid.sh.

CKPT_BASE=/home/chomeed/workspace/insertion_generation/ckpts/rewind
SUFFIX=${SUFFIX:-_hybrid}
EVAL_DIR_NAME="eval_val_last${SUFFIX}"

OUT_DIR=${OUT_DIR:-$CKPT_BASE/_compare_six${SUFFIX}}
NUM_VISUALIZE=${NUM_VISUALIZE:-8}

# For SUFFIX="" we fall back to eval_val_last50 for pro6000 since it never had
# an eval_val_last/ dir. For SUFFIX="_hybrid" all six share the same dir name.
PRO6000_DIR_NAME="$EVAL_DIR_NAME"
if [ -z "$SUFFIX" ]; then
  PRO6000_DIR_NAME="eval_val_last50"
fi

python -m lerobot_policy_rewind_state_ours_variant7.common.compare_rewind_models_reward_predictions \
  --output_dir   "$OUT_DIR" \
  --num_visualize "$NUM_VISUALIZE" \
  --model "variant7_base:$CKPT_BASE/rewind_state_ours_variant7_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "varlen_blend:$CKPT_BASE/rewind_state_ours_variant7_varlen_blend_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "varlen_cosine:$CKPT_BASE/rewind_state_ours_variant7_varlen_cosine_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "varlen_cosine_critanchor:$CKPT_BASE/rewind_state_ours_variant7_varlen_cosine_critanchor_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "varlen_dtw_only_strict:$CKPT_BASE/rewind_state_ours_variant7_varlen_dtw_only_strict_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "rewind_state_baseline:$CKPT_BASE/rewind_state_robomimic_square_v1_pro6000/$PRO6000_DIR_NAME"

echo "Wrote comparison artifacts to $OUT_DIR (sampling suffix='$SUFFIX')"
