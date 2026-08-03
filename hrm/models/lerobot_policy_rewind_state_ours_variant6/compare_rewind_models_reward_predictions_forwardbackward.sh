#!/bin/bash
source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Overlay the per-frame predicted-reward curves from all 10 rewind ckpts:
# the 6 original variant6 models PLUS the 4 forwardbackward variants
# (approaching+leaving failure clips).
#
# Reads each ckpt's eval_val_last_hybrid/trajectories.npz; the 4 FB ckpts must
# already have been evaluated by eval_all_four_forwardbackward_hybrid.sh, and
# the 6 originals by eval_all_six_hybrid.sh.

CKPT_BASE_ORIG=/home/chomeed/workspace/insertion_generation/ckpts/rewind
CKPT_BASE_FB=${CKPT_BASE_FB:-/scratch2/chomeed/outputs/rewind}
EVAL_DIR_NAME=eval_val_last_hybrid

OUT_DIR=${OUT_DIR:-$CKPT_BASE_ORIG/_compare_ten_hybrid}
NUM_VISUALIZE=${NUM_VISUALIZE:-8}

python -m lerobot_policy_rewind_state_ours_variant6.common.compare_rewind_models_reward_predictions \
  --output_dir   "$OUT_DIR" \
  --num_visualize "$NUM_VISUALIZE" \
  --model "variant6_base:$CKPT_BASE_ORIG/rewind_state_ours_variant6_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "varlen_blend:$CKPT_BASE_ORIG/rewind_state_ours_variant6_varlen_blend_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "varlen_cosine:$CKPT_BASE_ORIG/rewind_state_ours_variant6_varlen_cosine_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "varlen_cosine_critanchor:$CKPT_BASE_ORIG/rewind_state_ours_variant6_varlen_cosine_critanchor_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "varlen_dtw_only_strict:$CKPT_BASE_ORIG/rewind_state_ours_variant6_varlen_dtw_only_strict_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME" \
  --model "rewind_state_baseline:$CKPT_BASE_ORIG/rewind_state_robomimic_square_v1_pro6000/$EVAL_DIR_NAME" \
  --model "varlen_blend_fb:$CKPT_BASE_FB/rewind_state_ours_variant6_varlen_blend_robomimic_square_v1_eps200_clip100_longer_forwardbackward/$EVAL_DIR_NAME" \
  --model "varlen_cosine_critanchor_fb:$CKPT_BASE_FB/rewind_state_ours_variant6_varlen_cosine_critanchor_robomimic_square_v1_eps200_clip100_longer_forwardbackward/$EVAL_DIR_NAME" \
  --model "varlen_cosine_fb:$CKPT_BASE_FB/rewind_state_ours_variant6_varlen_cosine_robomimic_square_v1_eps200_clip100_longer_forwardbackward/$EVAL_DIR_NAME" \
  --model "varlen_dtw_only_strict_fb:$CKPT_BASE_FB/rewind_state_ours_variant6_varlen_dtw_only_strict_robomimic_square_v1_eps200_clip100_longer_forwardbackward/$EVAL_DIR_NAME"

echo "Wrote 10-model comparison artifacts to $OUT_DIR"
