#!/bin/bash
source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Render per-episode side-by-side mp4s with all 10 rewind ckpts overlaid:
# the 6 original variant3 models PLUS the 4 forwardbackward variants.
# Reads each ckpt's eval_val_last_hybrid/trajectories.npz (no model load).
#
# Knobs (env vars):
#   NUM_EPISODES  -> per split (default 20)
#   EPISODES      -> comma-separated explicit episode indices (overrides NUM_EPISODES)
#   SPLITS        -> default "success,failure"
#   FPS, PLOT_H   -> video fps / chart panel height
#   OUT_DIR       -> root for outputs; videos go under $OUT_DIR/videos/<split>/
#   CKPT_BASE_FB  -> override forwardbackward ckpts base (default /scratch2/...)

CKPT_BASE_ORIG=/home/chomeed/workspace/insertion_generation/ckpts/rewind
CKPT_BASE_FB=${CKPT_BASE_FB:-/scratch2/chomeed/outputs/rewind}
EVAL_DIR_NAME=eval_val_last_hybrid
NUM_EPISODES=${NUM_EPISODES:-20}
EPISODES=${EPISODES:-}
SPLITS=${SPLITS:-success,failure}
FPS=${FPS:-20}
PLOT_H=${PLOT_H:-360}
OUT_DIR=${OUT_DIR:-$CKPT_BASE_ORIG/_compare_ten_hybrid}

CMD=(
  python -m lerobot_policy_rewind_state_ours_variant3.common.render_compare_videos
  --output_dir   "$OUT_DIR"
  --splits       "$SPLITS"
  --num_episodes "$NUM_EPISODES"
  --fps          "$FPS"
  --plot_height  "$PLOT_H"
  --model "variant3_base:$CKPT_BASE_ORIG/rewind_state_ours_variant3_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "varlen_blend:$CKPT_BASE_ORIG/rewind_state_ours_variant3_varlen_blend_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "varlen_cosine:$CKPT_BASE_ORIG/rewind_state_ours_variant3_varlen_cosine_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "varlen_cosine_critanchor:$CKPT_BASE_ORIG/rewind_state_ours_variant3_varlen_cosine_critanchor_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "varlen_dtw_only_strict:$CKPT_BASE_ORIG/rewind_state_ours_variant3_varlen_dtw_only_strict_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "rewind_state_baseline:$CKPT_BASE_ORIG/rewind_state_robomimic_square_v1_pro6000/$EVAL_DIR_NAME"
  --model "varlen_blend_fb:$CKPT_BASE_FB/rewind_state_ours_variant3_varlen_blend_robomimic_square_v1_eps200_clip100_longer_forwardbackward/$EVAL_DIR_NAME"
  --model "varlen_cosine_critanchor_fb:$CKPT_BASE_FB/rewind_state_ours_variant3_varlen_cosine_critanchor_robomimic_square_v1_eps200_clip100_longer_forwardbackward/$EVAL_DIR_NAME"
  --model "varlen_cosine_fb:$CKPT_BASE_FB/rewind_state_ours_variant3_varlen_cosine_robomimic_square_v1_eps200_clip100_longer_forwardbackward/$EVAL_DIR_NAME"
  --model "varlen_dtw_only_strict_fb:$CKPT_BASE_FB/rewind_state_ours_variant3_varlen_dtw_only_strict_robomimic_square_v1_eps200_clip100_longer_forwardbackward/$EVAL_DIR_NAME"
)

if [ -n "$EPISODES" ]; then
  CMD+=(--episodes "$EPISODES")
fi

"${CMD[@]}"

echo "Wrote 10-model comparison videos under $OUT_DIR/videos/"
