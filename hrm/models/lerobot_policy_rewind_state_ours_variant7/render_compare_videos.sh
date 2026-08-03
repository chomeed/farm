#!/bin/bash
source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Render per-episode side-by-side mp4s for each of the six rewind ckpts:
# camera footage on the left, all six predicted-reward curves on the right with
# a vertical cursor synced to the current frame.
#
# Requires that the eval_val_last${SUFFIX}/trajectories.npz exist for every
# checkpoint. For SUFFIX="" we fall back to eval_val_last50/ for pro6000.
#
# Knobs:
#   SUFFIX        -> matches the comparison script (default _hybrid)
#   NUM_EPISODES  -> per split (success, failure)
#   EPISODES      -> comma-separated episode indices (overrides NUM_EPISODES)
#   SPLITS        -> e.g. "success" or "failure" or both
#   FPS, PLOT_H   -> output video fps / chart panel height
#   OUT_DIR       -> root for outputs; videos go under $OUT_DIR/videos/<split>/

CKPT_BASE=/home/chomeed/workspace/insertion_generation/ckpts/rewind
SUFFIX=${SUFFIX:-_hybrid}
EVAL_DIR_NAME="eval_val_last${SUFFIX}"
NUM_EPISODES=${NUM_EPISODES:-20}
EPISODES=${EPISODES:-}
SPLITS=${SPLITS:-success,failure}
FPS=${FPS:-20}
PLOT_H=${PLOT_H:-360}
OUT_DIR=${OUT_DIR:-$CKPT_BASE/_compare_six${SUFFIX}}

PRO6000_DIR_NAME="$EVAL_DIR_NAME"
if [ -z "$SUFFIX" ]; then
  PRO6000_DIR_NAME="eval_val_last50"
fi

CMD=(
  python -m lerobot_policy_rewind_state_ours_variant7.common.render_compare_videos
  --output_dir   "$OUT_DIR"
  --splits       "$SPLITS"
  --num_episodes "$NUM_EPISODES"
  --fps          "$FPS"
  --plot_height  "$PLOT_H"
  --model "variant7_base:$CKPT_BASE/rewind_state_ours_variant7_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "varlen_blend:$CKPT_BASE/rewind_state_ours_variant7_varlen_blend_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "varlen_cosine:$CKPT_BASE/rewind_state_ours_variant7_varlen_cosine_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "varlen_cosine_critanchor:$CKPT_BASE/rewind_state_ours_variant7_varlen_cosine_critanchor_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "varlen_dtw_only_strict:$CKPT_BASE/rewind_state_ours_variant7_varlen_dtw_only_strict_robomimic_square_v1_eps200_clip100_longer/$EVAL_DIR_NAME"
  --model "rewind_state_baseline:$CKPT_BASE/rewind_state_robomimic_square_v1_pro6000/$PRO6000_DIR_NAME"
)

if [ -n "$EPISODES" ]; then
  CMD+=(--episodes "$EPISODES")
fi

"${CMD[@]}"

echo "Wrote comparison videos under $OUT_DIR/videos/ (sampling suffix='$SUFFIX')"
