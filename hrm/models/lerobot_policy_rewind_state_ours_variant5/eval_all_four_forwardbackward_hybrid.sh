#!/bin/bash
#SBATCH --job-name=rewind_eval_fb_hybrid
#SBATCH --time=03:00:00
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%A_%a.out
#SBATCH --array=0-3

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Hybrid-sampling eval for the four forwardbackward-trained variant5 ckpts.
# Mirrors eval_all_six_hybrid.sh: per-frame predicted reward on the LAST 50
# success + 50 failure episodes (eps 150..199), written to
# <run_dir>/eval_val_last_hybrid/trajectories.npz so the comparison launcher
# can read them.
#
# Slurm array: one variant per array task on its own GPU.
#   TASK 0 -> varlen_blend
#   TASK 1 -> varlen_cosine_critanchor
#   TASK 2 -> varlen_cosine
#   TASK 3 -> varlen_dtw_only_strict
#
# Submit once:
#   sbatch hrm/models/lerobot_policy_rewind_state_ours_variant5/eval_all_four_forwardbackward_hybrid.sh

SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure
NUM_VAL_EPISODES=50
NUM_VISUALIZE=5
SAMPLING=hybrid
N_RECENT=8

# Training writes here; if you later move ckpts to .../insertion_generation/ckpts,
# override CKPT_BASE via env var.
CKPT_BASE=${CKPT_BASE:-/scratch2/chomeed/outputs/rewind}

VARIANT_NAMES=(
  varlen_blend
  varlen_cosine_critanchor
  varlen_cosine
  varlen_dtw_only_strict
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-}
if [ -z "$TASK_ID" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is unset. Submit via sbatch or set manually 0..3."
  exit 1
fi
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#VARIANT_NAMES[@]}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID out of range 0..$((${#VARIANT_NAMES[@]} - 1))"
  exit 1
fi

VARIANT="${VARIANT_NAMES[$TASK_ID]}"
RUN_TAG="rewind_state_ours_variant5_${VARIANT}_robomimic_square_v1_eps200_clip100_longer_forwardbackward"
RUN_DIR="$CKPT_BASE/$RUN_TAG"
CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
OUT_DIR="$RUN_DIR/eval_val_last_${SAMPLING}"

if [ ! -d "$CKPT" ]; then
  echo "[skip] $VARIANT - no checkpoint at $CKPT (training not finished?)"
  exit 0
fi
if [ -f "$OUT_DIR/trajectories.npz" ]; then
  echo "[skip] $VARIANT - trajectories.npz already exists at $OUT_DIR"
  exit 0
fi

echo "[$(date)] task $TASK_ID: eval ($SAMPLING) $VARIANT -> $OUT_DIR"
python -m lerobot_policy_rewind_state_ours_variant5.common.eval_val_loss_and_visualize \
  --ckpt              "$CKPT" \
  --success_repo_id   "$SUCCESS_REPO" \
  --failure_repo_id   "$FAILURE_REPO" \
  --num_val_episodes  "$NUM_VAL_EPISODES" \
  --num_visualize     "$NUM_VISUALIZE" \
  --sampling          "$SAMPLING" \
  --n_recent          "$N_RECENT" \
  --output_dir        "$OUT_DIR"

echo "[$(date)] task $TASK_ID: $VARIANT done -> $OUT_DIR"
