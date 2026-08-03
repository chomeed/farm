#!/bin/bash
#SBATCH --job-name=rewind_eval_robomimic_square_resized_peg_variant6_hybrid
#SBATCH --time=06:00:00
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Per-clip predicted-reward visualization for the variant-6 ckpt trained by
# train_rewind_state_ours_variant6_robomimic_square_resized_peg_blend{,_exaflops}.sh.
# Hybrid sampling (8 uniform + 8 most-recent frames) so the visualization
# matches the trainer's failure-clip sampler exactly.
#
# Override CKPT_BASE if reading from a non-default outputs root (e.g. exaflops
# /scratch2/exaFLOPs/chomeed/outputs/rewind/...).

CKPT_BASE=${CKPT_BASE:-/scratch2/chomeed/outputs/rewind}
NAME=rewind_state_ours_variant6_robomimic_square_resized_peg_v1_blend_mono_anchorstack_eps0.2
RUN_DIR=$CKPT_BASE/$NAME
CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
OUT_DIR="$RUN_DIR/eval_val_last_hybrid"

SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure
NUM_VAL_EPISODES=50
NUM_VISUALIZE=5
SAMPLING=hybrid
N_RECENT=8

if [ ! -d "$CKPT" ]; then
  echo "ERROR: ckpt missing at $CKPT" >&2
  echo "       (run train_rewind_state_ours_variant6_robomimic_square_resized_peg_blend{,_exaflops}.sh first)" >&2
  exit 1
fi
if [ -f "$OUT_DIR/trajectories.npz" ]; then
  echo "[skip] eval already done -- $OUT_DIR/trajectories.npz exists"
  exit 0
fi

echo "[$(date)] Eval ($SAMPLING) $NAME -> $OUT_DIR"
python -m lerobot_policy_rewind_state_ours_variant6.common.eval_val_loss_and_visualize \
  --ckpt              "$CKPT" \
  --success_repo_id   "$SUCCESS_REPO" \
  --failure_repo_id   "$FAILURE_REPO" \
  --num_val_episodes  "$NUM_VAL_EPISODES" \
  --num_visualize     "$NUM_VISUALIZE" \
  --sampling          "$SAMPLING" \
  --n_recent          "$N_RECENT" \
  --output_dir        "$OUT_DIR"

echo "[$(date)] Done -> $OUT_DIR"
