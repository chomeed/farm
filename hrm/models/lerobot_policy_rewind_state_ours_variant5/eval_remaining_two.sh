#!/bin/bash
#SBATCH --job-name=rewind_eval_missing
#SBATCH --time=03:00:00
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Fills in the missing eval_val_last/trajectories.npz for the two checkpoints
# that haven't been evaluated yet, using the same protocol as eval_varlen_variants.sh
# (last 50 episodes of each of success/failure).

SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure
NUM_VAL_EPISODES=50
NUM_VISUALIZE=5

CKPT_BASE=/home/chomeed/workspace/insertion_generation/ckpts/rewind

for NAME in \
    rewind_state_ours_variant5_robomimic_square_v1_eps200_clip100_longer \
    rewind_state_ours_variant5_varlen_dtw_only_strict_robomimic_square_v1_eps200_clip100_longer; do

  RUN_DIR=$CKPT_BASE/$NAME
  CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
  OUT_DIR="$RUN_DIR/eval_val_last"

  if [ ! -d "$CKPT" ]; then
    echo "[skip] $NAME - no checkpoint at $CKPT"
    continue
  fi
  if [ -f "$OUT_DIR/trajectories.npz" ]; then
    echo "[skip] $NAME - trajectories.npz already exists at $OUT_DIR"
    continue
  fi

  echo "[$(date)] Eval $NAME -> $OUT_DIR"
  python -m lerobot_policy_rewind_state_ours_variant5.common.eval_val_loss_and_visualize \
    --ckpt              "$CKPT" \
    --success_repo_id   "$SUCCESS_REPO" \
    --failure_repo_id   "$FAILURE_REPO" \
    --num_val_episodes  "$NUM_VAL_EPISODES" \
    --num_visualize     "$NUM_VISUALIZE" \
    --output_dir        "$OUT_DIR"
done

echo "[$(date)] Done. Now run compare_rewind_models_reward_predictions.sh to build the overlays."
