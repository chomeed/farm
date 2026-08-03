#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant3_varlen_eval
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

SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure
NUM_VAL_EPISODES=50
NUM_VISUALIZE=5

OUT_BASE=/scratch2/chomeed/outputs/rewind

for SUFFIX in varlen_cosine varlen_blend varlen_cosine_critanchor; do
  RUN_DIR=$OUT_BASE/rewind_state_ours_variant3_${SUFFIX}_robomimic_square_v1_eps200_clip100_longer
  CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
  OUT_DIR="$RUN_DIR/eval_val_last"

  if [ ! -d "$CKPT" ]; then
    echo "[skip] $SUFFIX — no checkpoint at $CKPT"
    continue
  fi

  echo "[$(date)] Eval $SUFFIX -> $OUT_DIR"
  python -m lerobot_policy_rewind_state_ours_variant3.common.eval_val_loss_and_visualize \
    --ckpt              "$CKPT" \
    --success_repo_id   "$SUCCESS_REPO" \
    --failure_repo_id   "$FAILURE_REPO" \
    --num_val_episodes  "$NUM_VAL_EPISODES" \
    --num_visualize     "$NUM_VISUALIZE" \
    --output_dir        "$OUT_DIR"
done

echo "[$(date)] All evals done."
