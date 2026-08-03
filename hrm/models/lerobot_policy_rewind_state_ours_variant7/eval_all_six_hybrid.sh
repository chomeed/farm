#!/bin/bash
#SBATCH --job-name=rewind_eval_hybrid_all
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

# Re-evaluate all six rewind checkpoints with the new hybrid sampling
# (8 uniform + 8 most-recent frames per 16-frame window) so the comparison
# is on equal footing. The existing eval_val_last/ dirs were produced with
# sampling="uniform" and are kept untouched; hybrid outputs go to
# eval_val_last_hybrid/.

SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure
NUM_VAL_EPISODES=50
NUM_VISUALIZE=5
SAMPLING=hybrid
N_RECENT=8

CKPT_BASE=/home/chomeed/workspace/insertion_generation/ckpts/rewind

NAMES=(
  rewind_state_ours_variant7_robomimic_square_v1_eps200_clip100_longer
  rewind_state_ours_variant7_varlen_blend_robomimic_square_v1_eps200_clip100_longer
  rewind_state_ours_variant7_varlen_cosine_robomimic_square_v1_eps200_clip100_longer
  rewind_state_ours_variant7_varlen_cosine_critanchor_robomimic_square_v1_eps200_clip100_longer
  rewind_state_ours_variant7_varlen_dtw_only_strict_robomimic_square_v1_eps200_clip100_longer
  rewind_state_robomimic_square_v1_pro6000
)

for NAME in "${NAMES[@]}"; do
  RUN_DIR=$CKPT_BASE/$NAME
  CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
  OUT_DIR="$RUN_DIR/eval_val_last_${SAMPLING}"

  if [ ! -d "$CKPT" ]; then
    echo "[skip] $NAME - no checkpoint at $CKPT"
    continue
  fi
  if [ -f "$OUT_DIR/trajectories.npz" ]; then
    echo "[skip] $NAME - trajectories.npz already exists at $OUT_DIR"
    continue
  fi

  echo "[$(date)] Eval ($SAMPLING) $NAME -> $OUT_DIR"
  python -m lerobot_policy_rewind_state_ours_variant7.common.eval_val_loss_and_visualize \
    --ckpt              "$CKPT" \
    --success_repo_id   "$SUCCESS_REPO" \
    --failure_repo_id   "$FAILURE_REPO" \
    --num_val_episodes  "$NUM_VAL_EPISODES" \
    --num_visualize     "$NUM_VISUALIZE" \
    --sampling          "$SAMPLING" \
    --n_recent          "$N_RECENT" \
    --output_dir        "$OUT_DIR"
done

echo "[$(date)] Done. Now run compare_rewind_models_reward_predictions.sh with SUFFIX=hybrid."
