#!/bin/bash
#SBATCH --job-name=rewind_state_ours_eval
#SBATCH --time=03:00:00
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# ---------- Config (shared across both runs) ----------
SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure

NUM_VAL_EPISODES=50
NUM_VISUALIZE=5
CLIP_LEN=16
CLIP_STRIDE=4
PREF_BATCH_SIZE=16

# ---------- 1. Eval our 50-eps / 20-clips-per-ep model ----------
OURS_RUN_DIR=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_robomimic_square_v1_eps50_clip20
OURS_CKPT="$OURS_RUN_DIR/checkpoints/010000/pretrained_model"
OURS_OUT="$OURS_RUN_DIR/eval_val_last50"

echo "[$(date)] Eval (ours) $OURS_CKPT -> $OURS_OUT"
python -m lerobot_policy_rewind_state_ours_variant5.common.eval_val_loss_and_visualize \
  --ckpt              "$OURS_CKPT" \
  --success_repo_id   "$SUCCESS_REPO" \
  --failure_repo_id   "$FAILURE_REPO" \
  --num_val_episodes  "$NUM_VAL_EPISODES" \
  --num_visualize     "$NUM_VISUALIZE" \
  --clip_len          "$CLIP_LEN" \
  --clip_stride       "$CLIP_STRIDE" \
  --pref_batch_size   "$PREF_BATCH_SIZE" \
  --output_dir        "$OURS_OUT"

# ---------- 2. Eval the baseline reward-only model ----------
# Same architecture root (rewind_state) but trained without the BT auxiliary
# objective. Our eval script skips pref scoring automatically because the
# baseline policy has no `predict_preference_score` method.
BASE_RUN_DIR=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000
BASE_CKPT="$BASE_RUN_DIR/checkpoints/010000/pretrained_model"
BASE_OUT="$BASE_RUN_DIR/eval_val_last50"

echo "[$(date)] Eval (baseline) $BASE_CKPT -> $BASE_OUT"
python -m lerobot_policy_rewind_state_ours_variant5.common.eval_val_loss_and_visualize \
  --ckpt              "$BASE_CKPT" \
  --success_repo_id   "$SUCCESS_REPO" \
  --failure_repo_id   "$FAILURE_REPO" \
  --num_val_episodes  "$NUM_VAL_EPISODES" \
  --num_visualize     "$NUM_VISUALIZE" \
  --clip_len          "$CLIP_LEN" \
  --clip_stride       "$CLIP_STRIDE" \
  --pref_batch_size   "$PREF_BATCH_SIZE" \
  --output_dir        "$BASE_OUT"

echo "[$(date)] Both evals done."
echo "  ours    -> $OURS_OUT"
echo "  baseline-> $BASE_OUT"
