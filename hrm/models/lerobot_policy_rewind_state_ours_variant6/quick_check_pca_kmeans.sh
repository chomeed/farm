#!/bin/bash
#SBATCH --job-name=pca_kmeans_check
#SBATCH --time=00:30:00
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Visual sanity check for the PCA + K-means failure-onset detector.
#
# Default: state + DINOv2 image embeddings (finetuned-pro6000 encoder).
# Set OBS_MODE=state to compare against the state-only baseline.
#
# Output PNG side-by-side: top row = rewind reward + smooth+argmax baseline,
# bottom row = PCA OOD score + threshold + first-cross critical_t.

SUCCESS_REPO=${SUCCESS_REPO:-chomeed/robomimic_square_resized_peg_success}
FAILURE_REPO=${FAILURE_REPO:-chomeed/robomimic_square_resized_peg_failure}
NUM_EPISODES=${NUM_EPISODES:-200}
NUM_VISUALIZE=${NUM_VISUALIZE:-8}
ALPHA=${ALPHA:-0.05}
OBS_MODE=${OBS_MODE:-state_image}
# state_image has a high-D feature vector (state + 2 * 768 DINOv2 dims); bump PCA.
DEFAULT_N_COMPONENTS=16
if [ "$OBS_MODE" = "state" ]; then
  DEFAULT_N_COMPONENTS=4
fi
N_COMPONENTS=${N_COMPONENTS:-$DEFAULT_N_COMPONENTS}
N_CLUSTERS=${N_CLUSTERS:-64}
SMOOTH_WINDOW=${SMOOTH_WINDOW:-7}

# Use the finetuned-pro6000 DINOv2 encoder for image embeddings.
BASE_CKPT=${BASE_CKPT:-/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model}
EVAL_DIR=${EVAL_DIR:-/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_variant6_robomimic_square_v1_eps200_clip100_longer/eval_val_last_hybrid}
OUT_DIR=${OUT_DIR:-/home/chomeed/workspace/insertion_generation/ckpts/rewind/pca_kmeans_test}

python -m lerobot_policy_rewind_state_ours_variant6.common.quick_check_pca_kmeans \
  --success_repo_id "$SUCCESS_REPO" \
  --failure_repo_id "$FAILURE_REPO" \
  --num_episodes    "$NUM_EPISODES" \
  --num_visualize   "$NUM_VISUALIZE" \
  --alpha           "$ALPHA" \
  --n_components    "$N_COMPONENTS" \
  --n_clusters      "$N_CLUSTERS" \
  --smooth_window   "$SMOOTH_WINDOW" \
  --obs_mode        "$OBS_MODE" \
  --base_ckpt       "$BASE_CKPT" \
  --eval_dir        "$EVAL_DIR" \
  --output_dir      "$OUT_DIR"

echo "Saved sanity-check artifacts under $OUT_DIR"
