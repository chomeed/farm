#!/bin/bash
#SBATCH --job-name=potential_shaping_viz
#SBATCH --time=01:00:00
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
set -euo pipefail

CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_variant4_robomimic_square_v1_eps200_clip100_longer/checkpoints/050000/pretrained_model
# OUT_DIR is relative to the submit directory; set an absolute /scratch2/... path if preferred
OUT_DIR=./potential_shaping_overlay

python -m lerobot_policy_rewind_state_ours_variant4.common.visualize_potential_shaping \
  --ckpt "$CKPT" \
  --num_episodes 10 \
  --output_dir "$OUT_DIR"

echo "Done. Output: $OUT_DIR"
