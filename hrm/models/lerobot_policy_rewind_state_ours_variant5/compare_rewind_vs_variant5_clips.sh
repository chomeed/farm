#!/bin/bash
#SBATCH --job-name=rewind_vs_variant5_clips
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

MANIFEST=/scratch2/chomeed/manifests/rewind_state_ours_variant5/preference_pairs.json
VARIANT3_CKPT=/scratch2/chomeed/outputs/rewind/rewind_state_ours_variant5_robomimic_square_v1_eps200_clip100/checkpoints/last/pretrained_model
OUT_DIR=./base_vs_variant5_on_clips

# --base_ckpt is omitted: the script defaults it to the manifest's base_ckpt.
python -m lerobot_policy_rewind_state_ours_variant5.common.compare_rewind_vs_variant5_clips \
  --manifest "$MANIFEST" \
  --variant5_ckpt "$VARIANT3_CKPT" \
  --output_dir "$OUT_DIR"

echo "Done. Output: $OUT_DIR"
