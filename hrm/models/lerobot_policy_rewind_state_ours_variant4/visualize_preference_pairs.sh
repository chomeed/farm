#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant4_viz_pref_pairs
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

MANIFEST=/scratch2/chomeed/manifests/rewind_state_ours_variant4/preference_pairs.json
OUT_DIR=/scratch2/chomeed/outputs/rewind/rewind_state_ours_variant4_robomimic_square_v1_eps200_clip100/pref_pairs_viz

python -m lerobot_policy_rewind_state_ours_variant4.common.visualize_preference_pairs \
  --manifest "$MANIFEST" \
  --n_pairs 8 \
  --seed 0 \
  --fps 10 \
  --output_dir "$OUT_DIR"

echo "Done. Output: $OUT_DIR"
