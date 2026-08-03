#!/bin/bash
#SBATCH --job-name=compare_potential_rewind_v4
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

# OUT_DIR is relative to the submit directory; set an absolute /scratch2/... path if preferred
OUT_DIR=./compare_rewind_vs_variant4_potential

python -m lerobot_policy_rewind_state_ours_variant4.common.compare_rewind_vs_variant4_potential \
  --num_episodes 10 \
  --output_dir "$OUT_DIR"

echo "Done. Output: $OUT_DIR"
