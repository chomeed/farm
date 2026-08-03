#!/bin/bash
#SBATCH --job-name=rewind_state_ours_bad_vs_good
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

CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_robomimic_square_v1_eps50_clip20/checkpoints/010000/pretrained_model

# The κ=0.10 ours tagged dataset has more bad frames to choose from, so picking
# from it gives more interesting mid-trajectory cases.
TAGGED_REPO=chomeed/robomimic_square_resized_peg_success_rabc_delta8_kappa0.1_using_rewind_ours

OUT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_robomimic_square_v1_eps50_clip20/bad_vs_good_clip.mp4

python -m lerobot_policy_rewind_state_ours_variant5.common.make_good_vs_bad_clip_pair \
  --ckpt "$CKPT" \
  --tagged_repo_id "$TAGGED_REPO" \
  --clip_len 16 \
  --frame_stride 1 \
  --bad_rel_pos_lo 0.40 \
  --bad_rel_pos_hi 0.60 \
  --n_good_candidates 100 \
  --bad_seed 0 \
  --good_seed 0 \
  --fps 10 \
  --output_mp4 "$OUT"

echo "Done. Output: $OUT"
