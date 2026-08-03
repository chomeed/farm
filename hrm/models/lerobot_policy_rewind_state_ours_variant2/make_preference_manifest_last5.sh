#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant2_pref_manifest_last5
#SBATCH --time=02:00:00
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
set -euo pipefail

cd /home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state_ours_variant2/src/lerobot_policy_rewind_state_ours_variant2

# Same base_ckpt + settings as the existing manifest
# (/scratch2/chomeed/manifests/rewind_state_ours_variant2/preference_pairs.json)
# so the ONLY difference is --match_window. This makes the two manifests
# directly comparable.
BASE_CKPT=/scratch2/chomeed/outputs/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model
SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure

OUT_MANIFEST=/scratch2/chomeed/manifests/rewind_state_ours_variant2/preference_pairs_match_last5.json
mkdir -p "$(dirname "$OUT_MANIFEST")"

python -m lerobot_policy_rewind_state_ours_variant2.common.precompute_preference_pairs \
  --base_ckpt          "$BASE_CKPT" \
  --success_repo_id    "$SUCCESS_REPO" \
  --failure_repo_id    "$FAILURE_REPO" \
  --num_episodes       200 \
  --clip_len           16 \
  --frame_stride       1 \
  --smooth_window      7 \
  --success_clips_per_ep 100 \
  --top_k              3 \
  --success_anchor_mode critical \
  --success_critical_threshold 0.75 \
  --match_window       5 \
  --output_manifest    "$OUT_MANIFEST"

echo "Done. Output: $OUT_MANIFEST"
