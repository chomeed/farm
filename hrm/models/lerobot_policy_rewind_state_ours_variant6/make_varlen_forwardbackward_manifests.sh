#!/bin/bash
#SBATCH --job-name=varlen_fb_manifests
#SBATCH --time=06:00:00
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%A_%a.out
#SBATCH --array=0-3

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

# Regenerate the four var-len preference manifests with the new
# "approaching + leaving" failure clip behaviour. Slurm array: each array
# task runs exactly one variant on its own GPU, so the four variants run in
# parallel. Submit once:
#
#   sbatch hrm/models/lerobot_policy_rewind_state_ours_variant6/make_varlen_forwardbackward_manifests.sh
#
# To run a single variant manually (no slurm), set SLURM_ARRAY_TASK_ID first:
#
#   SLURM_ARRAY_TASK_ID=2 bash .../make_varlen_forwardbackward_manifests.sh

OUT_DIR=/home/chomeed/workspace/proj-rl-finetuning/manifests/var_len_forwardbackward
mkdir -p "$OUT_DIR"

BASE_CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model
SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure

NUM_EPISODES=200
DIRECTIONS="leaving,approaching"

# Variant table: (name, extra args separated by '|'). Indexed by SLURM_ARRAY_TASK_ID.
VARIANT_NAMES=(
  varlen_cosine_critanchor
  varlen_blend
  varlen_cosine
  varlen_dtw_only_strict
)
VARIANT_ARGS=(
  "--success_clips_per_ep 1   --match_dtw_weight 0.0 --match_window 16 --match_state_weight 0.4 --success_anchor_mode critical"
  "--success_clips_per_ep 100 --match_dtw_weight 0.5 --match_window 16 --match_state_weight 0.4 --dtw_band 3 --dtw_time_penalty_start 0.5"
  "--success_clips_per_ep 100 --match_dtw_weight 0.0 --match_window 16 --match_state_weight 0.4"
  "--success_clips_per_ep 100 --match_dtw_weight 1.0 --dtw_band 2 --dtw_time_penalty_start 1.5"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-}
if [ -z "$TASK_ID" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is unset. Submit via sbatch (which sets it"
  echo "       for each array task), or set it manually 0..3 to run one variant:"
  echo "         SLURM_ARRAY_TASK_ID=2 bash $0"
  exit 1
fi
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#VARIANT_NAMES[@]}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID out of range 0..$((${#VARIANT_NAMES[@]} - 1))"
  exit 1
fi

NAME="${VARIANT_NAMES[$TASK_ID]}"
EXTRA_ARGS="${VARIANT_ARGS[$TASK_ID]}"
OUT="$OUT_DIR/preference_pairs_$NAME.json"

if [ -f "$OUT" ]; then
  echo "[skip] $NAME - manifest already exists at $OUT"
  exit 0
fi

echo "[$(date)] task $TASK_ID: $NAME -> $OUT"
# shellcheck disable=SC2086
python -m lerobot_policy_rewind_state_ours_variant6.common.precompute_preference_pairs \
  --base_ckpt                  "$BASE_CKPT" \
  --success_repo_id            "$SUCCESS_REPO" \
  --failure_repo_id            "$FAILURE_REPO" \
  --output_manifest            "$OUT" \
  --num_episodes               "$NUM_EPISODES" \
  --failure_clip_directions    "$DIRECTIONS" \
  $EXTRA_ARGS

echo "[$(date)] task $TASK_ID: $NAME done -> $OUT"
