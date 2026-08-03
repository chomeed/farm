#!/bin/bash
# Local-box variant of precompute_preference_pairs_robomimic_square_resized_peg_exaflops.sh.
# Builds the variant-6 "blend" preference-pair manifest against the local
# rewind base ckpt. Sbatch headers retained as comments so this can be sbatched
# on a local slurm partition too if needed.
#SBATCH --job-name=precompute_robomimic_square_resized_peg_pref_pairs
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

DRY_RUN=${DRY_RUN:-}
if [ -z "$DRY_RUN" ]; then
  source ~/.bashrc
  conda activate proj-rl-finetuning
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

set -euo pipefail

MANIFEST_ROOT=${MANIFEST_ROOT:-/home/chomeed/workspace/proj-rl-finetuning/manifests}

# ---- Hparams (threading varlen_blend parity) ------------------------------
NUM_EPISODES=200
CLIP_LEN=16
TOPK=3
SMOOTH_WINDOW=7
CLIPS_PER_EP=100
MATCH_DTW_WEIGHT=0.5
DTW_BAND=3
DTW_TIME_PENALTY_START=0.5
MATCH_WINDOW=16
MATCH_STATE_WEIGHT=0.4
SUCCESS_ANCHOR_MODE=random
DIRS=leaving

ROBOMIMIC_CAMERA_KEYS=(
  observation.images.agentview
  observation.images.eye_in_hand
)

# ---- Inputs / outputs -----------------------------------------------------
BASE_CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model
SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure
OUT_DIR=$MANIFEST_ROOT/robomimic_square_resized_peg
OUT="$OUT_DIR/preference_pairs_blend.json"

CMD=(python -m lerobot_policy_rewind_state_ours_variant6.common.precompute_preference_pairs
  --base_ckpt              "$BASE_CKPT"
  --success_repo_id        "$SUCCESS_REPO"
  --failure_repo_id        "$FAILURE_REPO"
  --camera_keys            "${ROBOMIMIC_CAMERA_KEYS[@]}"
  --num_episodes           "$NUM_EPISODES"
  --clip_len               "$CLIP_LEN"
  --smooth_window          "$SMOOTH_WINDOW"
  --success_clips_per_ep   "$CLIPS_PER_EP"
  --top_k                  "$TOPK"
  --success_anchor_mode    "$SUCCESS_ANCHOR_MODE"
  --match_window           "$MATCH_WINDOW"
  --match_state_weight     "$MATCH_STATE_WEIGHT"
  --match_dtw_weight       "$MATCH_DTW_WEIGHT"
  --dtw_band               "$DTW_BAND"
  --dtw_time_penalty_start "$DTW_TIME_PENALTY_START"
  --failure_clip_directions "$DIRS"
  --output_manifest        "$OUT"
)

if [ -n "$DRY_RUN" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

mkdir -p "$OUT_DIR"

if [ -f "$OUT" ]; then
  echo "[skip] manifest already exists at $OUT"
  exit 0
fi

echo "[$(date)] precompute robomimic_square_resized_peg / blend (dirs=$DIRS) -> $OUT"

"${CMD[@]}"

echo "[$(date)] done -> $OUT"
