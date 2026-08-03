#!/bin/bash
#SBATCH --job-name=precompute_robomimic_square_resized_peg_pref_pairs
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=precompute_robomimic_square_resized_peg_pref_pairs_%j.out

# EXAFLOPS-ACCOUNT precompute of the variant-6 "blend" (forward-only `leaving`)
# preference-pair manifest for the modified robomimic_square (resized peg +
# narrow init). Hparams match the threading-blend recipe in
# hrm/finetuning/lerobot_policy_resfit_td3/dexmg/precompute_preference_pairs_dexmg_exaflops.sh
# (cosine + banded DTW blend, success_anchor_mode=random).
#
# Submit:
#   sbatch hrm/models/lerobot_policy_rewind_state_ours_variant6/precompute_preference_pairs_robomimic_square_resized_peg_exaflops.sh
# Dry-run (no conda, no python):
#   DRY_RUN=1 bash .../precompute_preference_pairs_robomimic_square_resized_peg_exaflops.sh

DRY_RUN=${DRY_RUN:-}
if [ -z "$DRY_RUN" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /lustre/chomeed/exaflops/envs/resfit_td3_exaflops
  [ "${CONDA_PREFIX:-}" = /lustre/chomeed/exaflops/envs/resfit_td3_exaflops ] || {
    echo "ERROR: failed to activate resfit_td3_exaflops env" >&2; exit 1; }
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/lustre/chomeed/exaflops/data}
MANIFEST_ROOT=${MANIFEST_ROOT:-/lustre/chomeed/exaflops/manifests}
export HF_HOME=${HF_HOME:-/lustre/chomeed/exaflops/hf_cache}
mkdir -p "$HF_HOME"

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
BASE_CKPT=$DATA_ROOT/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model
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
