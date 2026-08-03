#!/bin/bash
#
# Render side-by-side (success | failure) videos for the 4 anchorstack runs at
# step 002000, with reward curves for all 4 models overlaid on a shared chart
# (success = solid, failure = dashed) and two synced cursors.
#
# Assumes the per-run `eval_step002000_hybrid8/trajectories.npz` files already
# exist (produced by eval_4_anchorstack_dexmg_threading_step2k.sh).
#
# Output: $OUT_ROOT/_compare_4_anchorstack_step002000_hybrid8_pairs/videos/ep_sXXXXX_fYYYYY.mp4

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-/home/chomeed/miniconda3/envs/proj-rl-finetuning}}"
[ -x "$ENV_PREFIX/bin/python" ] || {
  echo "ERROR: env not found at $ENV_PREFIX (set ENV_PREFIX to a valid conda env root)" >&2
  exit 1
}
export PATH="$ENV_PREFIX/bin:${PATH:-/usr/bin:/bin}"
export CONDA_PREFIX="$ENV_PREFIX"
export CONDA_DEFAULT_ENV="$(basename "$ENV_PREFIX")"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ---------- Config ----------
SUCCESS_REPO=chomeed/dexmg_two_arm_threading
FAILURE_REPO=chomeed/dexmg_two_arm_threading_failures

DEXMG_CAMERAS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

STEP=002000
NUM_VAL_EPISODES=50
NUM_VISUALIZE=10
SAMPLING=hybrid
N_RECENT=8
EVAL_SUBDIR="eval_step${STEP}_hybrid${N_RECENT}"

CKPT_ROOT=${CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
OUT_ROOT=${OUT_ROOT:-$CKPT_ROOT}

# Baseline rewind model — same dexmg threading task, no "ours" variants.
BASE_RUN=rewind_state_dexmg_threading_v1

RUNS=(
  rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.2
  rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.3
  rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.2_0.0
  rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.3_0.0
)

NUM_PAIRS=10
FPS=20
PLOT_H=360

VIDEO_OUT="$OUT_ROOT/_compare_5_anchorstack_step${STEP}_hybrid${N_RECENT}_pairs"

# Ensure the baseline has a trajectories.npz at this step. The variant7
# eval module is architecture-agnostic (loads via PreTrainedConfig +
# lerobot policy factory), so it accepts the baseline checkpoint too.
BASE_CKPT="$CKPT_ROOT/$BASE_RUN/checkpoints/$STEP/pretrained_model"
BASE_EVAL="$OUT_ROOT/$BASE_RUN/$EVAL_SUBDIR"
if [ ! -f "$BASE_EVAL/trajectories.npz" ]; then
  if [ ! -d "$BASE_CKPT" ]; then
    echo "ERROR: baseline ckpt missing: $BASE_CKPT" >&2
    exit 1
  fi
  echo "[$(date)] Eval baseline $BASE_RUN -> $EVAL_SUBDIR"
  mkdir -p "$(dirname "$BASE_EVAL")"
  python -m lerobot_policy_rewind_state_ours_variant7.common.eval_val_loss_and_visualize \
    --ckpt              "$BASE_CKPT" \
    --success_repo_id   "$SUCCESS_REPO" \
    --failure_repo_id   "$FAILURE_REPO" \
    --camera_keys       "${DEXMG_CAMERAS[@]}" \
    --num_val_episodes  "$NUM_VAL_EPISODES" \
    --num_visualize     "$NUM_VISUALIZE" \
    --sampling          "$SAMPLING" \
    --n_recent          "$N_RECENT" \
    --output_dir        "$BASE_EVAL"
fi

MODEL_ARGS=(--model "baseline:${BASE_EVAL}")
for run in "${RUNS[@]}"; do
  label="${run##*anchorstack_}"
  case "$run" in
    *blend*)        label="blend_${label}";;
    *forwardback*)  label="fb_${label}";;
  esac
  eval_dir="$OUT_ROOT/$run/$EVAL_SUBDIR"
  if [ ! -f "$eval_dir/trajectories.npz" ]; then
    echo "ERROR: missing $eval_dir/trajectories.npz — run eval_4_anchorstack_dexmg_threading_step2k.sh first" >&2
    exit 1
  fi
  MODEL_ARGS+=(--model "${label}:${eval_dir}")
done

echo "[$(date)] Rendering success|failure pair videos -> $VIDEO_OUT"
python -m lerobot_policy_rewind_state_ours_variant7.common.render_compare_videos_success_vs_failure \
  --success_repo_id "$SUCCESS_REPO" \
  --failure_repo_id "$FAILURE_REPO" \
  --num_pairs       "$NUM_PAIRS" \
  --fps             "$FPS" \
  --plot_height     "$PLOT_H" \
  --cameras         "$(IFS=,; echo "${DEXMG_CAMERAS[*]}")" \
  --output_dir      "$VIDEO_OUT" \
  "${MODEL_ARGS[@]}"

echo "[$(date)] Done. Videos under $VIDEO_OUT/videos/"
