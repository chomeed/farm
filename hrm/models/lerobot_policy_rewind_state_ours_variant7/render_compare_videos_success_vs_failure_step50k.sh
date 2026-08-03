#!/bin/bash
#
# Step-050000 sibling of render_compare_videos_success_vs_failure_step2k.sh.
#
# Evals (if missing) the baseline + 4 anchorstack variants at step 050000, then
# renders ONE side-by-side (success | failure) mp4 per (success_ep, failure_ep)
# pair. Reward chart shows all 5 model curves overlaid: success = solid,
# failure = dashed, with two synced cursors.
#
# Output: $OUT_ROOT/_compare_5_anchorstack_step050000_hybrid8_pairs/videos/

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

STEP=050000
NUM_VAL_EPISODES=50
NUM_VISUALIZE=10
SAMPLING=hybrid
N_RECENT=8
EVAL_SUBDIR="eval_step${STEP}_hybrid${N_RECENT}"

CKPT_ROOT=${CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
OUT_ROOT=${OUT_ROOT:-$CKPT_ROOT}

# (label, run_dir) pairs — baseline first, then the 4 variants.
LABELS=(
  baseline
  blend_eps0.2
  blend_eps0.3
  fb_eps0.2_0.0
  fb_eps0.3_0.0
)
RUNS=(
  rewind_state_dexmg_threading_v1
  rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.2
  rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.3
  rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.2_0.0
  rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.3_0.0
)

NUM_PAIRS=10
FPS=20
PLOT_H=360

VIDEO_OUT="$OUT_ROOT/_compare_5_anchorstack_step${STEP}_hybrid${N_RECENT}_pairs"

# ---------- 1) Eval each model at step 50k if missing ----------
# variant7 eval module loads via PreTrainedConfig + lerobot policy factory, so
# it accepts the baseline checkpoint too.
for i in "${!RUNS[@]}"; do
  run="${RUNS[$i]}"
  ckpt="$CKPT_ROOT/$run/checkpoints/$STEP/pretrained_model"
  out="$OUT_ROOT/$run/$EVAL_SUBDIR"

  if [ -f "$out/trajectories.npz" ]; then
    echo "[$(date)] SKIP eval $run — $EVAL_SUBDIR already exists"
    continue
  fi
  if [ ! -d "$ckpt" ]; then
    echo "ERROR: $run missing step $STEP ckpt at $ckpt" >&2
    exit 1
  fi

  echo "[$(date)] Eval $run -> $EVAL_SUBDIR"
  mkdir -p "$(dirname "$out")"
  python -m lerobot_policy_rewind_state_ours_variant7.common.eval_val_loss_and_visualize \
    --ckpt              "$ckpt" \
    --success_repo_id   "$SUCCESS_REPO" \
    --failure_repo_id   "$FAILURE_REPO" \
    --camera_keys       "${DEXMG_CAMERAS[@]}" \
    --num_val_episodes  "$NUM_VAL_EPISODES" \
    --num_visualize     "$NUM_VISUALIZE" \
    --sampling          "$SAMPLING" \
    --n_recent          "$N_RECENT" \
    --output_dir        "$out"
done

# ---------- 2) Render pair videos ----------
MODEL_ARGS=()
for i in "${!RUNS[@]}"; do
  label="${LABELS[$i]}"
  eval_dir="$OUT_ROOT/${RUNS[$i]}/$EVAL_SUBDIR"
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
