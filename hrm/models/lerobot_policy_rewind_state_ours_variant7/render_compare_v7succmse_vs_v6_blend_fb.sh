#!/bin/bash
#
# Side-by-side (success | failure) reward comparison: variant7 succmse vs the
# two variant6 anchorstack runs (blend + forwardbackward) at the same step.
#
# Variant7 succmse only has a step 001000 checkpoint right now (training is
# still in flight). Defaults to V7_STEP=001000 / V6_STEP=002000 — a step
# mismatch, but produces a useful early read. Once variant7 hits 002000,
# rsync again and rerun with V7_STEP=002000.
#
# Per-run eval (trajectories.npz) is computed on demand if missing.
# Compare videos land in $OUT_ROOT/_compare_v7succmse_vs_v6_v7step${V7_STEP}_v6step${V6_STEP}_pairs/videos/

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

V7_STEP="${V7_STEP:-001000}"
V6_STEP="${V6_STEP:-002000}"
NUM_VAL_EPISODES=50
NUM_VISUALIZE=10
SAMPLING=hybrid
N_RECENT=8
V7_EVAL_SUBDIR="eval_step${V7_STEP}_hybrid${N_RECENT}"
V6_EVAL_SUBDIR="eval_step${V6_STEP}_hybrid${N_RECENT}"

CKPT_ROOT=${CKPT_ROOT:-/home/chomeed/workspace/proj-rl-finetuning/scratch2_outputs_chomeed/rewind}
# V7_ROOT and V6_ROOT can be overridden independently — e.g. when running as
# gwanwoo13 the v7 ckpt lives on /scratch2/gwanwoo13/... while the v6 eval
# npzs sit under chomeed's /home tree (readable, not writable).
V7_ROOT=${V7_ROOT:-$CKPT_ROOT}
V6_ROOT=${V6_ROOT:-$CKPT_ROOT}
# Where the v7 eval output + compare videos get written. Must be writable.
OUT_ROOT=${OUT_ROOT:-$V7_ROOT}

V7_RUN=rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.2_succmse
V6_BLEND_RUN=rewind_state_ours_variant6_dexmg_threading_v1_blend_mono_anchorstack_eps0.2
V6_FB_RUN=rewind_state_ours_variant6_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.2_0.0

NUM_PAIRS=10
FPS=20
PLOT_H=360

VIDEO_OUT="$OUT_ROOT/_compare_v7succmse_vs_v6_v7step${V7_STEP}_v6step${V6_STEP}_pairs"

if [ "$V7_STEP" != "$V6_STEP" ]; then
  echo "WARNING: V7_STEP=$V7_STEP != V6_STEP=$V6_STEP — comparing across mismatched training steps." >&2
fi

# ---------- 1. Eval variant7 if needed ----------
V7_CKPT="$V7_ROOT/$V7_RUN/checkpoints/$V7_STEP/pretrained_model"
V7_EVAL="$OUT_ROOT/$V7_RUN/$V7_EVAL_SUBDIR"
if [ ! -f "$V7_EVAL/trajectories.npz" ]; then
  if [ ! -d "$V7_CKPT" ]; then
    echo "ERROR: variant7 ckpt missing: $V7_CKPT" >&2
    echo "       (re-rsync from /scratch2/gwanwoo13/chomeed/outputs/rewind/$V7_RUN/ if training has advanced)" >&2
    exit 1
  fi
  echo "[$(date)] Eval variant7 succmse step $V7_STEP -> $V7_EVAL_SUBDIR"
  mkdir -p "$(dirname "$V7_EVAL")"
  python -m lerobot_policy_rewind_state_ours_variant7.common.eval_val_loss_and_visualize \
    --ckpt              "$V7_CKPT" \
    --success_repo_id   "$SUCCESS_REPO" \
    --failure_repo_id   "$FAILURE_REPO" \
    --camera_keys       "${DEXMG_CAMERAS[@]}" \
    --num_val_episodes  "$NUM_VAL_EPISODES" \
    --num_visualize     "$NUM_VISUALIZE" \
    --sampling          "$SAMPLING" \
    --n_recent          "$N_RECENT" \
    --output_dir        "$V7_EVAL"
fi

# ---------- 2. Sanity-check variant6 eval outputs ----------
V6_BLEND_EVAL="$V6_ROOT/$V6_BLEND_RUN/$V6_EVAL_SUBDIR"
V6_FB_EVAL="$V6_ROOT/$V6_FB_RUN/$V6_EVAL_SUBDIR"
for eval_dir in "$V6_BLEND_EVAL" "$V6_FB_EVAL"; do
  if [ ! -f "$eval_dir/trajectories.npz" ]; then
    echo "ERROR: missing $eval_dir/trajectories.npz" >&2
    echo "       run eval_val_loss_and_visualize on the corresponding variant6 ckpt first" >&2
    exit 1
  fi
done

# ---------- 3. Render compare ----------
echo "[$(date)] Rendering success|failure pair videos -> $VIDEO_OUT"
python -m lerobot_policy_rewind_state_ours_variant7.common.render_compare_videos_success_vs_failure \
  --success_repo_id "$SUCCESS_REPO" \
  --failure_repo_id "$FAILURE_REPO" \
  --num_pairs       "$NUM_PAIRS" \
  --fps             "$FPS" \
  --plot_height     "$PLOT_H" \
  --cameras         "$(IFS=,; echo "${DEXMG_CAMERAS[*]}")" \
  --output_dir      "$VIDEO_OUT" \
  --model "v7_succmse_blend_eps0.2_step${V7_STEP}:${V7_EVAL}" \
  --model "v6_blend_eps0.2_step${V6_STEP}:${V6_BLEND_EVAL}" \
  --model "v6_fb_eps0.2_0.0_step${V6_STEP}:${V6_FB_EVAL}"

echo "[$(date)] Done. Videos under $VIDEO_OUT/videos/"
