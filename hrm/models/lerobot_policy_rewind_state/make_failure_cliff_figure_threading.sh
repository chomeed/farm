#!/bin/bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-/home/chomeed/miniconda3/envs/proj-rl-finetuning}}"
[ -x "$ENV_PREFIX/bin/python" ] || {
  echo "ERROR: env not found at $ENV_PREFIX (set ENV_PREFIX to a valid conda env root)" >&2
  exit 1
}
export PATH="$ENV_PREFIX/bin:${PATH:-/usr/bin:/bin}"
export CONDA_PREFIX="$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Threading sibling of make_failure_cliff_figure.sh. Reuses existing eval dirs
# under /scratch2/chomeed/outputs/rewind/{rewind_state_dexmg_threading_v1,
# rewind_state_ours_variant6_dexmg_threading_v1_blend_mono_anchorstack_eps0.3}.
#
# Browse example pairs at
#   /scratch2/chomeed/outputs/rewind/_compare_4_anchorstack_step002000_hybrid8_pairs/videos/
# then re-run with SUCCESS_EP=… FAILURE_EP=… to make the figure.

FAILURE_EP=${FAILURE_EP:-150}
SMOOTH_WINDOW=${SMOOTH_WINDOW:-5}
FRAME_T=${FRAME_T:--1}        # -1 = auto-detect divergence point
CAMERA_KEY=${CAMERA_KEY:-observation.images.agentview}
FAILURE_REPO=${FAILURE_REPO:-chomeed/dexmg_two_arm_threading_failures}

CKPT_ROOT=${CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
BASE_EVAL_DIR=${BASE_EVAL_DIR:-$CKPT_ROOT/rewind_state_dexmg_threading_v1/eval_step050000_hybrid8}
OURS_EVAL_DIR=${OURS_EVAL_DIR:-$CKPT_ROOT/rewind_state_ours_variant6_dexmg_threading_v1_blend_mono_anchorstack_eps0.3/eval_step002000_hybrid8}

OUT_PNG=${OUT_PNG:-$CKPT_ROOT/_compare_threading_base050000_vs_ours002000_hybrid8_pairs/figures/failure_f${FAILURE_EP}_divergence.png}

# Failure-only layout: omit --success-ep / --success-repo-id.
python -m lerobot_policy_rewind_state.common.make_failure_cliff_figure \
  --base-eval-dir   "$BASE_EVAL_DIR" \
  --ours-eval-dir   "$OURS_EVAL_DIR" \
  --failure-repo-id "$FAILURE_REPO" \
  --failure-ep      "$FAILURE_EP" \
  --camera-key      "$CAMERA_KEY" \
  --smooth-window   "$SMOOTH_WINDOW" \
  --frame-t         "$FRAME_T" \
  --output-png      "$OUT_PNG"
