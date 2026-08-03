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

# Reuses the eval dirs produced by
#   eval_baseline_dexmg_three_piece_success_vs_failure_step2k.sh
# Defaults match what that script writes; override with env vars to retarget.

SUCCESS_EP=${SUCCESS_EP:-970}
FAILURE_EP=${FAILURE_EP:-164}
SMOOTH_WINDOW=${SMOOTH_WINDOW:-5}
FRAME_T=${FRAME_T:--1}        # -1 = auto-detect divergence point
CAMERA_KEY=${CAMERA_KEY:-observation.images.agentview}
SUCCESS_REPO=${SUCCESS_REPO:-chomeed/dexmg_two_arm_three_piece_assembly}
FAILURE_REPO=${FAILURE_REPO:-chomeed/dexmg_two_arm_three_piece_assembly_failures}

CKPT_ROOT=${CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
BASE_EVAL_DIR=${BASE_EVAL_DIR:-$CKPT_ROOT/rewind_state_dexmg_three_piece_v1/eval_step050000_hybrid8}
OURS_EVAL_DIR=${OURS_EVAL_DIR:-$CKPT_ROOT/rewind_state_ours_variant6_dexmg_three_piece_v1_blend_mono_anchorstack_eps0.3/eval_step002000_hybrid8}

OUT_PNG=${OUT_PNG:-$CKPT_ROOT/_compare_three_piece_base050000_vs_ours002000_hybrid8_pairs/figures/pair_s${SUCCESS_EP}_f${FAILURE_EP}_divergence.png}

python -m lerobot_policy_rewind_state.common.make_failure_cliff_figure \
  --base-eval-dir   "$BASE_EVAL_DIR" \
  --ours-eval-dir   "$OURS_EVAL_DIR" \
  --success-repo-id "$SUCCESS_REPO" \
  --failure-repo-id "$FAILURE_REPO" \
  --success-ep      "$SUCCESS_EP" \
  --failure-ep      "$FAILURE_EP" \
  --camera-key      "$CAMERA_KEY" \
  --smooth-window   "$SMOOTH_WINDOW" \
  --frame-t         "$FRAME_T" \
  --output-png      "$OUT_PNG"
