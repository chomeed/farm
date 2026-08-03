#!/bin/bash
#SBATCH --job-name=rewind_variant7_eval_4_5k
#SBATCH --time=04:00:00
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

# Evaluate the step-002000 checkpoints of the 4 anchorstack runs on the
# dexmg threading SUCCESS and FAILURE datasets (held-out episodes). Each
# eval writes:
#   <run>/eval_step002000_hybrid8/
#     trajectories.npz        — per-episode per-frame predicted rewards
#     failure_reward.png      — N_VISUALIZE failure trajectories (full episode)
#     success_reward.png      — N_VISUALIZE success trajectories
#
# After this finishes, render_compare_videos.sh can stitch side-by-side
# videos using the four trajectories.npz files.

# Env activation that works under both interactive bash AND sbatch
# --export=NONE (conda is NOT on PATH in that case, so we bypass it).
# `ENV_PREFIX` must point at a conda env's root dir (the one with bin/python).
# Default = chomeed's local env discovered via $CONDA_PREFIX or known path.
#   Local (chomeed):   bash <script.sh>                       → uses already-active env
#   sbatch (gwanwoo):  ENV_PREFIX=/lustre/.../resfit_td3_gwanwoo13 sbatch ...
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-/home/chomeed/miniconda3/envs/proj-rl-finetuning}}"
[ -x "$ENV_PREFIX/bin/python" ] || {
  echo "ERROR: env not found at $ENV_PREFIX (set ENV_PREFIX to a valid conda env root)" >&2
  exit 1
}
export PATH="$ENV_PREFIX/bin:${PATH:-/usr/bin:/bin}"
export CONDA_PREFIX="$ENV_PREFIX"
export CONDA_DEFAULT_ENV="$(basename "$ENV_PREFIX")"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

set -euo pipefail

# ---------- Config ----------
SUCCESS_REPO=chomeed/dexmg_two_arm_threading
FAILURE_REPO=chomeed/dexmg_two_arm_threading_failures

DEXMG_CAMERAS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

STEP=005000
NUM_VAL_EPISODES=50
NUM_VISUALIZE=10              # per source (success + failure)
SAMPLING=hybrid
N_RECENT=8

EVAL_SUBDIR="eval_step${STEP}_hybrid${N_RECENT}"

# CKPT_ROOT: where the trained checkpoints live (READ-only when run from a
#            different account that doesn't own this tree).
# OUT_ROOT:  where eval outputs + comparison videos get written. Defaults to
#            CKPT_ROOT for the original owner; override when running as a
#            different account (e.g. OUT_ROOT=/scratch2/gwanwoo13/outputs/rewind).
CKPT_ROOT=${CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
OUT_ROOT=${OUT_ROOT:-$CKPT_ROOT}

RUNS=(
  rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.2
  rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.3
  rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.2_0.0
  rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.3_0.0
)

for run in "${RUNS[@]}"; do
  CKPT="$CKPT_ROOT/$run/checkpoints/$STEP/pretrained_model"
  OUT="$OUT_ROOT/$run/$EVAL_SUBDIR"
  mkdir -p "$(dirname "$OUT")"

  if [ ! -d "$CKPT" ]; then
    echo "[$(date)] SKIP $run — no step $STEP checkpoint"
    continue
  fi
  if [ -d "$OUT" ]; then
    echo "[$(date)] SKIP $run — $EVAL_SUBDIR already exists"
    continue
  fi

  echo "[$(date)] Eval $run -> $EVAL_SUBDIR"
  python -m lerobot_policy_rewind_state_ours_variant7.common.eval_val_loss_and_visualize \
    --ckpt              "$CKPT" \
    --success_repo_id   "$SUCCESS_REPO" \
    --failure_repo_id   "$FAILURE_REPO" \
    --camera_keys       "${DEXMG_CAMERAS[@]}" \
    --num_val_episodes  "$NUM_VAL_EPISODES" \
    --num_visualize     "$NUM_VISUALIZE" \
    --sampling          "$SAMPLING" \
    --n_recent          "$N_RECENT" \
    --output_dir        "$OUT"
done

echo "[$(date)] All evals done."
for run in "${RUNS[@]}"; do
  echo "  $OUT_ROOT/$run/$EVAL_SUBDIR"
done

# ---------- Render comparison videos ----------
# Side-by-side mp4: 3 camera strips on the left, the 4 reward curves on the
# right with a vertical cursor synced to the frame. One mp4 per episode per
# split.
VIDEO_OUT="$OUT_ROOT/_compare_4_anchorstack_step${STEP}_hybrid${N_RECENT}"
NUM_RENDER_EPISODES=10
FPS=20
PLOT_H=360

MODEL_ARGS=()
for run in "${RUNS[@]}"; do
  label="${run##*anchorstack_}"     # e.g. "eps0.2" or "eps0.3_0.0"
  case "$run" in
    *blend*)        label="blend_${label}";;
    *forwardback*)  label="fb_${label}";;
  esac
  MODEL_ARGS+=(--model "${label}:$OUT_ROOT/$run/$EVAL_SUBDIR")
done

echo "[$(date)] Rendering side-by-side videos -> $VIDEO_OUT"
python -m lerobot_policy_rewind_state_ours_variant7.common.render_compare_videos \
  --success_repo_id "$SUCCESS_REPO" \
  --failure_repo_id "$FAILURE_REPO" \
  --splits          "success,failure" \
  --num_episodes    "$NUM_RENDER_EPISODES" \
  --fps             "$FPS" \
  --plot_height     "$PLOT_H" \
  --cameras         "$(IFS=,; echo "${DEXMG_CAMERAS[*]}")" \
  --output_dir      "$VIDEO_OUT" \
  "${MODEL_ARGS[@]}"

echo "[$(date)] Done. Videos under $VIDEO_OUT/videos/{success,failure}/"
