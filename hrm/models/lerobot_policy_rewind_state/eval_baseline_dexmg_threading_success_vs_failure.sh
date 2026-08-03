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

# Threading sibling of eval_baseline_dexmg_three_piece_success_vs_failure_step2k.sh.
# Evals ReWiND baseline @ last (50k) and variant6 anchorstack-blend eps0.3 @ 2k
# on success+failure demos, then renders side-by-side success|failure pair
# videos. Skips eval steps whose trajectories.npz already exists.

# Eval knobs
BASE_STEP=050000
OURS_STEP=002000
SAMPLING=hybrid
N_RECENT=8
NUM_VAL_EPISODES=50
NUM_VISUALIZE=10

# Figure-quality render knobs
NUM_PAIRS=50
FPS=20
PLOT_H=540
FIGURE_CAMERAS=(observation.images.agentview)

# Bare ReWiND baseline (last ckpt)
BASE_CKPT_ROOT=${BASE_CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
BASE_RUN=rewind_state_dexmg_threading_v1
BASE_CKPT="$BASE_CKPT_ROOT/$BASE_RUN/checkpoints/$BASE_STEP/pretrained_model"
BASE_OUT="$BASE_CKPT_ROOT/$BASE_RUN/eval_step${BASE_STEP}_hybrid${N_RECENT}"

# Variant6 anchorstack-blend eps0.3 ("ours") @ early stop
OURS_REWARD_CKPT_ROOT=${OURS_REWARD_CKPT_ROOT:-$BASE_CKPT_ROOT}
OURS_RUN=rewind_state_ours_variant6_dexmg_threading_v1_blend_mono_anchorstack_eps0.3
OURS_CKPT="$OURS_REWARD_CKPT_ROOT/$OURS_RUN/checkpoints/$OURS_STEP/pretrained_model"
OURS_OUT="$OURS_REWARD_CKPT_ROOT/$OURS_RUN/eval_step${OURS_STEP}_hybrid${N_RECENT}"

VIDEO_OUT="$BASE_CKPT_ROOT/_compare_threading_base${BASE_STEP}_vs_ours${OURS_STEP}_hybrid${N_RECENT}_pairs"

SUCCESS_REPO=chomeed/dexmg_two_arm_threading
FAILURE_REPO=chomeed/dexmg_two_arm_threading_failures

DEXMG_CAMERAS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

run_eval() {
  local label="$1" ckpt="$2" out="$3"
  if [ ! -d "$ckpt" ]; then
    echo "ERROR: $label ckpt missing: $ckpt" >&2
    exit 1
  fi
  if [ -f "$out/trajectories.npz" ]; then
    echo "[$(date)] Skip eval ($label) — trajectories.npz exists at $out"
    return
  fi
  echo "[$(date)] Eval ($label) $ckpt -> $out"
  mkdir -p "$(dirname "$out")"
  python -m lerobot_policy_rewind_state_ours_variant6.common.eval_val_loss_and_visualize \
    --ckpt              "$ckpt" \
    --success_repo_id   "$SUCCESS_REPO" \
    --failure_repo_id   "$FAILURE_REPO" \
    --camera_keys       "${DEXMG_CAMERAS[@]}" \
    --num_val_episodes  "$NUM_VAL_EPISODES" \
    --num_visualize     "$NUM_VISUALIZE" \
    --sampling          "$SAMPLING" \
    --n_recent          "$N_RECENT" \
    --output_dir        "$out"
}

run_eval "ReWiND"        "$BASE_CKPT" "$BASE_OUT"
run_eval "ours_variant6" "$OURS_CKPT" "$OURS_OUT"

for d in "$BASE_OUT" "$OURS_OUT"; do
  if [ ! -f "$d/trajectories.npz" ]; then
    echo "ERROR: missing $d/trajectories.npz" >&2
    exit 1
  fi
done

echo "[$(date)] Rendering ReWiND vs ours success|failure pair videos -> $VIDEO_OUT"
python -m lerobot_policy_rewind_state_ours_variant6.common.render_compare_videos_success_vs_failure \
  --model           "ReWiND:$BASE_OUT" \
  --model           "ours:$OURS_OUT" \
  --success_repo_id "$SUCCESS_REPO" \
  --failure_repo_id "$FAILURE_REPO" \
  --num_pairs       "$NUM_PAIRS" \
  --fps             "$FPS" \
  --plot_height     "$PLOT_H" \
  --cameras         "$(IFS=,; echo "${FIGURE_CAMERAS[*]}")" \
  --output_dir      "$VIDEO_OUT"

echo "[$(date)] Done."
echo "  ReWiND eval: $BASE_OUT"
echo "  ours eval:   $OURS_OUT"
echo "  videos:      $VIDEO_OUT/videos/"
