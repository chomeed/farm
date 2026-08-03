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

# Eval bare-rewind baseline vs variant6 anchorstack-blend eps0.3 on dexmg
# two-arm three-piece-assembly success+failure demos at step 002000, then
# render side-by-side success|failure videos with both models' reward curves
# overlaid (solid = success, dashed = failure). Uses the variant6 eval+render
# modules (architecture-agnostic: load via PreTrainedConfig + lerobot policy
# factory, so the bare rewind_state ckpt loads fine through them).

# Baseline trains to convergence (50k steps, last == 050000). Variant6
# anchorstack-blend is intentionally early-stopped at 2k because its BT
# auxiliary signal peaks there. Use each model's best ckpt for a fair
# comparison rather than the same step.
BASE_STEP=050000
OURS_STEP=002000
SAMPLING=hybrid
N_RECENT=8
NUM_VAL_EPISODES=50
NUM_VISUALIZE=10

# Figure-quality render knobs:
#  - single agentview camera (cleanest, no in-hand chaos)
#  - 540px panel for crisp reward curves at paper-figure scale
#  - render the full pool (cap = NUM_VAL_EPISODES) so we can browse all pairs
NUM_PAIRS=50
FPS=20
PLOT_H=540
FIGURE_CAMERAS=(observation.images.agentview)

# Bare rewind baseline (chomeed-trained, on /scratch2) — last ckpt (50k).
BASE_CKPT_ROOT=${BASE_CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
BASE_RUN=rewind_state_dexmg_three_piece_v1
BASE_CKPT="$BASE_CKPT_ROOT/$BASE_RUN/checkpoints/$BASE_STEP/pretrained_model"
BASE_OUT="$BASE_CKPT_ROOT/$BASE_RUN/eval_step${BASE_STEP}_hybrid${N_RECENT}"

# Variant6 anchorstack-blend eps0.3 (gwanwoo13-trained, on /lustre,
# world-readable) — early-stopped ckpt (2k). Eval outputs land alongside
# chomeed's baseline so we don't try to write into gwanwoo13's scratch.
OURS_REWARD_CKPT_ROOT=${OURS_REWARD_CKPT_ROOT:-/lustre/chomeed/gwanwoo13/scratch2_outputs/rewind}
OURS_RUN=rewind_state_ours_variant6_dexmg_three_piece_v1_blend_mono_anchorstack_eps0.3
OURS_CKPT="$OURS_REWARD_CKPT_ROOT/$OURS_RUN/checkpoints/$OURS_STEP/pretrained_model"
OURS_OUT_ROOT=${OURS_OUT_ROOT:-$BASE_CKPT_ROOT}
OURS_OUT="$OURS_OUT_ROOT/$OURS_RUN/eval_step${OURS_STEP}_hybrid${N_RECENT}"

VIDEO_OUT="$BASE_CKPT_ROOT/_compare_three_piece_base${BASE_STEP}_vs_ours${OURS_STEP}_hybrid${N_RECENT}_pairs"

SUCCESS_REPO=chomeed/dexmg_two_arm_three_piece_assembly
FAILURE_REPO=chomeed/dexmg_two_arm_three_piece_assembly_failures

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

run_eval "bare"          "$BASE_CKPT" "$BASE_OUT"
run_eval "ours_variant6" "$OURS_CKPT" "$OURS_OUT"

for d in "$BASE_OUT" "$OURS_OUT"; do
  if [ ! -f "$d/trajectories.npz" ]; then
    echo "ERROR: missing $d/trajectories.npz" >&2
    exit 1
  fi
done

echo "[$(date)] Rendering bare vs ours success|failure pair videos -> $VIDEO_OUT"
python -m lerobot_policy_rewind_state_ours_variant6.common.render_compare_videos_success_vs_failure \
  --model           "bare:$BASE_OUT" \
  --model           "ours:$OURS_OUT" \
  --success_repo_id "$SUCCESS_REPO" \
  --failure_repo_id "$FAILURE_REPO" \
  --num_pairs       "$NUM_PAIRS" \
  --fps             "$FPS" \
  --plot_height     "$PLOT_H" \
  --cameras         "$(IFS=,; echo "${FIGURE_CAMERAS[*]}")" \
  --output_dir      "$VIDEO_OUT"

echo "[$(date)] Done."
echo "  bare eval:  $BASE_OUT"
echo "  ours eval:  $OURS_OUT"
echo "  videos:     $VIDEO_OUT/videos/"
