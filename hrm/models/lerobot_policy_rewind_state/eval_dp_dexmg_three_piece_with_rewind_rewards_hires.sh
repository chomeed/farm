#!/bin/bash
set -euo pipefail

# Variant of eval_baseline_dexmg_three_piece_success_vs_failure_step2k.sh.
#
# Instead of comparing two rewind reward checkpoints on real success/failure
# demos, this script:
#   1) rolls out the *frozen* DiffusionPolicy (last ckpt) on
#      dexmg_two_arm_three_piece_assembly at HIGH render resolution and
#      writes per-episode mp4s, per-camera mp4s, and obs_policy.npz;
#   2) scores each saved rollout offline with both the bare baseline (50k)
#      and variant6-anchorstack-blend (2k) rewind reward checkpoints,
#      writing rewind_reward_<label>.npy plus a single-curve png per model;
#   3) renders a combined per-episode figure (filmstrip of K keyframes +
#      both reward curves overlaid).
#
# Submit on a GPU box; needs MuJoCo (egl) for the DP rollout step.

ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-/home/chomeed/miniconda3/envs/proj-rl-finetuning}}"
[ -x "$ENV_PREFIX/bin/python" ] || {
  echo "ERROR: env not found at $ENV_PREFIX (set ENV_PREFIX to a valid conda env root)" >&2
  exit 1
}
export PATH="$ENV_PREFIX/bin:${PATH:-/usr/bin:/bin}"
export CONDA_PREFIX="$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

# Reward checkpoint steps (kept consistent with the source script so the
# scored rollouts compare against the same models).
BASE_STEP=050000
OURS_STEP=002000
SAMPLING=hybrid
N_RECENT=8

# DiffusionPolicy rollout knobs.
DP_CKPT=${DP_CKPT:-/lustre/chomeed/exaflops/data/ckpts/train/dexmg_two_arm_three_piece_assembly_dp_Tp8_Ta8/checkpoints/last/pretrained_model}
ENV_DATASET=${ENV_DATASET:-/lustre/chomeed/exaflops/data/dexmimicgen_datasets/generated/two_arm_three_piece_assembly.hdf5}
NUM_EPISODES=${NUM_EPISODES:-8}
NUM_ENVS=${NUM_ENVS:-4}
SEED=${SEED:-100}
POLICY_IMAGE_SIZE=${POLICY_IMAGE_SIZE:-224}
RENDER_HEIGHT=${RENDER_HEIGHT:-720}
RENDER_WIDTH=${RENDER_WIDTH:-720}
VIDEO_FPS=${VIDEO_FPS:-20}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-6}
FILMSTRIP_CAMERA=${FILMSTRIP_CAMERA:-agentview}

# Rewind reward checkpoints (same as the source script).
BASE_CKPT_ROOT=${BASE_CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
BASE_RUN=rewind_state_dexmg_three_piece_v1
BASE_CKPT=${BASE_CKPT:-$BASE_CKPT_ROOT/$BASE_RUN/checkpoints/$BASE_STEP/pretrained_model}

OURS_REWARD_CKPT_ROOT=${OURS_REWARD_CKPT_ROOT:-/lustre/chomeed/gwanwoo13/scratch2_outputs/rewind}
OURS_RUN=rewind_state_ours_variant6_dexmg_three_piece_v1_blend_mono_anchorstack_eps0.3
OURS_CKPT=${OURS_CKPT:-$OURS_REWARD_CKPT_ROOT/$OURS_RUN/checkpoints/$OURS_STEP/pretrained_model}

BOOTSTRAP_REPO=${BOOTSTRAP_REPO:-chomeed/dexmg_two_arm_three_piece_assembly}

ROLLOUT_OUT=${ROLLOUT_OUT:-$BASE_CKPT_ROOT/_dp_three_piece_rollouts_base${BASE_STEP}_vs_ours${OURS_STEP}_hybrid${N_RECENT}_${RENDER_HEIGHT}x${RENDER_WIDTH}_seed${SEED}}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROLLOUT_PY="$REPO_ROOT/hrm/finetuning/lerobot_policy_resfit_td3/dexmg/render_dp_rollouts_dexmg.py"
SCORE_PY="$REPO_ROOT/hrm/finetuning/lerobot_policy_resfit_td3/dexmg/score_dp_rollouts_with_rewinds.py"

[ -d "$DP_CKPT" ] || { echo "ERROR: DP ckpt missing: $DP_CKPT" >&2; exit 1; }
[ -f "$ENV_DATASET" ] || { echo "ERROR: env hdf5 missing: $ENV_DATASET" >&2; exit 1; }
[ -d "$BASE_CKPT" ] || { echo "ERROR: bare rewind ckpt missing: $BASE_CKPT" >&2; exit 1; }
[ -d "$OURS_CKPT" ] || { echo "ERROR: ours rewind ckpt missing: $OURS_CKPT" >&2; exit 1; }
[ -f "$ROLLOUT_PY" ] || { echo "ERROR: $ROLLOUT_PY not found" >&2; exit 1; }
[ -f "$SCORE_PY" ]   || { echo "ERROR: $SCORE_PY not found"   >&2; exit 1; }

mkdir -p "$ROLLOUT_OUT"

# ---------------------------------------------------------------------------
# Step 1: roll out the DP at high resolution. Skip if episodes.json exists.
# ---------------------------------------------------------------------------
if [ -f "$ROLLOUT_OUT/episodes.json" ]; then
  echo "[$(date)] Skip DP rollout — episodes.json exists at $ROLLOUT_OUT"
else
  echo "[$(date)] DP rollout (high-res) -> $ROLLOUT_OUT"
  python "$ROLLOUT_PY" \
    --checkpoint         "$DP_CKPT" \
    --env_dataset        "$ENV_DATASET" \
    --out_dir            "$ROLLOUT_OUT" \
    --num_envs           "$NUM_ENVS" \
    --num_episodes       "$NUM_EPISODES" \
    --seed               "$SEED" \
    --policy_image_size  "$POLICY_IMAGE_SIZE" \
    --render_height      "$RENDER_HEIGHT" \
    --render_width       "$RENDER_WIDTH" \
    --video_fps          "$VIDEO_FPS" \
    --save_individual_cameras \
    --save_obs_npz
fi

# ---------------------------------------------------------------------------
# Step 2: score each saved rollout with bare + ours reward models, write
# combined per-episode figure.
# ---------------------------------------------------------------------------
echo "[$(date)] Scoring rollouts with bare ($BASE_STEP) + ours ($OURS_STEP)"
python "$SCORE_PY" \
  --rollout_dir        "$ROLLOUT_OUT" \
  --rewind             "bare:$BASE_CKPT:$BOOTSTRAP_REPO" \
  --rewind             "ours:$OURS_CKPT:$BOOTSTRAP_REPO" \
  --sampling           "$SAMPLING" \
  --n_recent           "$N_RECENT" \
  --num_keyframes      "$NUM_KEYFRAMES" \
  --filmstrip_camera   "$FILMSTRIP_CAMERA" \
  --colors             "bare=tab:gray,ours=tab:red"

echo "[$(date)] Done."
echo "  DP rollouts:        $ROLLOUT_OUT"
echo "  per-camera mp4s:    $ROLLOUT_OUT/ep*/{${FILMSTRIP_CAMERA},robot0_eye_in_hand,robot1_eye_in_hand}.mp4"
echo "  rewind reward npys: $ROLLOUT_OUT/ep*/rewind_reward_{bare,ours}.npy"
echo "  per-model curves:   $ROLLOUT_OUT/ep*/rewind_reward_{bare,ours}.png"
echo "  combined figure:    $ROLLOUT_OUT/ep*/combined_reward_curves.png"
