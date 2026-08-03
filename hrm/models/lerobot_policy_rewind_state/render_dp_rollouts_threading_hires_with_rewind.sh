#!/bin/bash
#SBATCH --job-name=render_dp_threading_hires_rewind
#SBATCH --time=6:00:00
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=render_dp_threading_hires_rewind_%j.out

# Two-stage pipeline for dexmg two_arm_threading:
#   1. Roll out the DP last ckpt once -> saves hi-res mp4s + per-camera mp4s
#      + obs_policy.npz (policy-res frames + state + action) per episode.
#   2. Score those saved rollouts post-hoc with BOTH rewind reward models
#      (bare baseline @50k, ours-variant6 anchorstack-blend eps0.3 @2k) ->
#      rewind_reward__<label>.npy next to each obs_policy.npz.
#
# Decoupling rollouts from reward inference lets you add/remove reward models,
# tune sampling/n_recent, or build cliff figures without paying for DP rollouts
# again. Skip-sentinels: stage 1 skips if OUT_DIR/episodes.json exists; stage 2
# skips per-label files that already exist (use --overwrite to force).
#
# Per-episode layout under $OUT_DIR after both stages:
#   ep{NNNN}_env{i}_seed{S}_{success|failure}.mp4        composite 3-cam strip
#   ep{NNNN}_env{i}_seed{S}_{success|failure}/
#       agentview.mp4                                    per-camera hi-res mp4s
#       robot0_eye_in_hand.mp4
#       robot1_eye_in_hand.mp4
#       obs_policy.npz                                   policy-res frames + state + action
#       rewind_reward__bare.npy                          bare reward curve  (stage 2)
#       rewind_reward__ours_variant6.npy                 ours reward curve  (stage 2)
#       meta.json
#   episodes.json
#
# Submit on slurm:
#   sbatch --export=NONE \
#     /home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state/render_dp_rollouts_threading_hires_with_rewind.sh
#
# Local dry-run (prints commands, no env activation):
#   DRY_RUN=1 bash .../render_dp_rollouts_threading_hires_with_rewind.sh

DRY_RUN=${DRY_RUN:-}
if [ -z "$DRY_RUN" ]; then
  source ~/.bashrc
  conda activate "${CONDA_ENV:-robomimic_square2}"
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/chomeed/workspace/proj-rl-finetuning/hrm}
ROLLOUT_SCRIPT="$REPO_ROOT/finetuning/lerobot_policy_resfit_td3/dexmg/render_dp_rollouts_dexmg.py"
REWIND_SCRIPT="$REPO_ROOT/finetuning/lerobot_policy_resfit_td3/dexmg/compute_rewind_rewards_offline.py"
FIGURE_SCRIPT="$REPO_ROOT/finetuning/lerobot_policy_resfit_td3/dexmg/make_rewind_overlay_figures.py"
for s in "$ROLLOUT_SCRIPT" "$REWIND_SCRIPT" "$FIGURE_SCRIPT"; do
  if [ ! -f "$s" ]; then
    echo "ERROR: python script missing: $s (set REPO_ROOT to your hrm/ checkout)" >&2
    exit 1
  fi
done

CHECKPOINT=${CHECKPOINT:-/lustre/chomeed/exaflops/data/ckpts/train/dexmg_two_arm_threading_dp_Tp8_Ta8/checkpoints/last/pretrained_model}
ENV_DATASET=${ENV_DATASET:-/lustre/chomeed/exaflops/data/dexmimicgen_datasets/generated/two_arm_threading.hdf5}
REWIND_BOOTSTRAP_REPO=${REWIND_BOOTSTRAP_REPO:-chomeed/dexmg_two_arm_threading}

NUM_EPISODES=${NUM_EPISODES:-8}
NUM_ENVS=${NUM_ENVS:-1}
SEED=${SEED:-100}
POLICY_IMAGE_SIZE=${POLICY_IMAGE_SIZE:-84}    # MUST match DP training resolution.
                                              # The DP normalizer config (and
                                              # the training dataset features)
                                              # both fix observation.images.*
                                              # at shape [3, 84, 84]; there is
                                              # no resize in the policy or its
                                              # preprocessor, so feeding 224
                                              # silently degrades performance.
RENDER_HEIGHT=${RENDER_HEIGHT:-368}
RENDER_WIDTH=${RENDER_WIDTH:-368}
VIDEO_FPS=${VIDEO_FPS:-20}
REWIND_SAMPLING=${REWIND_SAMPLING:-hybrid}
REWIND_N_RECENT=${REWIND_N_RECENT:-8}

BASE_STEP=050000
OURS_STEP=002000

BASE_CKPT_ROOT=${BASE_CKPT_ROOT:-/scratch2/chomeed/outputs/rewind}
BASE_RUN=rewind_state_dexmg_threading_v1
BASE_REWIND_CKPT=${BASE_REWIND_CKPT:-$BASE_CKPT_ROOT/$BASE_RUN/checkpoints/$BASE_STEP/pretrained_model}

OURS_REWARD_CKPT_ROOT=${OURS_REWARD_CKPT_ROOT:-$BASE_CKPT_ROOT}
OURS_RUN=rewind_state_ours_variant6_dexmg_threading_v1_blend_mono_anchorstack_eps0.3
OURS_REWIND_CKPT=${OURS_REWIND_CKPT:-$OURS_REWARD_CKPT_ROOT/$OURS_RUN/checkpoints/$OURS_STEP/pretrained_model}

OUT_DIR=${OUT_DIR:-$BASE_CKPT_ROOT/_dp_rollouts_threading_hires_seed${SEED}_${RENDER_HEIGHT}x${RENDER_WIDTH}}

if [ ! -d "$CHECKPOINT" ]; then
  echo "ERROR: DP checkpoint missing: $CHECKPOINT" >&2
  exit 1
fi
if [ ! -f "$ENV_DATASET" ]; then
  echo "ERROR: env dataset missing: $ENV_DATASET" >&2
  exit 1
fi
for r in "$BASE_REWIND_CKPT" "$OURS_REWIND_CKPT"; do
  if [ ! -d "$r" ]; then
    echo "ERROR: rewind ckpt missing: $r" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

ROLLOUT_CMD=(env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  python "$ROLLOUT_SCRIPT"
  --checkpoint            "$CHECKPOINT"
  --env_dataset           "$ENV_DATASET"
  --out_dir               "$OUT_DIR"
  --num_envs              "$NUM_ENVS"
  --num_episodes          "$NUM_EPISODES"
  --seed                  "$SEED"
  --policy_image_size     "$POLICY_IMAGE_SIZE"
  --render_height         "$RENDER_HEIGHT"
  --render_width          "$RENDER_WIDTH"
  --video_fps             "$VIDEO_FPS"
)

if [ -f "$OUT_DIR/episodes.json" ]; then
  echo "[$(date)] Skip stage 1 (DP rollouts) -- episodes.json exists at $OUT_DIR"
else
  echo "[$(date)] Stage 1: DP rollouts -> $OUT_DIR"
  echo "  DP ckpt: $CHECKPOINT"
  echo "  episodes=$NUM_EPISODES envs=$NUM_ENVS seed=$SEED"
  echo "  policy_img=$POLICY_IMAGE_SIZE  render=${RENDER_HEIGHT}x${RENDER_WIDTH}  fps=$VIDEO_FPS"
  if [ -n "$DRY_RUN" ]; then
    printf '%q ' "${ROLLOUT_CMD[@]}"; echo
  else
    "${ROLLOUT_CMD[@]}"
  fi
fi

REWIND_CMD=(python "$REWIND_SCRIPT"
  --rollout_dir           "$OUT_DIR"
  --rewind_bootstrap_repo "$REWIND_BOOTSTRAP_REPO"
  --rewind_ckpt           "bare:$BASE_REWIND_CKPT"
  --rewind_ckpt           "ours_variant6:$OURS_REWIND_CKPT"
  --rewind_sampling       "$REWIND_SAMPLING"
  --rewind_n_recent       "$REWIND_N_RECENT"
)

echo "[$(date)] Stage 2: post-hoc rewind scoring -> $OUT_DIR/ep*/rewind_reward__<label>.npy"
echo "  bare:           $BASE_REWIND_CKPT"
echo "  ours_variant6:  $OURS_REWIND_CKPT"
if [ -n "$DRY_RUN" ]; then
  printf '%q ' "${REWIND_CMD[@]}"; echo
else
  "${REWIND_CMD[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 3: per-episode overlay figures. For each episode, writes both:
#   $OUT_DIR/figures/cliff_<ep>.png   agentview frame at the cliff (= argmax
#                                     of |ours - bare|, smoothed) + reward
#                                     chart with both curves and a marker
#   $OUT_DIR/figures/plot_<ep>.png    same chart only (no image)
# ---------------------------------------------------------------------------
FIGURE_CMD=(python "$FIGURE_SCRIPT"
  --rollout_dir "$OUT_DIR"
  --labels      "bare,ours_variant6"
  --smooth      5
  --camera      agentview
)

echo "[$(date)] Stage 3: overlay figures -> $OUT_DIR/figures/"
if [ -n "$DRY_RUN" ]; then
  printf '%q ' "${FIGURE_CMD[@]}"; echo
else
  "${FIGURE_CMD[@]}"
fi

echo "[$(date)] Done -> $OUT_DIR"
