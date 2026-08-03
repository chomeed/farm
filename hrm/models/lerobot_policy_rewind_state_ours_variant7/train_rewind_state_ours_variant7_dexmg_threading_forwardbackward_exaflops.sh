#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant7_dexmg_threading_fb
#SBATCH --time=48:00:00
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/exaflops/chomeed/rewind_state_ours_variant7_dexmg_threading_fb_%j.out

# Forwardbackward sibling of train_rewind_state_ours_variant7_dexmg_threading_exaflops.sh
# (the "blend" run). Only `--failure_clip_directions` differs (matches the
# distinction in your existing variant3 v1_blend / v1_forwardbackward manifests):
#   blend           -> --failure_clip_directions "leaving"
#                      (one forward view of each failure clip, anchored AT the
#                       critical_t with the clip going forward in time)
#   forwardbackward -> --failure_clip_directions "leaving,approaching"
#                      (emits BOTH the forward view AND a mirrored view ending
#                       AT critical_t — twice the failure-side supervision)
#
# All other manifest knobs (match_dtw_weight=0.5, dtw_band=3,
# dtw_time_penalty_start=0.5, match_state_weight=0.4, match_window=16,
# success_anchor_mode=random) are identical between the two runs and match
# the existing variant3 v1_blend/v1_forwardbackward manifests.
#
# Distinct manifest path + output dir + wandb tag so this run and the blend
# run can coexist on disk without collisions.

# Activate the /lustre prefix env directly via PATH (no `conda` needed).
# gigabyte_a6000 nodes lack conda on PATH and --export=NONE strips ours, so
# we bypass `conda activate` and replicate its effect by hand.
ENV_PREFIX=${ENV_PREFIX:-/lustre/chomeed/exaflops/envs/resfit_td3_exaflops}
[ -x "$ENV_PREFIX/bin/python" ] || {
  echo "ERROR: env not found at $ENV_PREFIX" >&2; exit 1; }
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"
export CONDA_DEFAULT_ENV="$(basename "$ENV_PREFIX")"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/lustre/chomeed/exaflops/data}
OUT_ROOT=${OUT_ROOT:-/scratch2/exaFLOPs/chomeed/outputs}
MANIFEST_ROOT=${MANIFEST_ROOT:-/lustre/chomeed/exaflops/manifests}

export HF_HOME=${HF_HOME:-/lustre/chomeed/exaflops/hf_cache}
mkdir -p "$HF_HOME"

# ---------- Config ----------
BASE_CKPT=$DATA_ROOT/ckpts/rewind/rewind_state_dexmg_threading_v1/checkpoints/last/pretrained_model

# Reuse the existing variant3 forwardbackward manifest — reward-model-independent,
# so the same JSON that fed the variant3 v1_forwardbackward run drops straight in.
# This gives an apples-to-apples comparison vs the variant3 v1_forwardbackward
# plots (the only thing that changes is the model's loss function).
PREF_MANIFEST=$MANIFEST_ROOT/dexmg_two_arm_threading/preference_pairs_forwardbackward.json
# Mirror the manifest's own settings.
PRECOMPUTE_CLIP_LEN=16
PRECOMPUTE_FRAME_STRIDE=1
PRECOMPUTE_TOPK=3

DEXMG_CAMERA_KEYS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

SUCCESS_REPO=chomeed/dexmg_two_arm_threading
FAILURE_REPO=chomeed/dexmg_two_arm_threading_failures
# Both runs sample failure clips from FAILURE_REPO. This flag controls how
# each failure clip is anchored around its critical_t:
#   leaving      -> clip starts AT critical_t and goes forward (post-mistake)
#   approaching  -> clip ENDS at critical_t (pre-mistake window)
# forwardbackward emits both views per critical point; blend emits only "leaving".
FAILURE_CLIP_DIRECTIONS="leaving,approaching"

OUTPUT_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward
JOB_NAME=rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward

# Variant 7 knobs — BT on γ-discounted weighted mean of per-step rewards.
SUCCESS_ANCHOR_MODE=random
REWARD_PREF_LOSS_WEIGHT=1.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85          # gamma -- per-step BT weight = gamma^(T-1-t)
STOP_GRAD_ON_SUCCESS=True

# ---------- 1. Reuse the existing variant3 forwardbackward manifest ----------
# To regenerate from scratch, run precompute_preference_pairs with
# --failure_clip_directions "leaving,approaching" --match_dtw_weight 0.5
# --dtw_band 3 --dtw_time_penalty_start 0.5  (see git history for the
# previous block).
if [ ! -f "$PREF_MANIFEST" ]; then
  echo "ERROR: expected preference manifest at $PREF_MANIFEST" >&2
  echo "       (this script reuses the variant3 forwardbackward manifest)" >&2
  exit 1
fi
echo "[$(date)] Using preference manifest: $PREF_MANIFEST"

# ---------- 2. Train ----------
echo "[$(date)] Launching training"
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lerobot_policy_rewind_state_ours_variant7.common.train_rewind_state_ours_robomimic_variant7 \
  --policy.type=rewind_state_ours_variant7 \
  --policy.repo_id=chomeed/rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward \
  --policy.push_to_hub=False \
  --policy.num_cameras=3 \
  --policy.finetune_encoder=True \
  --policy.finetune_encoder_blocks=2 \
  --policy.state_dim=18 \
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.robot0_eye_in_hand","observation.images.robot1_eye_in_hand"]' \
  --policy.pretrained_path="$BASE_CKPT" \
  --policy.preference_pairs_path="$PREF_MANIFEST" \
  --policy.preference_clip_len="$PRECOMPUTE_CLIP_LEN" \
  --policy.preference_frame_stride="$PRECOMPUTE_FRAME_STRIDE" \
  --policy.preference_topk="$PRECOMPUTE_TOPK" \
  --policy.reward_pref_loss_weight="$REWARD_PREF_LOSS_WEIGHT" \
  --policy.reward_pref_warmup_steps="$REWARD_PREF_WARMUP_STEPS" \
  --policy.reward_pref_discount="$REWARD_PREF_DISCOUNT" \
  --policy.stop_grad_on_success="$STOP_GRAD_ON_SUCCESS" \
  --batch_size=4 --num_workers=8 --steps=100_000 --log_freq=100 \
  --eval_freq=1000 --save_freq=1000 --wandb.enable=True \
  --wandb.project=long_horizon_reward_modeling \
  --save_checkpoint=True \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.scheduler_decay_steps=100_000 \
  --policy.scheduler_peak_lr=5e-5 \
  --dataset.repo_id="$SUCCESS_REPO" \
  --val.repo_id="$SUCCESS_REPO"

echo "[$(date)] Done."
