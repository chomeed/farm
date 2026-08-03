#!/bin/bash
# Local 4090 launch — "forwardbackward" sibling of
# train_rewind_state_ours_variant7_dexmg_threading.sh (the blend run).
# Reuses the existing variant3 forwardbackward manifest, which emits BOTH
# the forward ("leaving") and mirrored ("approaching") view of each
# failure clip — twice the failure-side supervision vs blend.
# Sbatch headers retained as comments so this file is sbatch-launchable too.
#SBATCH --job-name=rewind_state_ours_variant7_dexmg_threading_fb
#SBATCH --time=48:00:00
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

cd /home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state_ours_variant7/src/lerobot_policy_rewind_state_ours_variant7

# ---------- Config ----------
BASE_CKPT=/lustre/chomeed/exaflops/ckpts/rewind/rewind_state_dexmg_threading_v1/checkpoints/last/pretrained_model

PREF_MANIFEST=/lustre/chomeed/exaflops/manifests/dexmg_two_arm_threading/preference_pairs_forwardbackward.json
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

OUTPUT_DIR=/scratch2/chomeed/outputs/rewind/rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.2_0.0
JOB_NAME=rewind_state_ours_variant7_dexmg_threading_v1_forwardbackward_mono_anchorstack_eps0.2_0.0

# Variant 7 knobs — BT OFF, monotonicity hinge on pref-clip predictions.
SUCCESS_ANCHOR_MODE=random
REWARD_PREF_LOSS_WEIGHT=0.0           # BT off for this run — isolate mono effect
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85
STOP_GRAD_ON_SUCCESS=False
BT_TARGET_MARGIN=2.0

# Monotonicity hinge. NOTE: this forwardbackward manifest emits failure
# clips in BOTH directions ("leaving" → frames after critical_t, should
# decrease; "approaching" → frames before critical_t, should INCREASE).
# Applying mono_on_failure=True here will mis-supervise the "approaching"
# half. Disable until we add direction-aware handling (see config docstring).
MONO_LOSS_WEIGHT=1.0
MONO_WARMUP_STEPS=500
MONO_ON_FAILURE=True
MONO_ON_SUCCESS=True
# Directional eps on fb (manifest has both "leaving" and "approaching"):
#   leaving pairs    → strict descent (eps=0.3, same as blend)
#   approaching pairs → soft (eps=0, only prevent increases — MSE on rewind
#                       is naturally pulling these up since they're pre-failure)
MONO_STEP_EPS=0.2                     # leaving direction
MONO_STEP_EPS_APPROACHING=0.0         # approaching direction

# Optional TV smoothness regularizer. Default OFF.
TV_LOSS_WEIGHT=0.0
TV_WARMUP_STEPS=500
TV_ORDER=2
TV_ON_FAILURE=True
TV_ON_SUCCESS=True

FAILURE_SAMPLING_MODE=hybrid
FAILURE_HYBRID_N_RECENT=8

# forwardbackward = failure clips anchored at critical_t in BOTH directions.
FAILURE_CLIP_DIRECTIONS="leaving,approaching"

# ---------- 1. Reuse the existing variant3 forwardbackward manifest ----------
if [ ! -f "$PREF_MANIFEST" ]; then
  echo "ERROR: expected preference manifest at $PREF_MANIFEST" >&2
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
  --policy.bt_target_margin="$BT_TARGET_MARGIN" \
  --policy.mono_loss_weight="$MONO_LOSS_WEIGHT" \
  --policy.mono_warmup_steps="$MONO_WARMUP_STEPS" \
  --policy.mono_on_failure="$MONO_ON_FAILURE" \
  --policy.mono_on_success="$MONO_ON_SUCCESS" \
  --policy.mono_step_eps="$MONO_STEP_EPS" \
  --policy.mono_step_eps_approaching="$MONO_STEP_EPS_APPROACHING" \
  --policy.tv_loss_weight="$TV_LOSS_WEIGHT" \
  --policy.tv_warmup_steps="$TV_WARMUP_STEPS" \
  --policy.tv_order="$TV_ORDER" \
  --policy.tv_on_failure="$TV_ON_FAILURE" \
  --policy.tv_on_success="$TV_ON_SUCCESS" \
  --policy.failure_sampling_mode="$FAILURE_SAMPLING_MODE" \
  --policy.failure_hybrid_n_recent="$FAILURE_HYBRID_N_RECENT" \
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
