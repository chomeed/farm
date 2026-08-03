#!/bin/bash
# Local 4090 launch — "blend" variant (failure clips emitted in the
# "leaving" direction only). Sbatch headers retained as comments so this
# file is sbatch-launchable too if needed.
#SBATCH --job-name=rewind_state_ours_variant7_dexmg_threading
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

# Enable strict mode AFTER sourcing bashrc — the system /etc/bashrc has an
# unbound BASHRCSOURCED reference that trips `set -u`.
set -euo pipefail

cd /home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state_ours_variant7/src/lerobot_policy_rewind_state_ours_variant7

# ---------- Config ----------
# Canonical paths (shared with exaflops; /lustre is mounted on the local box).
BASE_CKPT=/lustre/chomeed/exaflops/ckpts/rewind/rewind_state_dexmg_threading_v1/checkpoints/last/pretrained_model

# Reuse the existing variant3 blend manifest — reward-model-independent,
# so the same JSON used for variant3/variant5 blend runs drops in here.
PREF_MANIFEST=/lustre/chomeed/exaflops/manifests/dexmg_two_arm_threading/preference_pairs_blend.json
PRECOMPUTE_CLIP_LEN=16
PRECOMPUTE_FRAME_STRIDE=1
PRECOMPUTE_TOPK=3

# dexmg 3-camera observation keys.
DEXMG_CAMERA_KEYS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

SUCCESS_REPO=chomeed/dexmg_two_arm_threading
FAILURE_REPO=chomeed/dexmg_two_arm_threading_failures

OUTPUT_DIR=/scratch2/chomeed/outputs/rewind/rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.2_succmse
JOB_NAME=rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.2_succmse

# Variant 7 knobs — BT OFF, monotonicity hinge on pref-clip predictions.
# Mono enforces rise-then-fall SHAPE via pairwise constraints on adjacent
# frames; no absolute progress targets needed.
SUCCESS_ANCHOR_MODE=random          # 'critical' (Variant 3) or 'random' (base ours)
REWARD_PREF_LOSS_WEIGHT=0.0           # BT off for this run — isolate mono effect
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85
STOP_GRAD_ON_SUCCESS=False
BT_TARGET_MARGIN=2.0

# Monotonicity hinge (sigmoid-space, on adjacent-frame differences).
MONO_LOSS_WEIGHT=1.0
MONO_WARMUP_STEPS=500
MONO_ON_FAILURE=True                  # penalize r(t+1) > r(t) on failure clips
MONO_ON_SUCCESS=True                  # penalize r(t+1) < r(t) on success clips
MONO_STEP_EPS=0.2                     # logit-space min descent per step (leaving dir)
                                      # → expected margin ≈ 3 → succ ≈ 0.82, fail tail ≈ 0.18

# Optional TV smoothness regularizer (logit space). Default OFF
# (weight=0) — opt in by raising TV_LOSS_WEIGHT to e.g. 0.05.
TV_LOSS_WEIGHT=0.0
TV_WARMUP_STEPS=500
TV_ORDER=2                            # 1=L1 (chatter), 2=L2 (cliff)
TV_ON_FAILURE=True
TV_ON_SUCCESS=True

# Hybrid failure-clip sampling: ignore manifest's 16 indices around critical_t;
# load 8 uniform from [0, raw_end-8) + 8 contiguous at [raw_end-8, raw_end).
# Mono on failure auto-switches to SEGMENTED in hybrid mode (first 8 frames
# mono-INCREASE, last 8 mono-DECREASE).
FAILURE_SAMPLING_MODE=hybrid
FAILURE_HYBRID_N_RECENT=8

# Variant 7: success-side hybrid + MSE (peer of the failure-side block above).
# Success preference clip becomes an N_anchors-deep anchor stack, and the
# per-anchor last logit is supervised by MSE against frame-progress.
SUCCESS_SAMPLING_MODE=hybrid
SUCCESS_HYBRID_N_RECENT=8
SUCC_ANCHOR_MSE_WEIGHT=1.0
SUCC_ANCHOR_MSE_WARMUP_STEPS=500

# Variant 7 flips stop_grad_on_success default → False so the new MSE has
# gradient access. Override explicitly here for clarity (already the default).
STOP_GRAD_ON_SUCCESS=False

# "blend" = failure clips anchored at critical_t going forward only.
# (forwardbackward sibling adds the mirrored "approaching" view too.)
FAILURE_CLIP_DIRECTIONS="leaving"

# ---------- 1. Reuse the existing variant3 blend manifest ----------
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
  --policy.repo_id=chomeed/rewind_state_ours_variant7_dexmg_threading_v1 \
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
  --policy.tv_loss_weight="$TV_LOSS_WEIGHT" \
  --policy.tv_warmup_steps="$TV_WARMUP_STEPS" \
  --policy.tv_order="$TV_ORDER" \
  --policy.tv_on_failure="$TV_ON_FAILURE" \
  --policy.tv_on_success="$TV_ON_SUCCESS" \
  --policy.failure_sampling_mode="$FAILURE_SAMPLING_MODE" \
  --policy.failure_hybrid_n_recent="$FAILURE_HYBRID_N_RECENT" \
  --policy.success_sampling_mode="$SUCCESS_SAMPLING_MODE" \
  --policy.success_hybrid_n_recent="$SUCCESS_HYBRID_N_RECENT" \
  --policy.succ_anchor_mse_weight="$SUCC_ANCHOR_MSE_WEIGHT" \
  --policy.succ_anchor_mse_warmup_steps="$SUCC_ANCHOR_MSE_WARMUP_STEPS" \
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
