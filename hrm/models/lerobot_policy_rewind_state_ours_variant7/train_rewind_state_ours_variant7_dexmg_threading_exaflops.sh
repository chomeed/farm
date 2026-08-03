#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant7_dexmg_threading
#SBATCH --time=48:00:00
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

# EXAFLOPS-ACCOUNT variant of train_rewind_state_ours_variant7_dexmg_threading.sh.
# Identical trainer/args; only the env + paths differ so it runs under a second
# user account (own QOS quota) off shared /lustre:
#   - conda env  -> the /lustre prefix env (clone of robomimic_square2)
#   - base ckpt  -> /lustre/chomeed/exaflops/data  (run copy_exaflops_data.sh first)
#   - manifest + outputs -> /lustre/chomeed/exaflops
# The success/failure datasets load straight from the HuggingFace hub by
# repo_id (compute nodes have hub access), so no local --root is needed.
# sbatch this file as the exaflops account.

# Activate the /lustre prefix env directly via PATH (no `conda` needed).
# Background: the `gigabyte_a6000` compute nodes don't have conda on the
# default PATH, and `sbatch --export=NONE` strips the submitting shell's PATH,
# so neither `conda info --base` nor `source ~/.bashrc` finds conda.
# Setting PATH/CONDA_PREFIX/LD_LIBRARY_PATH manually replicates what
# `conda activate` would do — sufficient for Python + pip + dynamic libs.
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

# HuggingFace / LeRobot dataset cache on /lustre. /home/exaflops is near-full,
# so dataset downloads (hub blobs + lerobot cache) must NOT land in the default
# ~/.cache. copy_exaflops_data.sh creates this dir world-writable.
export HF_HOME=${HF_HOME:-/lustre/chomeed/exaflops/hf_cache}
mkdir -p "$HF_HOME"

# ---------- Config ----------
# copy_exaflops_data.sh stages the rewind base ckpt's `last` checkpoint
# (currently == step 050000).
BASE_CKPT=$DATA_ROOT/ckpts/rewind/rewind_state_dexmg_threading_v1/checkpoints/last/pretrained_model

# Reuse the existing variant3 blend manifest — it's reward-model-independent
# (just a list of (success_clip, failure_clip) pairs) and using it gives an
# apples-to-apples comparison vs the variant3 v1_blend run.
PREF_MANIFEST=$MANIFEST_ROOT/dexmg_two_arm_threading/preference_pairs_blend.json
# Mirror the manifest's own settings (read from preference_pairs_blend.json:
# clip_len=16, frame_stride=1, top_k=3). The trainer's --policy.preference_*
# args MUST match the manifest, otherwise indexing breaks.
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

OUTPUT_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.2_succmse
JOB_NAME=rewind_state_ours_variant7_dexmg_threading_v1_blend_mono_anchorstack_eps0.2_succmse

# Variant 6 knobs — BT OFF, monotonicity hinge on pref-clip predictions.
SUCCESS_ANCHOR_MODE=random            # 'critical' (Variant 3) or 'random' (base ours)
REWARD_PREF_LOSS_WEIGHT=0.0           # BT off — isolate mono + variant7 succ-anchor MSE
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85
BT_TARGET_MARGIN=2.0

# Monotonicity hinge (sigmoid-space, on adjacent-frame differences).
MONO_LOSS_WEIGHT=1.0
MONO_WARMUP_STEPS=500
MONO_ON_FAILURE=True                  # penalize r(t+1) > r(t) on failure clips
MONO_ON_SUCCESS=True                  # penalize r(t+1) < r(t) on success clips
MONO_STEP_EPS=0.2                     # logit-space min descent per step (leaving dir)

# Optional TV smoothness regularizer (logit space). OFF by default.
TV_LOSS_WEIGHT=0.0
TV_WARMUP_STEPS=500
TV_ORDER=2
TV_ON_FAILURE=True
TV_ON_SUCCESS=True

# Hybrid failure-clip sampling: 8 uniform + 8 contiguous around manifest anchor.
FAILURE_SAMPLING_MODE=hybrid
FAILURE_HYBRID_N_RECENT=8

FAILURE_CLIP_DIRECTIONS="leaving"

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

# ---------- 1. Reuse the existing variant3 blend manifest ----------
if [ ! -f "$PREF_MANIFEST" ]; then
  echo "ERROR: expected preference manifest at $PREF_MANIFEST" >&2
  echo "       (this script reuses the variant3 blend manifest)" >&2
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
