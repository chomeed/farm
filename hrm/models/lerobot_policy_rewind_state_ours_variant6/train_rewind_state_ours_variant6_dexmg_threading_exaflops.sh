#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant6_dexmg_threading
#SBATCH --time=48:00:00
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/exaflops/chomeed/rewind_state_ours_variant6_dexmg_threading_%j.out

# EXAFLOPS-ACCOUNT variant of train_rewind_state_ours_variant6_dexmg_threading.sh.
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

OUTPUT_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant6_dexmg_threading_v1
JOB_NAME=rewind_state_ours_variant6_dexmg_threading_v1

# Variant 6 knobs — BT on γ-discounted weighted mean of per-step rewards.
SUCCESS_ANCHOR_MODE=random          # 'critical' (Variant 3) or 'random' (base ours)
REWARD_PREF_LOSS_WEIGHT=1.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85          # gamma -- per-step BT weight = gamma^(T-1-t)
STOP_GRAD_ON_SUCCESS=True             # detach succ_score — BT only suppresses failure

# ---------- 1. Reuse the existing variant3 blend manifest ----------
# The manifest is reward-model-independent (just (success_clip, failure_clip)
# pairs), so we use the same JSON that fed the variant3 v1_blend run.
# To regenerate from scratch, run precompute_preference_pairs with
# --failure_clip_directions "leaving" --match_dtw_weight 0.5 --dtw_band 3
# --dtw_time_penalty_start 0.5  (see git history for the previous block).
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
  -m lerobot_policy_rewind_state_ours_variant6.common.train_rewind_state_ours_robomimic_variant6 \
  --policy.type=rewind_state_ours_variant6 \
  --policy.repo_id=chomeed/rewind_state_ours_variant6_dexmg_threading_v1 \
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
