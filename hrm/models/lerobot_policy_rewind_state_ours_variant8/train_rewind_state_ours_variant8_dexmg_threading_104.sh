#!/bin/bash
# variant8 reward model for dexmg two_arm_threading, on the 104 box.
# Adapted from train_rewind_state_ours_variant8_dexmg_three_piece_exaflops.sh
# (variant8 knobs + module) and the variant6 threading script (task paths).
#
# Edit the partition/qos below for the 104 cluster, or override at submit:
#   sbatch --partition=<p> --qos=<q> <this file>
# Manual (no slurm):  bash <this file>
#SBATCH --job-name=rewind_state_ours_variant8_dexmg_threading
#SBATCH --time=12:00:00
#SBATCH --partition=SET_104_PARTITION
#SBATCH --qos=SET_104_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate farm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

cd /data4/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state_ours_variant8/src/lerobot_policy_rewind_state_ours_variant8

# ---------- Config ----------
# DATA_ROOT points at the staged data on 104. No sudo there, so instead of a
# /lustre symlink the manifest's baked-in base_ckpt string is patched to /data4
# (run patch_threading_paths_to_data4_104.py once after staging); BASE_CKPT
# below must then match the patched string exactly.
DATA_ROOT=${DATA_ROOT:-/data4/chomeed/data}
OUT_ROOT=${OUT_ROOT:-/data4/chomeed/outputs}

BASE_CKPT=$DATA_ROOT/ckpts/rewind/rewind_state_dexmg_threading_v1/checkpoints/last/pretrained_model
PREF_MANIFEST=$DATA_ROOT/manifests/dexmg_two_arm_threading/preference_pairs_blend.json
PRECOMPUTE_CLIP_LEN=16
PRECOMPUTE_FRAME_STRIDE=1
PRECOMPUTE_TOPK=3

SUCCESS_REPO=chomeed/dexmg_two_arm_threading
FAILURE_REPO=chomeed/dexmg_two_arm_threading_failures

OUTPUT_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant8_dexmg_threading_v1
JOB_NAME=rewind_state_ours_variant8_dexmg_threading_v1

# Variant 6 inherited knobs
SUCCESS_ANCHOR_MODE=random
REWARD_PREF_LOSS_WEIGHT=1.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85
STOP_GRAD_ON_SUCCESS=True

# Variant 8 knobs. mono_skip_pre_critical_anchors=2 is correct for this
# manifest: clip_len=16, pre_critical_frames=5, max_post_critical_frames=60
# puts anchors 0 and 1 before critical_t.
FAILURE_SAMPLING_MODE=hybrid
MONO_SKIP_PRE_CRITICAL_ANCHORS=2

# ---------- 1. Manifest sanity check ----------
if [ ! -f "$PREF_MANIFEST" ]; then
  echo "ERROR: expected preference manifest at $PREF_MANIFEST" >&2
  echo "       Stage it with sync_threading_data_to_data4.sh, then run" >&2
  echo "       patch_threading_paths_to_data4_104.py on 104." >&2
  exit 1
fi
echo "[$(date)] Using preference manifest: $PREF_MANIFEST"

# ---------- 2. Train ----------
echo "[$(date)] Launching training (30k steps, save every 2k)"
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lerobot_policy_rewind_state_ours_variant8.common.train_rewind_state_ours_robomimic_variant8 \
  --policy.type=rewind_state_ours_variant8 \
  --policy.repo_id=chomeed/rewind_state_ours_variant8_dexmg_threading_v1 \
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
  --policy.failure_sampling_mode="$FAILURE_SAMPLING_MODE" \
  --policy.mono_skip_pre_critical_anchors="$MONO_SKIP_PRE_CRITICAL_ANCHORS" \
  --batch_size=4 --num_workers=8 --steps=30000 --log_freq=100 \
  --eval_freq=2000 --save_freq=2000 --wandb.enable=True \
  --wandb.project=long_horizon_reward_modeling \
  --save_checkpoint=True \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_peak_lr=5e-5 \
  --dataset.repo_id="$SUCCESS_REPO" \
  --val.repo_id="$SUCCESS_REPO"

echo "[$(date)] Done. Reward ckpt for ResFit: $OUTPUT_DIR/checkpoints/002000/pretrained_model"
