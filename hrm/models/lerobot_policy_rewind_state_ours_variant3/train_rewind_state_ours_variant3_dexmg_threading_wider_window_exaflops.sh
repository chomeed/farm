#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant3_dexmg_threading_wider_window
#SBATCH --time=48:00:00
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/exaflops/chomeed/rewind_state_ours_variant3_dexmg_threading_wider_window_%j.out

# Wider-K ablation of train_rewind_state_ours_variant3_dexmg_threading_exaflops.sh.
# Diffs from the K=8 baseline:
#   - REWARD_PREF_CRITICAL_WINDOW: 8 -> 32
#   - PRECOMPUTE_CLIP_LEN:         16 -> 32 (so K=32 isn't clamped at runtime)
#   - PREF_MANIFEST path:          .../preference_pairs_clip32.json (new file)
#   - output / wandb naming:       _wider_window suffix
#
# Motivation: with K=8 on clip_len=16 BT supervises only the trailing half of
# each clip, leaving the first half as an unsupervised boundary. The reward
# model can satisfy BT with a sharp step right at frame T-K, producing a cliff.
# Widening K to the whole clip averages BT pressure across all frames,
# removing the boundary and forcing a smoother monotone failure signal.
#
# Manifest regenerates from scratch at the new clip_len=32 — this takes
# noticeably longer than the K=8 run's precompute step. Comment out the
# precompute block on re-runs to reuse the existing clip32 manifest.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/chomeed/exaflops/envs/resfit_td3_exaflops
[ "${CONDA_PREFIX:-}" = /lustre/chomeed/exaflops/envs/resfit_td3_exaflops ] || {
  echo "ERROR: failed to activate resfit_td3_exaflops env" >&2; exit 1; }
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/lustre/chomeed/exaflops/data}
OUT_ROOT=${OUT_ROOT:-/lustre/chomeed/exaflops/outputs}
MANIFEST_ROOT=${MANIFEST_ROOT:-/lustre/chomeed/exaflops/manifests}

export HF_HOME=${HF_HOME:-/lustre/chomeed/exaflops/hf_cache}
mkdir -p "$HF_HOME"

# ---------- Config ----------
BASE_CKPT=$DATA_ROOT/ckpts/rewind/rewind_state_dexmg_threading_v1/checkpoints/last/pretrained_model

# Use a separate manifest path so the K=8 baseline run's manifest
# (preference_pairs.json at clip_len=16) is preserved untouched.
PREF_MANIFEST=$MANIFEST_ROOT/rewind_state_ours_variant3_dexmg_threading/preference_pairs_clip32.json
PRECOMPUTE_NUM_EPISODES=200
PRECOMPUTE_CLIPS_PER_EP=100
PRECOMPUTE_TOPK=3
PRECOMPUTE_CLIP_LEN=32                # was 16 — must match the K=32 critical window
PRECOMPUTE_FRAME_STRIDE=1
PRECOMPUTE_SMOOTH_WINDOW=7

DEXMG_CAMERA_KEYS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

SUCCESS_REPO=chomeed/dexmg_two_arm_threading
FAILURE_REPO=chomeed/dexmg_two_arm_threading_failures

OUTPUT_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant3_dexmg_threading_v1_wider_window
JOB_NAME=rewind_state_ours_variant3_dexmg_threading_v1_wider_window

SUCCESS_ANCHOR_MODE=random
REWARD_PREF_LOSS_WEIGHT=1.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_CRITICAL_WINDOW=32        # K = whole clip at clip_len=32
STOP_GRAD_ON_SUCCESS=True

# ---------- 1. Build the preference manifest against the base ckpt ----------
echo "[$(date)] Building preference manifest -> $PREF_MANIFEST"
mkdir -p "$(dirname "$PREF_MANIFEST")"
python -m lerobot_policy_rewind_state_ours_variant3.common.precompute_preference_pairs \
  --base_ckpt          "$BASE_CKPT" \
  --success_repo_id    "$SUCCESS_REPO" \
  --failure_repo_id    "$FAILURE_REPO" \
  --camera_keys        "${DEXMG_CAMERA_KEYS[@]}" \
  --num_episodes       "$PRECOMPUTE_NUM_EPISODES" \
  --clip_len           "$PRECOMPUTE_CLIP_LEN" \
  --frame_stride       "$PRECOMPUTE_FRAME_STRIDE" \
  --smooth_window      "$PRECOMPUTE_SMOOTH_WINDOW" \
  --success_clips_per_ep "$PRECOMPUTE_CLIPS_PER_EP" \
  --top_k              "$PRECOMPUTE_TOPK" \
  --success_anchor_mode "$SUCCESS_ANCHOR_MODE" \
  --match_dtw_weight 0.3 --match_state_weight 0.4 --match_window 16 \
  --output_manifest    "$PREF_MANIFEST"

# ---------- 2. Train ----------
echo "[$(date)] Launching training"
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lerobot_policy_rewind_state_ours_variant3.common.train_rewind_state_ours_robomimic_variant3 \
  --policy.type=rewind_state_ours_variant3 \
  --policy.repo_id=chomeed/rewind_state_ours_variant3_dexmg_threading_v1_wider_window \
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
  --policy.reward_pref_critical_window="$REWARD_PREF_CRITICAL_WINDOW" \
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
