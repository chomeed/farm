#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant3_dexmg_three_piece
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

cd /home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state_ours_variant3/src/lerobot_policy_rewind_state_ours_variant3

# ---------- Config ----------
BASE_CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_dexmg_three_piece_v1/checkpoints/050000/pretrained_model

# Per-task manifest folder — kept separate from the robomimic-square manifests.
PREF_MANIFEST=/scratch2/chomeed/manifests/rewind_state_ours_variant3_dexmg_three_piece/preference_pairs.json
# NOTE: precompute loads range(PRECOMPUTE_NUM_EPISODES) episodes from each repo;
# lower this if either dexmg repo has fewer episodes.
PRECOMPUTE_NUM_EPISODES=200
PRECOMPUTE_CLIPS_PER_EP=100
PRECOMPUTE_TOPK=3
PRECOMPUTE_CLIP_LEN=16
PRECOMPUTE_FRAME_STRIDE=1
PRECOMPUTE_SMOOTH_WINDOW=7

# dexmg 3-camera observation keys.
DEXMG_CAMERA_KEYS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

SUCCESS_REPO=chomeed/dexmg_two_arm_three_piece_assembly
FAILURE_REPO=chomeed/dexmg_two_arm_three_piece_assembly_failures

OUTPUT_DIR=/scratch2/chomeed/outputs/rewind/rewind_state_ours_variant3_dexmg_three_piece_v1
JOB_NAME=rewind_state_ours_variant3_dexmg_three_piece_v1

# Variant 3 knobs
SUCCESS_ANCHOR_MODE=random          # 'critical' (Variant 3) or 'random' (base ours)
REWARD_PREF_LOSS_WEIGHT=1.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_CRITICAL_WINDOW=8         # K — last K frames of each clip
STOP_GRAD_ON_SUCCESS=True             # detach succ_tail — BT only suppresses failure

# ---------- 1. Build the preference manifest against the base ckpt ----------
# Always regenerate so the manifest stays tied to the current $BASE_CKPT.
# (Comment out this block to reuse an existing manifest.)
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
  --policy.repo_id=chomeed/rewind_state_ours_variant3_dexmg_three_piece_v1 \
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
