#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant8_dexmg_three_piece
#SBATCH --time=12:00:00
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/exaflops/chomeed/rewind_state_ours_variant8_dexmg_three_piece_%j.out

# Variant 8 = variant 6 + `mono_skip_pre_critical_anchors` (skip mono-decrease
# hinge on the leading hybrid anchors that overlap the pre-critical "still
# on-task" region, so the success per-frame supervision can keep those
# anchors at a high plateau via generalization).
#
# The new flag is ONLY meaningful in hybrid failure-sampling mode (in
# manifest mode the failure clip is a single 16-frame window, no anchor
# dim to skip on), so this script explicitly sets failure_sampling_mode=hybrid.
#
# Submit from the exaflops account with: sbatch --export=NONE <this file>
# (`--export=NONE` is required: Claude Code's auto-activated venv leaks PATH
#  into the job and breaks conda activation.)

# Activate the /lustre prefix env directly via PATH (no `conda` needed).
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
BASE_CKPT=$DATA_ROOT/ckpts/rewind/rewind_state_dexmg_three_piece_v1/checkpoints/last/pretrained_model

# Existing leaving-only manifest (failure_clip_directions=['leaving']),
# precomputed with pre_critical_frames=5, clip_len=16, frame_stride=1 —
# 2 anchors before crit_t per failure clip (matches mono_skip_pre_critical_anchors=2).
PREF_MANIFEST=$MANIFEST_ROOT/dexmg_two_arm_three_piece_assembly/preference_pairs_blend.json

PRECOMPUTE_CLIP_LEN=16
PRECOMPUTE_FRAME_STRIDE=1
PRECOMPUTE_TOPK=3

DEXMG_CAMERA_KEYS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

SUCCESS_REPO=chomeed/dexmg_two_arm_three_piece_assembly
FAILURE_REPO=chomeed/dexmg_two_arm_three_piece_assembly_failures

OUTPUT_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant8_dexmg_three_piece_v1
JOB_NAME=rewind_state_ours_variant8_dexmg_three_piece_v1

# Variant 6 inherited knobs
SUCCESS_ANCHOR_MODE=random
REWARD_PREF_LOSS_WEIGHT=1.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85
STOP_GRAD_ON_SUCCESS=True

# Variant 8 knobs
FAILURE_SAMPLING_MODE=hybrid
# Number of leading hybrid anchors that fall before critical_t under the
# default precompute schedule (pre_critical_frames=5, max_post=60, clip_len=16):
# np.linspace(crit_t-5, crit_t+60, 16) puts anchors 0 (-5) and 1 (-1) before
# crit_t, anchor 2 at +4. If you changed precompute window/clip_len, recompute.
MONO_SKIP_PRE_CRITICAL_ANCHORS=2

# ---------- 1. Manifest sanity check ----------
if [ ! -f "$PREF_MANIFEST" ]; then
  echo "ERROR: expected preference manifest at $PREF_MANIFEST" >&2
  echo "       Generate it via the variant6 three_piece script's precompute" >&2
  echo "       block first, then re-run this script." >&2
  exit 1
fi
echo "[$(date)] Using preference manifest: $PREF_MANIFEST"

# ---------- 2. Train ----------
echo "[$(date)] Launching training (max 2k steps, save every 400)"
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lerobot_policy_rewind_state_ours_variant8.common.train_rewind_state_ours_robomimic_variant8 \
  --policy.type=rewind_state_ours_variant8 \
  --policy.repo_id=chomeed/rewind_state_ours_variant8_dexmg_three_piece_v1 \
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
  --batch_size=4 --num_workers=8 --steps=2000 --log_freq=100 \
  --eval_freq=400 --save_freq=400 --wandb.enable=True \
  --wandb.project=long_horizon_reward_modeling \
  --save_checkpoint=True \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.scheduler_decay_steps=2000 \
  --policy.scheduler_peak_lr=5e-5 \
  --dataset.repo_id="$SUCCESS_REPO" \
  --val.repo_id="$SUCCESS_REPO"

echo "[$(date)] Done."
