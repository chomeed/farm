#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant6_robomimic_square_resized_peg_blend
#SBATCH --time=12:00:00
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=rewind_state_ours_variant6_robomimic_square_resized_peg_blend_%j.out

# EXAFLOPS-ACCOUNT variant-6 reward-model training for the modified
# robomimic_square (resized peg + narrow init). Mono-only loss config
# (BT off, mono on, hybrid failure sampling) matching
# train_rewind_state_ours_variant6_dexmg_threading.sh; paths and obs config
# (2 cameras, state_dim=9, robomimic_square_resized_peg datasets) match the
# earlier robomimic variant-6 scripts under this directory.
#
# Pre-flight: requires the blend manifest produced by
#   precompute_preference_pairs_robomimic_square_resized_peg_exaflops.sh

# Activate the /lustre prefix env directly via PATH (gigabyte_a6000 nodes
# lack conda on PATH and --export=NONE strips ours).
ENV_PREFIX=${ENV_PREFIX:-/lustre/chomeed/exaflops/envs/resfit_td3_exaflops}
[ -x "$ENV_PREFIX/bin/python" ] || {
  echo "ERROR: env not found at $ENV_PREFIX" >&2; exit 1; }
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"
export CONDA_DEFAULT_ENV="$(basename "$ENV_PREFIX")"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/lustre/chomeed/exaflops/data}
# OUT_ROOT defaults to a writable scratch path based on the unix uid the job
# runs as (NOT $USER — slurm can inject a different USER env var). `id -un`
# reads the effective uid. Override OUT_ROOT to bypass this detection.
if [ -z "${OUT_ROOT:-}" ]; then
  _unix_user=$(id -un)
  case "$_unix_user" in
    exaflops*) OUT_ROOT=/scratch2/exaFLOPs/chomeed/outputs ;;
    *)         OUT_ROOT=/scratch2/${_unix_user}/outputs ;;
  esac
fi
MANIFEST_ROOT=${MANIFEST_ROOT:-/lustre/chomeed/exaflops/manifests}
export HF_HOME=${HF_HOME:-/lustre/chomeed/exaflops/hf_cache}
mkdir -p "$HF_HOME"

# ---------- Config ----------
BASE_CKPT=$DATA_ROOT/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model

PREF_MANIFEST=$MANIFEST_ROOT/robomimic_square_resized_peg/preference_pairs_blend.json
PRECOMPUTE_CLIP_LEN=16
PRECOMPUTE_FRAME_STRIDE=1
PRECOMPUTE_TOPK=3

SUCCESS_REPO=chomeed/robomimic_square_resized_peg_success
FAILURE_REPO=chomeed/robomimic_square_resized_peg_failure

OUTPUT_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant6_robomimic_square_resized_peg_v1_blend_mono_anchorstack_eps0.2
JOB_NAME=rewind_state_ours_variant6_robomimic_square_resized_peg_v1_blend_mono_anchorstack_eps0.2

# Variant 6 loss — BT OFF, monotonicity hinge ON.
SUCCESS_ANCHOR_MODE=random
REWARD_PREF_LOSS_WEIGHT=0.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85
STOP_GRAD_ON_SUCCESS=False
BT_TARGET_MARGIN=2.0
MONO_LOSS_WEIGHT=1.0
MONO_WARMUP_STEPS=500
MONO_ON_FAILURE=True
MONO_ON_SUCCESS=True
MONO_STEP_EPS=0.2
TV_LOSS_WEIGHT=0.0
TV_WARMUP_STEPS=500
TV_ORDER=2
TV_ON_FAILURE=True
TV_ON_SUCCESS=True
FAILURE_SAMPLING_MODE=hybrid
FAILURE_HYBRID_N_RECENT=8
FAILURE_CLIP_DIRECTIONS="leaving"

# ---------- 1. Pre-flight ----------
if [ ! -f "$PREF_MANIFEST" ]; then
  echo "ERROR: expected preference manifest at $PREF_MANIFEST" >&2
  echo "       (run precompute_preference_pairs_robomimic_square_resized_peg{,_exaflops}.sh first)" >&2
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
  --policy.repo_id=chomeed/rewind_state_ours_variant6_robomimic_square_resized_peg_v1 \
  --policy.push_to_hub=False \
  --policy.num_cameras=2 \
  --policy.finetune_encoder=True \
  --policy.finetune_encoder_blocks=2 \
  --policy.state_dim=9 \
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.eye_in_hand"]' \
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
  --batch_size=4 --num_workers=8 --steps=30_000 --log_freq=100 \
  --eval_freq=1000 --save_freq=1000 --wandb.enable=True \
  --wandb.project=long_horizon_reward_modeling \
  --save_checkpoint=True \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.scheduler_decay_steps=30_000 \
  --policy.scheduler_peak_lr=5e-5 \
  --dataset.repo_id="$SUCCESS_REPO" \
  --val.repo_id="$SUCCESS_REPO"

echo "[$(date)] Done."
