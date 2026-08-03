#!/bin/bash
#SBATCH --job-name=rewind_state_ours_variant5_dexmg
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/exaflops/chomeed/rewind_state_ours_variant5_dexmg_%A_%a.out

# EXAFLOPS-ACCOUNT variant5 reward-model training array. Replaces the two
# per-task scripts' inline-precompute pattern: manifests come from
# precompute_preference_pairs_dexmg_exaflops.sh, which writes
#   $MANIFEST_ROOT/dexmg_two_arm_<task>/preference_pairs_<mode>.json
# for blend + forwardbackward. One array task per (task, mode) on its own A100.
#
# Submit (exaflops account):
#   sbatch hrm/models/lerobot_policy_rewind_state_ours_variant5/train_rewind_state_ours_variant5_dexmg_exaflops.sh
# Single-variant manual rerun (no slurm):
#   SLURM_ARRAY_TASK_ID=1 bash .../train_rewind_state_ours_variant5_dexmg_exaflops.sh
# Dry-run command assembly (no conda, no python -- safe on dev workstation):
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash .../train_rewind_state_ours_variant5_dexmg_exaflops.sh

DRY_RUN=${DRY_RUN:-}
if [ -z "$DRY_RUN" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /lustre/chomeed/exaflops/envs/resfit_td3_exaflops
  [ "${CONDA_PREFIX:-}" = /lustre/chomeed/exaflops/envs/resfit_td3_exaflops ] || {
    echo "ERROR: failed to activate resfit_td3_exaflops env" >&2; exit 1; }
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/lustre/chomeed/exaflops/data}
# Ckpts land on /scratch2 (heavy: 100k step ckpts every 1k). Data + manifests
# stay on /lustre.
OUT_ROOT=${OUT_ROOT:-/scratch2/exaFLOPs/chomeed/outputs}
MANIFEST_ROOT=${MANIFEST_ROOT:-/lustre/chomeed/exaflops/manifests}
export HF_HOME=${HF_HOME:-/lustre/chomeed/exaflops/hf_cache}
mkdir -p "$HF_HOME"

# ---- Trainer-side knobs (must match the manifest's clip shape) -------------
# The standalone precompute writes manifests with clip_len=16, frame_stride=1,
# top_k=3; the trainer needs the same values to consume them.
PRECOMPUTE_CLIP_LEN=16
PRECOMPUTE_FRAME_STRIDE=1
PRECOMPUTE_TOPK=3

# ---- Variant 3 loss knobs --------------------------------------------------
REWARD_PREF_LOSS_WEIGHT=1.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_CRITICAL_WINDOW=8         # K -- last K frames of each clip
STOP_GRAD_ON_SUCCESS=True             # detach succ_tail -- BT only suppresses failure

# ---- Array dispatch table --------------------------------------------------
# Parallel arrays indexed by SLURM_ARRAY_TASK_ID. The MANIFEST path uses the
# full HF task name (two_arm_threading, two_arm_three_piece_assembly); the
# OUTPUT_DIR/JOB_NAME/policy.repo_id use the short ckpt-tree token
# (threading, three_piece).
MANIFEST_TASK_DIRS=(
  dexmg_two_arm_threading
  dexmg_two_arm_threading
  dexmg_two_arm_three_piece_assembly
  dexmg_two_arm_three_piece_assembly
)
TASK_TOKS=(
  threading
  threading
  three_piece
  three_piece
)
SUCCESS_REPOS=(
  chomeed/dexmg_two_arm_threading
  chomeed/dexmg_two_arm_threading
  chomeed/dexmg_two_arm_three_piece_assembly
  chomeed/dexmg_two_arm_three_piece_assembly
)
MODES=(
  blend
  forwardbackward
  blend
  forwardbackward
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-}
if [ -z "$TASK_ID" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is unset. Submit via sbatch (4 array tasks),"
  echo "       or set it manually 0..3 to run one variant:"
  echo "         SLURM_ARRAY_TASK_ID=2 bash $0"
  exit 1
fi
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#MODES[@]}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID out of range 0..$((${#MODES[@]} - 1))"
  exit 1
fi

MANIFEST_TASK_DIR=${MANIFEST_TASK_DIRS[$TASK_ID]}
TASK_TOK=${TASK_TOKS[$TASK_ID]}
SUCCESS_REPO=${SUCCESS_REPOS[$TASK_ID]}
MODE=${MODES[$TASK_ID]}

BASE_CKPT=$DATA_ROOT/ckpts/rewind/rewind_state_dexmg_${TASK_TOK}_v1/checkpoints/last/pretrained_model
PREF_MANIFEST=$MANIFEST_ROOT/$MANIFEST_TASK_DIR/preference_pairs_${MODE}.json
OUTPUT_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant5_dexmg_${TASK_TOK}_v1_${MODE}
JOB_NAME=rewind_state_ours_variant5_dexmg_${TASK_TOK}_v1_${MODE}
POLICY_REPO_ID=chomeed/rewind_state_ours_variant5_dexmg_${TASK_TOK}_v1_${MODE}

CMD=(accelerate launch
  --num_processes=1
  --num_machines=1
  --mixed_precision=bf16
  -m lerobot_policy_rewind_state_ours_variant5.common.train_rewind_state_ours_robomimic_variant5
  --policy.type=rewind_state_ours_variant5
  --policy.repo_id="$POLICY_REPO_ID"
  --policy.push_to_hub=False
  --policy.num_cameras=3
  --policy.finetune_encoder=True
  --policy.finetune_encoder_blocks=2
  --policy.state_dim=18
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.robot0_eye_in_hand","observation.images.robot1_eye_in_hand"]'
  --policy.pretrained_path="$BASE_CKPT"
  --policy.preference_pairs_path="$PREF_MANIFEST"
  --policy.preference_clip_len="$PRECOMPUTE_CLIP_LEN"
  --policy.preference_frame_stride="$PRECOMPUTE_FRAME_STRIDE"
  --policy.preference_topk="$PRECOMPUTE_TOPK"
  --policy.reward_pref_loss_weight="$REWARD_PREF_LOSS_WEIGHT"
  --policy.reward_pref_warmup_steps="$REWARD_PREF_WARMUP_STEPS"
  --policy.reward_pref_critical_window="$REWARD_PREF_CRITICAL_WINDOW"
  --policy.stop_grad_on_success="$STOP_GRAD_ON_SUCCESS"
  --batch_size=4 --num_workers=8 --steps=100_000 --log_freq=100
  --eval_freq=1000 --save_freq=1000 --wandb.enable=True
  --wandb.project=long_horizon_reward_modeling
  --save_checkpoint=True
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --policy.scheduler_decay_steps=100_000
  --policy.scheduler_peak_lr=5e-5
  --dataset.repo_id="$SUCCESS_REPO"
  --val.repo_id="$SUCCESS_REPO"
)

if [ -n "$DRY_RUN" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

if [ -d "$OUTPUT_DIR/checkpoints/last/pretrained_model" ]; then
  echo "[skip] $TASK_TOK/$MODE - trained ckpt already exists at $OUTPUT_DIR"
  exit 0
fi

if [ ! -f "$PREF_MANIFEST" ]; then
  echo "ERROR: preference manifest not found at $PREF_MANIFEST" >&2
  echo "       Run precompute_preference_pairs_dexmg_exaflops.sh first." >&2
  exit 1
fi

# Don't `mkdir -p "$OUTPUT_DIR"` here -- LeRobot's cfg.validate() (train.py)
# rejects a pre-existing output_dir unless --resume is set. The trainer creates
# the dir itself on first write.

echo "[$(date)] task $TASK_ID: $TASK_TOK / $MODE"
echo "  base ckpt:  $BASE_CKPT"
echo "  manifest:   $PREF_MANIFEST"
echo "  output:     $OUTPUT_DIR"

"${CMD[@]}"

echo "[$(date)] task $TASK_ID: $TASK_TOK / $MODE done -> $OUTPUT_DIR"
