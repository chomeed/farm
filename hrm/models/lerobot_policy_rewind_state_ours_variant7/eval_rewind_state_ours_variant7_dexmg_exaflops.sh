#!/bin/bash
#SBATCH --job-name=eval_variant7_dexmg
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/exaflops/chomeed/eval_variant7_dexmg_%A_%a.out

# EXAFLOPS-ACCOUNT variant7 reward-model eval array. Reads the train output
# ckpts produced by train_rewind_state_ours_variant7_dexmg_exaflops.sh and
# writes BT preference-scoring metrics + side-by-side reward visualization
# videos into the same v1_<mode>/ tree alongside checkpoints/.
#
# Requires the --camera_keys patch (commit 9e0aeab) so the eval entrypoint
# accepts dexmg's 3-camera obs instead of the robomimic 2-cam default. The
# patch is backward compatible -- robomimic eval scripts are unaffected.
#
# Submit (exaflops account):
#   sbatch hrm/models/lerobot_policy_rewind_state_ours_variant7/eval_rewind_state_ours_variant7_dexmg_exaflops.sh
# Single-variant manual rerun (no slurm):
#   SLURM_ARRAY_TASK_ID=1 bash .../eval_rewind_state_ours_variant7_dexmg_exaflops.sh
# Dry-run command assembly (no conda, no python -- safe on dev workstation):
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash .../eval_rewind_state_ours_variant7_dexmg_exaflops.sh

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
# Eval reads ckpts + writes eval_val_last/ under the variant7 train output
# tree, which lives on /scratch2 (see train_rewind_state_ours_variant7_dexmg_exaflops.sh).
OUT_ROOT=${OUT_ROOT:-/scratch2/exaFLOPs/chomeed/outputs}
export HF_HOME=${HF_HOME:-/lustre/chomeed/exaflops/hf_cache}
mkdir -p "$HF_HOME"

# ---- Eval hparams (match existing variant7 eval scripts) ------------------
NUM_VAL_EPISODES=50
NUM_VISUALIZE=5

# dexmg 3-camera observation keys (overrides the 2-camera robomimic default).
DEXMG_CAMERA_KEYS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

# ---- Array dispatch table -------------------------------------------------
# Parallel arrays indexed by SLURM_ARRAY_TASK_ID. CKPT and OUT_DIR derive
# from the train script's output naming: v1_<mode>/{checkpoints/last,eval_val_last}.
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
FAILURE_REPOS=(
  chomeed/dexmg_two_arm_threading_failures
  chomeed/dexmg_two_arm_threading_failures
  chomeed/dexmg_two_arm_three_piece_assembly_failures
  chomeed/dexmg_two_arm_three_piece_assembly_failures
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

TASK_TOK=${TASK_TOKS[$TASK_ID]}
SUCCESS_REPO=${SUCCESS_REPOS[$TASK_ID]}
FAILURE_REPO=${FAILURE_REPOS[$TASK_ID]}
MODE=${MODES[$TASK_ID]}

RUN_DIR=$OUT_ROOT/rewind/rewind_state_ours_variant7_dexmg_${TASK_TOK}_v1_${MODE}
CKPT=$RUN_DIR/checkpoints/last/pretrained_model
OUT_DIR=$RUN_DIR/eval_val_last

CMD=(python -m lerobot_policy_rewind_state_ours_variant7.common.eval_val_loss_and_visualize
  --ckpt              "$CKPT"
  --success_repo_id   "$SUCCESS_REPO"
  --failure_repo_id   "$FAILURE_REPO"
  --camera_keys       "${DEXMG_CAMERA_KEYS[@]}"
  --num_val_episodes  "$NUM_VAL_EPISODES"
  --num_visualize     "$NUM_VISUALIZE"
  --output_dir        "$OUT_DIR"
)

if [ -n "$DRY_RUN" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  echo "[skip] $TASK_TOK/$MODE - eval already exists at $OUT_DIR"
  exit 0
fi

if [ ! -d "$CKPT" ]; then
  echo "ERROR: trained ckpt not found at $CKPT" >&2
  echo "       Run train_rewind_state_ours_variant7_dexmg_exaflops.sh first." >&2
  exit 1
fi

echo "[$(date)] task $TASK_ID: $TASK_TOK / $MODE"
echo "  ckpt:    $CKPT"
echo "  out:     $OUT_DIR"

"${CMD[@]}"

echo "[$(date)] task $TASK_ID: $TASK_TOK / $MODE done -> $OUT_DIR"
