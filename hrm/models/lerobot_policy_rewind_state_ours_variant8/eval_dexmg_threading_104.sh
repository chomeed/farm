#!/bin/bash
# Evaluate the variant8 threading reward model at step 002000 on the dexmg
# threading SUCCESS + FAILURE datasets, then render side-by-side reward videos.
# Adapted from variant6 eval_4_anchorstack_dexmg_threading_step2k.sh (single run).
#
# Edit partition/qos for 104 or submit with sbatch --partition=<p> --qos=<q>.
#SBATCH --job-name=rewind_variant8_threading_eval_2k
#SBATCH --time=04:00:00
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
OUT_ROOT=${OUT_ROOT:-/data4/chomeed/outputs}
CKPT_ROOT=${CKPT_ROOT:-$OUT_ROOT/rewind}

SUCCESS_REPO=chomeed/dexmg_two_arm_threading
FAILURE_REPO=chomeed/dexmg_two_arm_threading_failures

DEXMG_CAMERAS=(
  observation.images.agentview
  observation.images.robot0_eye_in_hand
  observation.images.robot1_eye_in_hand
)

STEP=002000
NUM_VAL_EPISODES=50
NUM_VISUALIZE=10
SAMPLING=hybrid
N_RECENT=8
EVAL_SUBDIR="eval_step${STEP}_hybrid${N_RECENT}"

RUN=rewind_state_ours_variant8_dexmg_threading_v1
CKPT="$CKPT_ROOT/$RUN/checkpoints/$STEP/pretrained_model"
OUT="$CKPT_ROOT/$RUN/$EVAL_SUBDIR"

if [ ! -d "$CKPT" ]; then
  echo "ERROR: no step $STEP checkpoint at $CKPT" >&2
  echo "       Run the reward-model training script (Task 1) first." >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")"

echo "[$(date)] Eval $RUN -> $EVAL_SUBDIR"
python -m lerobot_policy_rewind_state_ours_variant8.common.eval_val_loss_and_visualize \
  --ckpt              "$CKPT" \
  --success_repo_id   "$SUCCESS_REPO" \
  --failure_repo_id   "$FAILURE_REPO" \
  --camera_keys       "${DEXMG_CAMERAS[@]}" \
  --num_val_episodes  "$NUM_VAL_EPISODES" \
  --num_visualize     "$NUM_VISUALIZE" \
  --sampling          "$SAMPLING" \
  --n_recent          "$N_RECENT" \
  --output_dir        "$OUT"

# ---------- Render comparison video (single model) ----------
VIDEO_OUT="$CKPT_ROOT/_compare_variant8_threading_step${STEP}_hybrid${N_RECENT}"
NUM_RENDER_EPISODES=10
FPS=20
PLOT_H=360

echo "[$(date)] Rendering side-by-side videos -> $VIDEO_OUT"
python -m lerobot_policy_rewind_state_ours_variant8.common.render_compare_videos \
  --success_repo_id "$SUCCESS_REPO" \
  --failure_repo_id "$FAILURE_REPO" \
  --splits          "success,failure" \
  --num_episodes    "$NUM_RENDER_EPISODES" \
  --fps             "$FPS" \
  --plot_height     "$PLOT_H" \
  --cameras         "$(IFS=,; echo "${DEXMG_CAMERAS[*]}")" \
  --output_dir      "$VIDEO_OUT" \
  --model "v8:$OUT"

echo "[$(date)] Done. Eval at $OUT ; videos under $VIDEO_OUT/videos/{success,failure}/"
