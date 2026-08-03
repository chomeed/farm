#!/bin/bash
#SBATCH --job-name=rabc_variants_topreward
#SBATCH --time=06:00:00
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Enable strict mode AFTER sourcing bashrc.
set -euo pipefail

cd /home/chomeed/workspace/proj-rl-finetuning

# ---------- Config ----------
DATASET=chomeed/robomimic_square_resized_peg_success

BASE_CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model
OURS_CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_robomimic_square_v1_eps50_clip20/checkpoints/010000/pretrained_model

DELTA=8
KAPPA=0.04          # carried into the output name only; ignored by --top_reward=true
TAU=2.0             # TOPReward (arxiv 2602.19313, Eq. 3)
DELTA_MAX=2.0
NUM_WORKERS=4

SHARED_FLAGS=(
  --policy.push_to_hub=False
  --policy.num_cameras=2
  --policy.finetune_encoder=true
  --policy.finetune_encoder_blocks=2
  --policy.state_dim=9
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.eye_in_hand"]'
  --dataset.repo_id="$DATASET"
  --delta="$DELTA"
  --kappa="$KAPPA"
  --top_reward=true
  --tau="$TAU"
  --delta_max="$DELTA_MAX"
  --num_workers="$NUM_WORKERS"
)

# ---------- 1. rewind_state (base) -> _using_rewind_TOPReward ----------
echo "[$(date)] (1/2) base rewind_state -> ${DATASET}_rabc_delta${DELTA}_kappa${KAPPA}_using_rewind_TOPReward"
python -m hrm.datasets.datasets_tool.add_rewind_state_rabc_to_lerobot \
  --policy.type=rewind_state \
  --policy.pretrained_path="$BASE_CKPT" \
  --suffix=TOPReward \
  "${SHARED_FLAGS[@]}"

# ---------- 2. rewind_state_ours -> _using_rewind_ours_TOPReward ----------
echo "[$(date)] (2/2) rewind_state_ours -> ${DATASET}_rabc_delta${DELTA}_kappa${KAPPA}_using_rewind_ours_TOPReward"
python -m hrm.datasets.datasets_tool.add_rewind_state_rabc_to_lerobot \
  --policy.type=rewind_state_ours \
  --policy.pretrained_path="$OURS_CKPT" \
  --suffix=ours_TOPReward \
  "${SHARED_FLAGS[@]}"

echo "[$(date)] Done."
echo "  ${DATASET}_rabc_delta${DELTA}_kappa${KAPPA}_using_rewind_TOPReward"
echo "  ${DATASET}_rabc_delta${DELTA}_kappa${KAPPA}_using_rewind_ours_TOPReward"
