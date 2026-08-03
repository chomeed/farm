#!/bin/bash
#SBATCH --job-name=rabc_variants_using_rewind_ours
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
OURS_CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_variant2_robomimic_square_v1_eps200_clip100/checkpoints/last/pretrained_model

DELTA=8
NUM_WORKERS=4

# Shared policy flags (must match the architecture the checkpoint was trained with).
POLICY_FLAGS=(
  --policy.type=rewind_state_ours
  --policy.pretrained_path="$OURS_CKPT"
  --policy.push_to_hub=False
  --policy.num_cameras=2
  --policy.finetune_encoder=true
  --policy.finetune_encoder_blocks=2
  --policy.state_dim=9
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.eye_in_hand"]'
)

# ---------- 1. delta=8, kappa=0.04 ----------  (skipped — uncomment to regenerate)
# echo "[$(date)] (1/2) delta=$DELTA kappa=0.04 -> ${DATASET}_rabc_delta${DELTA}_kappa0.04_using_rewind_ours"
# python -m hrm.datasets.datasets_tool.add_rewind_state_rabc_to_lerobot \
#   "${POLICY_FLAGS[@]}" \
#   --dataset.repo_id="$DATASET" \
#   --delta="$DELTA" \
#   --kappa=0.04 \
#   --suffix=ours \
#   --num_workers="$NUM_WORKERS"

# ---------- 2. delta=8, kappa=0.1 ----------
echo "[$(date)] (2/2) delta=$DELTA kappa=0.1 -> ${DATASET}_rabc_delta${DELTA}_kappa0.1_using_rewind_ours"
python -m hrm.datasets.datasets_tool.add_rewind_state_rabc_to_lerobot \
  "${POLICY_FLAGS[@]}" \
  --dataset.repo_id="$DATASET" \
  --delta="$DELTA" \
  --kappa=0.1 \
  --suffix=ours \
  --num_workers="$NUM_WORKERS"

echo "[$(date)] Done."
echo "  ${DATASET}_rabc_delta${DELTA}_kappa0.04_using_rewind_ours"
echo "  ${DATASET}_rabc_delta${DELTA}_kappa0.1_using_rewind_ours"
