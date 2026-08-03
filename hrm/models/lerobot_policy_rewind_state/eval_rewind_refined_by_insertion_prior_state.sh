#!/bin/bash
#SBATCH --job-name=rewind_state_refined
#SBATCH --time=02:00:00
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate proj-rl-finetuning

cd /home/chomeed/workspace/proj-rl-finetuning

REWIND_CKPT=outputs/train/.../checkpoints/last/pretrained_model
INSERTION_CKPT=outputs/train/.../checkpoints/last/pretrained_model

python -m lerobot_policy_rewind_state.common.eval_rewind_refined_by_insertion_prior \
    --policy.path=${REWIND_CKPT} \
    --insertion_prior_path=${INSERTION_CKPT} \
    --dataset.repo_id=chomeed/peg_insertion_success_1000_for_rewind_test \
    --dataset.video_backend=torchcodec \
    --num_workers=4 \
    --weights='[0.5]'
