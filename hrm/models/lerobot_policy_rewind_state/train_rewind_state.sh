#!/bin/bash
#SBATCH --job-name=rewind_peg_insertion_ft_v2
#SBATCH --time=48:00:00
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate proj-rl-finetuning

cd /home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state/src/lerobot_policy_rewind_state


accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lerobot_policy_rewind_state.common.train_rewind_state \
  --policy.type=rewind_state \
  --policy.repo_id=chomeed/rewind_state_peg_insertion_v2 \
  --policy.push_to_hub=False \
  --policy.num_cameras=2 \
  --policy.finetune_encoder=True \
  --policy.finetune_encoder_blocks=2 \
  --policy.state_dim=19 \
  --batch_size=4 --num_workers=8 --steps=10_000 --log_freq=100 \
  --eval_freq=1000 --save_freq=1000 --wandb.enable=True \
  --wandb.project=long_horizon_reward_modeling \
  --save_checkpoint=True \
  --output_dir=/scratch2/chomeed/outputs/rewind/rewind_state_peg_insertion_v2 \
  --job_name=rewind_state_peg_insertion_v2 \
  --policy.scheduler_decay_steps=10_000 \
  --policy.scheduler_peak_lr=5e-5 \
  --dataset.repo_id=chomeed/peg_insertion_full_500_unprivileged_fixed_v2_lerobot_success_for_rewind_train \
  --val.repo_id=chomeed/peg_insertion_full_500_unprivileged_fixed_v2_lerobot_success_for_rewind_train
