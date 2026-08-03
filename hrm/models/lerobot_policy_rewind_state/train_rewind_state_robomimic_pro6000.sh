#!/bin/bash
#SBATCH --job-name=rewind_state_robomimic_square_v1_pro6000
#SBATCH --time=48:00:00
#SBATCH --partition=suma_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state/src/lerobot_policy_rewind_state

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lerobot_policy_rewind_state.common.train_rewind_state_robomimic \
  --policy.type=rewind_state \
  --policy.repo_id=chomeed/rewind_state_robomimic_square_v1_pro6000 \
  --policy.push_to_hub=False \
  --policy.num_cameras=2 \
  --policy.finetune_encoder=True \
  --policy.finetune_encoder_blocks=2 \
  --policy.state_dim=9 \
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.eye_in_hand"]' \
  --batch_size=4 --num_workers=8 --steps=10_000 --log_freq=100 \
  --eval_freq=1000 --save_freq=1000 --wandb.enable=True \
  --wandb.project=long_horizon_reward_modeling \
  --save_checkpoint=True \
  --output_dir=/scratch2/chomeed/outputs/rewind/rewind_state_robomimic_square_v1_pro6000 \
  --job_name=rewind_state_robomimic_square_v1_pro6000 \
  --policy.scheduler_decay_steps=10_000 \
  --policy.scheduler_peak_lr=5e-5 \
  --dataset.repo_id=chomeed/robomimic_square_resized_peg_success \
  --val.repo_id=chomeed/robomimic_square_resized_peg_success
