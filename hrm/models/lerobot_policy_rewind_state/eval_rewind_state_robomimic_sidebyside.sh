#!/bin/bash
#SBATCH --job-name=rewind_state_robomimic_sbs
#SBATCH --time=03:00:00
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/chomeed/workspace/proj-rl-finetuning

CKPT=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model
OUT_DIR=videos_sidebyside_robomimic_square
NUM_EP=5
FPS=20

mkdir -p ${OUT_DIR}

# Success episodes
python -m lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense \
  --policy.type=rewind_state \
  --policy.pretrained_path=${CKPT} \
  --policy.finetune_encoder=True \
  --policy.num_cameras=2 \
  --policy.state_dim=9 \
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.eye_in_hand"]' \
  --dataset.repo_id=chomeed/robomimic_square_resized_peg_success \
  --dataset.video_backend=torchcodec \
  --num_workers=2 \
  --num_episodes=${NUM_EP} \
  --render_sidebyside_videos=True \
  --video_save_dir=${OUT_DIR} \
  --video_fps=${FPS} \
  --output_tag=robomimic_square_success

# Failure episodes
python -m lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense \
  --policy.type=rewind_state \
  --policy.pretrained_path=${CKPT} \
  --policy.finetune_encoder=True \
  --policy.num_cameras=2 \
  --policy.state_dim=9 \
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.eye_in_hand"]' \
  --dataset.repo_id=chomeed/robomimic_square_resized_peg_failure \
  --dataset.video_backend=torchcodec \
  --num_workers=2 \
  --num_episodes=${NUM_EP} \
  --render_sidebyside_videos=True \
  --video_save_dir=${OUT_DIR} \
  --video_fps=${FPS} \
  --output_tag=robomimic_square_failure
