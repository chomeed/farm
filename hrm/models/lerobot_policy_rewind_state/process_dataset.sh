#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --time=10:00:00
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate proj-rl-finetuning

cd /home/chomeed/workspace/proj-rl-finetuning

python hrm/models/lerobot_policy_rewind/src/utils/process_dataset.py --repo_id chomeed/peg_insertion_full_500_unprivileged_fixed_v2_lerobot_failure
# python hrm/models/lerobot_policy_rewind/src/utils/process_dataset.py --repo_id chomeed/peg_insertion_failure_500
# python hrm/models/lerobot_policy_rewind/src/utils/process_dataset.py --repo_id chomeed/anubis_pick_carrot_out_of_pot_randomized_train


