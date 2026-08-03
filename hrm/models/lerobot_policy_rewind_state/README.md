# lerobot_policy_rewind_state

Sibling of `lerobot_policy_rewind` that consumes proprioceptive state in
addition to multi-camera vision and language. The fusion follows a SARM-style
per-modality MLP projection (`x_i = MLP(v_i) + MLP(s_i) + MLP(g)`) but
preserves the existing `(N+1)*T` transformer token structure: state is summed
additively into every per-frame token (cameras and language), and visual MLPs
are per-camera.

## Training

```bash
conda activate proj-rl-finetuning

python -m lerobot_policy_rewind_state.common.train_rewind_state \
  --policy.type=rewind_state \
  --policy.num_cameras=2 \
  --policy.state_dim=19 \
  --policy.finetune_encoder=true \
  --batch_size=16 \
  --steps=5_000 \
  --dataset.repo_id=chomeed/peg_insertion_success_1000_for_rewind_train \
  --val.repo_id=chomeed/peg_insertion_success_1000_for_rewind_val \
  --wandb.enable=True \
  --wandb.project=long_horizon_reward_modeling \
  --job_name=rewind_state_peg_insertion \
  --save_checkpoint=True
```

`--policy.state_dim` must match the dataset's `observation.state` shape
(19 for the peg insertion task).

## Evaluation and visualization

Same entrypoints as `lerobot_policy_rewind` but routed through this package:

- `python -m lerobot_policy_rewind_state.common.eval_rewind_state`
- `python -m lerobot_policy_rewind_state.common.eval_rewind_reward_visualization`
- `python -m lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense`
- `python -m lerobot_policy_rewind_state.common.eval_rewind_refined_by_insertion_prior`

## RABC relabeling

The existing tool routes via `--policy.type=rewind_state`:

```bash
python -m hrm.datasets.datasets_tool.add_rewind_rabc_to_lerobot \
  --policy.type=rewind_state \
  --policy.pretrained_path=/path/to/checkpoint/pretrained_model \
  --policy.num_cameras=2 \
  --policy.state_dim=19 \
  --dataset.repo_id=...
```

python -m hrm.datasets.datasets_tool.add_rewind_state_rabc_to_lerobot \
  --policy.type=rewind_state \
  --policy.pretrained_path=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_peg_insertion_v2/checkpoints/last/pretrained_model \
  --policy.repo_id=chomeed/rewind_state_peg_insertion_v2 \
  --policy.push_to_hub=False \
  --policy.num_cameras=2 \
  --policy.state_dim=19 \
  --policy.finetune_encoder=True \
  --policy.finetune_encoder_blocks=2 \
  --dataset.repo_id=chomeed/peg_insertion_full_500_unprivileged_fixed_v2_lerobot_success \
  --num_workers=4
  
**DINOv2 fintuned**
python -m hrm.datasets.datasets_tool.add_rewind_rabc_to_lerobot \
  --policy.type=rewind \
  --policy.repo_id=chomeed/testing \
  --policy.pretrained_path=/home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind/src/lerobot_policy_rewind/outputs/train/2026-04-23/04-01-03_rewind_peg_insertion_v2/checkpoints/last/pretrained_model \
  --policy.finetune_encoder=true \
  --policy.finetune_encoder_blocks=2 \
  --policy.num_cameras=2 \
  --dataset.repo_id=chomeed/peg_insertion_full_500_unprivileged_fixed_v2_lerobot_success \
  --num_workers=4


## Training 

python -m lerobot_policy_rewind_state.common.train_rewind_state \
  --policy.type=rewind_state \
  --policy.repo_id=chomeed/rewind_state_peg_insertion_v2 \
  --policy.push_to_hub=False \
  --policy.num_cameras=2 \
  --policy.finetune_encoder=True \
  --policy.finetune_encoder_blocks=2 \
  --policy.state_dim=19 \
  --batch_size=16 --num_workers=16 --steps=10_000 --log_freq=100 \
  --eval_freq=1000 --save_freq=1000 --wandb.enable=True \
  --save_checkpoint=True \
  --job_name=rewind_state_peg_insertion_v2 \
  --policy.scheduler_decay_steps=10_000 \
  --policy.scheduler_peak_lr=5e-5 \
  --dataset.repo_id=chomeed/peg_insertion_full_500_unprivileged_fixed_v2_lerobot_success_for_rewind_train \
  --val.repo_id=chomeed/peg_insertion_full_500_unprivileged_fixed_v2_lerobot_success_for_rewind_train
