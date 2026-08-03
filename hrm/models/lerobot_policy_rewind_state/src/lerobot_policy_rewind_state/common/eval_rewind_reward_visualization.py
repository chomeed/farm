import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import accelerate

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

from lerobot_policy_rewind_state.common.factory import make_eval_dataset


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _to_episode_index_list(x) -> list[int]:
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().cpu().tolist()]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def _to_cpu_2d_reward_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    x = x.detach().float().cpu()

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 2:
        raise ValueError(f"Expected reward tensor shape [B, T] or [B, T, 1], got {tuple(x.shape)}")

    return x


@torch.no_grad()
def collect_reward_predictions(
    dataloader: DataLoader,
    reward_model,
    accelerator: accelerate.Accelerator | None = None,
    preprocessor=None,
):
    assert preprocessor is not None, "preprocessor must be provided"

    episode_to_pred_rewards = {}
    episode_to_gt_reward = {}

    if accelerator is not None:
        device = accelerator.device
    else:
        device = next(reward_model.parameters()).device

    for batch in tqdm(dataloader, desc="Collecting reward trajectories"):
        batch = _move_batch_to_device(batch, device)
        batch = preprocessor(batch)

        pred_reward = reward_model.predict_reward(batch)
        pred_reward = _to_cpu_2d_reward_tensor(pred_reward)

        gt_reward = None
        if "reward" in batch:
            gt_reward = _to_cpu_2d_reward_tensor(batch["reward"])

        episode_indices = _to_episode_index_list(batch["episode_index"])
        batch_size = pred_reward.shape[0]

        if len(episode_indices) != batch_size:
            raise ValueError(
                f"episode_index length {len(episode_indices)} does not match batch size {batch_size}"
            )

        for i in range(batch_size):
            ep_idx = int(episode_indices[i])
            episode_to_pred_rewards[ep_idx] = pred_reward[i].flatten().tolist()

            if gt_reward is not None:
                episode_to_gt_reward[ep_idx] = gt_reward[i].flatten().tolist()

    return episode_to_pred_rewards, episode_to_gt_reward


def plot_reward_trajectories(
    episode_to_pred_rewards,
    episode_to_gt_reward=None,
    num_episodes_to_plot: int = 5,
    save_path: str = "reward_trajectories.png",
):
    episode_ids = sorted(episode_to_pred_rewards.keys())[:num_episodes_to_plot]
    if len(episode_ids) == 0:
        print("No trajectories to plot.")
        return

    plt.figure(figsize=(15, 3 * len(episode_ids)))

    for row, ep_idx in enumerate(episode_ids, start=1):
        plt.subplot(len(episode_ids), 1, row)

        pred = np.asarray(episode_to_pred_rewards[ep_idx], dtype=np.float32)
        plt.plot(pred, label="pred_reward")

        if episode_to_gt_reward is not None and ep_idx in episode_to_gt_reward:
            gt = np.asarray(episode_to_gt_reward[ep_idx], dtype=np.float32)
            plt.plot(gt, label="gt_reward")

        plt.title(f"Episode {ep_idx} Reward Trajectory")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


"""
python -m lerobot_policy_rewind.common.eval_rewind \
  --policy.type rewind \
  --policy.pretrained_path /path/to/checkpoint/pretrained_model \
  --dataset.repo_id chomeed/anubis_pick_carrot_out_of_pot_task_completion_for_rewind_same \
  --dataset.video_backend torchcodec \
  --batch_size 2 \
  --num_workers 16
"""
@parser.wrap()
def infer_reward_trajectory_main(cfg: TrainPipelineConfig):
    accelerator = accelerate.Accelerator()

    dataset = make_eval_dataset(cfg)
    print(f"Dataset loaded. Num episodes: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers if cfg.num_workers is not None else 4,
        pin_memory=True,
    )

    ds_meta = dataset.meta if hasattr(dataset, "meta") else dataset._datasets[0].meta

    reward_model = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=ds_meta.stats if hasattr(ds_meta, "stats") else None,
    )

    print("Model loaded successfully")

    dataloader, reward_model = accelerator.prepare(dataloader, reward_model)

    print("Start inference...")
    episode_to_pred_rewards, episode_to_gt_reward = collect_reward_predictions(
        dataloader=dataloader,
        reward_model=reward_model,
        accelerator=accelerator,
        preprocessor=preprocessor,
    )

    if accelerator.is_main_process:
        plot_reward_trajectories(
            episode_to_pred_rewards=episode_to_pred_rewards,
            episode_to_gt_reward=None,
            num_episodes_to_plot=5,
            save_path="reward_trajectories.png",
        )


def main():
    infer_reward_trajectory_main()


if __name__ == "__main__":
    main()