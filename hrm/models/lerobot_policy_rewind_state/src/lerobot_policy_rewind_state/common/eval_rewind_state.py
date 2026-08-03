import json
from collections import defaultdict
from typing import Dict, Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import accelerate

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

from lerobot_policy_rewind_state.common.factory import make_eval_dataset


def predict_stage(
    progress: Sequence[float],
    alpha: Sequence[float] = (0.2, 0.35, 0.55, 0.65, 0.8),
    beta: Sequence[float] = (0.15, 0.30, 0.50, 0.60, 0.75),
    last_window_ratio: float = 1 / 3,
    default_stage: int = 0,
) -> int:
    """
    Returns the highest satisfied stage index (1-based).
    Returns default_stage if no stage condition is satisfied.
    """
    assert progress is not None, "`progress` must be provided."
    assert len(progress) > 0, "`progress` must be non-empty."
    assert len(alpha) == len(beta), "`alpha` and `beta` must have the same length."

    T = len(progress)
    window_size = max(1, int(round(T * last_window_ratio)))
    W = progress[-window_size:]

    P_T = progress[-1]
    late_avg = sum(W) / len(W)

    satisfied_stages = [
        j
        for j, (a_j, b_j) in enumerate(zip(alpha, beta), start=1)
        if P_T >= a_j and late_avg >= b_j
    ]
    return max(satisfied_stages, default=default_stage)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
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


def compute_kendall_tau_from_final_rewards(
    episode_to_final_reward: Dict[int, float],
    episode_to_gt_stage: Dict[int, int],
) -> Dict[str, Any]:
    pred_scores = []
    gt_scores = []
    used_episode_indices = []

    for ep_idx, gt_stage in sorted(episode_to_gt_stage.items()):
        if ep_idx not in episode_to_final_reward:
            continue

        pred_scores.append(float(episode_to_final_reward[ep_idx]))
        gt_scores.append(int(gt_stage))
        used_episode_indices.append(int(ep_idx))

    tau, p_value = kendalltau(gt_scores, pred_scores)

    return {
        "kendall_tau": float(tau) if tau is not None else float("nan"),
        "p_value": float(p_value) if p_value is not None else float("nan"),
        "num_episodes_used": len(used_episode_indices),
        "episode_indices": used_episode_indices,
        "gt_scores": gt_scores,
        "pred_scores": pred_scores,
    }


def compute_task_completion_classification_score_from_json(
    episode_to_pred_stage: Dict[int, int],
    episode_to_gt_stage: Dict[int, int],
    stage_names: Dict[int, str],
) -> Dict[str, Any]:
    per_stage_total = defaultdict(int)
    per_stage_correct = defaultdict(int)

    for ep_idx, gt_stage in episode_to_gt_stage.items():
        if ep_idx not in episode_to_pred_stage:
            continue

        pred_stage = int(episode_to_pred_stage[ep_idx])
        per_stage_total[int(gt_stage)] += 1
        if pred_stage == int(gt_stage):
            per_stage_correct[int(gt_stage)] += 1

    per_stage_accuracy = {}
    for stage in sorted(stage_names):
        total = per_stage_total[stage]
        correct = per_stage_correct[stage]
        per_stage_accuracy[stage] = correct / total if total > 0 else 0.0

    num_eval = sum(per_stage_total.values())
    num_correct = sum(per_stage_correct.values())
    overall_acc = num_correct / num_eval if num_eval > 0 else 0.0

    return {
        "per_stage_total": dict(per_stage_total),
        "per_stage_correct": dict(per_stage_correct),
        "per_stage_accuracy": per_stage_accuracy,
        "overall_accuracy": overall_acc,
        "num_eval": num_eval,
        "num_correct": num_correct,
    }


@torch.no_grad()
def evaluate_episode_dataset_task_completion(
    dataloader: DataLoader,
    reward_model,
    annotation_path: str,
    accelerator: accelerate.Accelerator | None = None,
    alpha: Sequence[float] = (0.2, 0.35, 0.55, 0.65, 0.8),
    beta: Sequence[float] = (0.15, 0.30, 0.50, 0.60, 0.75),
    last_window_ratio: float = 1 / 3,
    default_stage: int = 0,
    preprocessor: Any = None,
) -> Dict[str, Any]:
    """
    Evaluates an episode-level dataset where each batch item is one episode.

    Assumptions:
      - batch["reward"] has shape [B, T]
      - batch["episode_index"] identifies the episode
      - reward_model.predict_reward(batch) returns [B, T] or [B, T, 1]
      - JSON file provides ground-truth ranking/stage via episode_to_stage
    """
    assert preprocessor is not None, "preprocessor must be provided for evaluation"

    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    episode_to_gt_stage = {int(k): int(v) for k, v in ann["episode_to_stage"].items()}
    stage_names = {int(k): v for k, v in ann["classes"].items()}

    episode_to_pred_rewards = {}
    episode_to_final_reward = {}
    episode_to_pred_stage = {}
    episode_to_gt_reward = {}

    total_abs_error = 0.0
    total_count = 0

    if accelerator is not None:
        device = accelerator.device
    else:
        device = next(reward_model.parameters()).device

    for batch in tqdm(dataloader, desc="Evaluating episodes"):
        batch = _move_batch_to_device(batch, device)

        batch = preprocessor(batch) 

        pred_reward = reward_model.predict_reward(batch)
        gt_reward = batch["reward"]

        pred_reward = _to_cpu_2d_reward_tensor(pred_reward)
        gt_reward = _to_cpu_2d_reward_tensor(gt_reward)

        if pred_reward.shape != gt_reward.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred_reward.shape)} does not match "
                f"ground-truth shape {tuple(gt_reward.shape)}"
            )

        abs_error = torch.abs(pred_reward - gt_reward)
        total_abs_error += abs_error.sum().item()
        total_count += abs_error.numel()

        episode_indices = _to_episode_index_list(batch["episode_index"])
        batch_size = pred_reward.shape[0]
        if len(episode_indices) != batch_size:
            raise ValueError(
                f"episode_index length {len(episode_indices)} does not match batch size {batch_size}"
            )

        for i in range(batch_size):
            ep_idx = int(episode_indices[i])

            pred_progress = pred_reward[i].flatten().tolist()
            gt_progress = gt_reward[i].flatten().tolist()

            final_reward = float(pred_progress[-1])
            pred_stage = predict_stage(
                progress=pred_progress,
                alpha=alpha,
                beta=beta,
                last_window_ratio=last_window_ratio,
                default_stage=default_stage,
            )

            episode_to_pred_rewards[ep_idx] = pred_progress
            episode_to_final_reward[ep_idx] = final_reward
            episode_to_pred_stage[ep_idx] = pred_stage
            episode_to_gt_reward[ep_idx] = gt_progress

    mae = total_abs_error / total_count if total_count > 0 else float("nan")

    kendall_results = compute_kendall_tau_from_final_rewards(
        episode_to_final_reward=episode_to_final_reward,
        episode_to_gt_stage=episode_to_gt_stage,
    )

    classification_results = compute_task_completion_classification_score_from_json(
        episode_to_pred_stage=episode_to_pred_stage,
        episode_to_gt_stage=episode_to_gt_stage,
        stage_names=stage_names,
    )

    return {
        "reward_mae": mae,
        "episode_to_pred_rewards": episode_to_pred_rewards,
        "episode_to_gt_reward": episode_to_gt_reward,
        "episode_to_final_reward": episode_to_final_reward,
        "episode_to_pred_stage": episode_to_pred_stage,
        "episode_to_gt_stage": episode_to_gt_stage,
        "kendall": kendall_results,
        "classification": classification_results,
        "stage_names": stage_names,
    }


def plot_reward_trajectories(
    episode_to_pred_rewards: Dict[int, Sequence[float]],
    episode_to_gt_reward: Dict[int, Sequence[float]] | None = None,
    num_episodes_to_plot: int = 5,
    save_path: str = "reward_trajectories.png",
) -> None:
    episode_ids = sorted(episode_to_pred_rewards.keys())[:num_episodes_to_plot]
    if len(episode_ids) == 0:
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


def plot_per_stage_accuracy(per_stage_accuracy: Dict[int, float], save_path: str) -> None:
    stages = sorted(per_stage_accuracy.keys())
    accs = [per_stage_accuracy[s] for s in stages]

    plt.figure(figsize=(8, 5))
    plt.bar(stages, accs)
    plt.title("Per-Stage Classification Accuracy")
    plt.xlabel("Ground-Truth Stage")
    plt.ylabel("Accuracy")
    plt.xticks(stages)
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_stage_histogram(episode_to_pred_stage: Dict[int, int], save_path: str) -> None:
    hist = defaultdict(int)
    for _, stage in episode_to_pred_stage.items():
        hist[int(stage)] += 1

    stages = sorted(hist.keys())
    counts = [hist[s] for s in stages]

    plt.figure(figsize=(8, 5))
    plt.bar(stages, counts)
    plt.title("Predicted Stage Histogram")
    plt.xlabel("Predicted Stage")
    plt.ylabel("Number of Episodes")
    plt.xticks(stages)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


"""
python -m lerobot_policy_rewind.common.eval_rewind \
  --policy.type rewind \
  --policy.pretrained_path /home/chomeed/workspace/proj-rl-finetuning/outputs/train/2026-04-10/17-02-53_rewind_anubis/checkpoints/003000/pretrained_model \
  --dataset.repo_id chomeed/anubis_pick_carrot_out_of_pot_task_completion_for_rewind_same \
  --dataset.video_backend torchcodec \
  --batch_size 2 \
  --num_workers 16
"""
@parser.wrap()
def infer_task_completion_main(cfg: TrainPipelineConfig):
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

    annotation_path = "/home/chomeed/workspace/proj-rl-finetuning/hrm/training/anubis_pick_carrot_out_of_pot_task_completion_annotations.json"

    print("Start evaluation...")
    results = evaluate_episode_dataset_task_completion(
        dataloader=dataloader,
        reward_model=reward_model,
        annotation_path=annotation_path,
        accelerator=accelerator,
        alpha=(0.2, 0.35, 0.55, 0.65, 0.8),
        beta=(0.15, 0.30, 0.50, 0.60, 0.75),
        last_window_ratio=1 / 3,
        default_stage=0,
        preprocessor=preprocessor,
    )

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("Evaluation summary")
        print("=" * 80)

        print(f"Reward-level MAE: {results['reward_mae']:.6f}")

        kendall_results = results["kendall"]
        print(f"Kendall's tau: {kendall_results['kendall_tau']:.6f}")
        print(f"p-value: {kendall_results['p_value']:.6g}")
        print(f"#episodes used: {kendall_results['num_episodes_used']}")

        cls = results["classification"]
        print()
        print(
            f"Overall stage accuracy: "
            f"{cls['num_correct']}/{cls['num_eval']} = {cls['overall_accuracy']:.4f}"
        )

        print("\nPer-stage classification success rate:")
        for stage in sorted(results["stage_names"]):
            name = results["stage_names"][stage]
            total = cls["per_stage_total"].get(stage, 0)
            correct = cls["per_stage_correct"].get(stage, 0)
            acc = cls["per_stage_accuracy"].get(stage, 0.0)
            print(
                f"Stage {stage} ({name}): "
                f"{correct}/{total} = {acc:.4f} ({acc * 100:.2f}%)"
            )

        print("\nEpisode to Predicted Stage:")
        for ep_idx in sorted(results["episode_to_pred_stage"].keys()):
            pred_stage = results["episode_to_pred_stage"][ep_idx]
            final_reward = results["episode_to_final_reward"][ep_idx]
            print(
                f"Episode {ep_idx}: Predicted Stage {pred_stage}, "
                f"Final Reward {final_reward:.6f}"
            )

        plot_reward_trajectories(
            episode_to_pred_rewards=results["episode_to_pred_rewards"],
            episode_to_gt_reward=results["episode_to_gt_reward"],
            num_episodes_to_plot=5,
            save_path="reward_trajectories.png",
        )
        plot_per_stage_accuracy(
            per_stage_accuracy=cls["per_stage_accuracy"],
            save_path="per_stage_accuracy.png",
        )
        plot_stage_histogram(
            episode_to_pred_stage=results["episode_to_pred_stage"],
            save_path="predicted_stage_histogram.png",
        )

        with open("task_completion_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "reward_mae": results["reward_mae"],
                    "kendall": results["kendall"],
                    "classification": results["classification"],
                    "episode_to_final_reward": results["episode_to_final_reward"],
                    "episode_to_pred_stage": results["episode_to_pred_stage"],
                    "stage_names": results["stage_names"],
                },
                f,
                indent=2,
            )
        print("\nSaved:")
        print("  - reward_trajectories.png")
        print("  - per_stage_accuracy.png")
        print("  - predicted_stage_histogram.png")
        print("  - task_completion_eval_results.json")


def main():
    infer_task_completion_main()


if __name__ == "__main__":
    main()