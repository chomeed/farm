# farm

Rewind-state reward-model packages, extracted from `proj-rl-finetuning`.

## Layout

All packages live under `hrm/models/`, each an installable package with its own
`pyproject.toml`, `src/` tree and `tests/`, plus the `*.sh` launcher scripts used
to train and evaluate it.

| Package | Role |
| --- | --- |
| `lerobot_policy_rewind_state` | Baseline rewind-state policy. The `_ours` variants import from it. |
| `lerobot_policy_rewind_state_ours` | First "ours" variant. |
| `lerobot_policy_rewind_state_ours_variant2` … `_variant8` | Successive variants. |

Install a package with `pip install -e hrm/models/<package>`.

## Not included

Experiment artifacts are deliberately left out of git (see `.gitignore`):
`wandb/` run directories, `outputs/`, and training `checkpoints/`. This repo holds
source and launcher scripts only.
