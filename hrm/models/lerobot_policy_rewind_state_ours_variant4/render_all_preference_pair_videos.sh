#!/bin/bash
#SBATCH --job-name=render_all_pref_pair_videos
#SBATCH --time=04:00:00
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

# Render side-by-side FAIL | GOOD#1.. preference-pair MP4s for every
# preference_pairs*.json under the four rewind_state_ours* manifest folders.
# One MP4 per pair lands in a subfolder named after the manifest stem, e.g.
#   manifests/rewind_state_ours_variant2/preference_pairs_match_last5/pair_000_*.mp4
# The renderer loads no model (GPU not strictly needed); kept SLURM-compatible
# for consistency with the package's other launchers.

source ~/.bashrc
conda activate proj-rl-finetuning
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
set -uo pipefail

MANIFEST_ROOT=/home/chomeed/workspace/proj-rl-finetuning/manifests
DIRS=(rewind_state_ours rewind_state_ours_variant2 rewind_state_ours_variant3 rewind_state_ours_variant4)
N_PAIRS=100000   # clamped by the tool to each manifest's pair count -> renders all

failures=()
for d in "${DIRS[@]}"; do
  for json in "$MANIFEST_ROOT/$d"/preference_pairs*.json; do
    [ -e "$json" ] || continue
    stem=$(basename "$json" .json)
    out="$MANIFEST_ROOT/$d/$stem"
    echo "=== Rendering $json -> $out ==="
    if python -m lerobot_policy_rewind_state_ours_variant4.common.visualize_preference_pairs \
        --manifest "$json" --n_pairs "$N_PAIRS" --seed 0 --fps 10 --output_dir "$out"; then
      echo "OK: $json"
    else
      echo "FAILED: $json"
      failures+=("$json")
    fi
  done
done

echo
if [ ${#failures[@]} -eq 0 ]; then
  echo "All manifests rendered successfully."
else
  echo "Failed manifests (${#failures[@]}):"
  printf '  %s\n' "${failures[@]}"
fi
