#!/bin/bash
# CPU-only smoke test for the variant 7 dexmg_threading exaflops trainer.
#
# Same `accelerate launch ...` invocation as the _exaflops.sh launcher (same
# --policy.* flags), but with: --cpu, batch_size=1, num_workers=0, steps=3,
# wandb off, save off, output to /tmp. Reads the SAME base ckpt + preference
# manifest from /lustre that the exaflops job will read.
#
# Purpose: catch CLI arg typos, missing package install, manifest format
# mismatches, forward-pass shape errors BEFORE burning an sbatch slot.
#
# Runtime: ~2-5 min on CPU (DINOv2 finetune forward dominates; batch_size=1
# keeps it bearable). The first dataset download caches into ~/.cache, so
# re-runs are faster.
#
# Run from anywhere with the `proj-rl-finetuning` conda env active:
#   bash /home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind_state_ours_variant7/cpu_smoke_test_dexmg_threading.sh
set -euo pipefail

# Same paths the _exaflops.sh launcher uses (the local box can read /lustre).
BASE_CKPT=/lustre/chomeed/exaflops/ckpts/rewind/rewind_state_dexmg_threading_v1/checkpoints/last/pretrained_model
PREF_MANIFEST=/lustre/chomeed/exaflops/manifests/dexmg_two_arm_threading/preference_pairs_blend.json
SUCCESS_REPO=chomeed/dexmg_two_arm_threading

OUTPUT_DIR=/tmp/variant7_cpu_smoke
JOB_NAME=variant7_cpu_smoke

# --policy.* knob set MUST match the _exaflops launcher exactly — that's the
# parity check. If a knob name typo'd, accelerate launch will reject it here.
REWARD_PREF_LOSS_WEIGHT=0.0
REWARD_PREF_WARMUP_STEPS=500
REWARD_PREF_DISCOUNT=0.85
STOP_GRAD_ON_SUCCESS=False
BT_TARGET_MARGIN=2.0
MONO_LOSS_WEIGHT=1.0
MONO_WARMUP_STEPS=500
MONO_ON_FAILURE=True
MONO_ON_SUCCESS=True
MONO_STEP_EPS=0.2
TV_LOSS_WEIGHT=0.0
TV_WARMUP_STEPS=500
TV_ORDER=2
TV_ON_FAILURE=True
TV_ON_SUCCESS=True
FAILURE_SAMPLING_MODE=hybrid
FAILURE_HYBRID_N_RECENT=8
SUCCESS_SAMPLING_MODE=hybrid
SUCCESS_HYBRID_N_RECENT=8
SUCC_ANCHOR_MSE_WEIGHT=1.0
SUCC_ANCHOR_MSE_WARMUP_STEPS=500

# Sanity-check the inputs before invoking the trainer.
[ -d "$BASE_CKPT" ]  || { echo "ERROR: base ckpt missing at $BASE_CKPT" >&2; exit 1; }
[ -f "$PREF_MANIFEST" ] || { echo "ERROR: manifest missing at $PREF_MANIFEST" >&2; exit 1; }

# Force CPU. --cpu flag of accelerate is the contract; the export is a belt.
export CUDA_VISIBLE_DEVICES=""

rm -rf "$OUTPUT_DIR"

accelerate launch \
  --cpu \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=no \
  -m lerobot_policy_rewind_state_ours_variant7.common.train_rewind_state_ours_robomimic_variant7 \
  --policy.type=rewind_state_ours_variant7 \
  --policy.repo_id=chomeed/rewind_state_ours_variant7_dexmg_threading_v1 \
  --policy.push_to_hub=False \
  --policy.num_cameras=3 \
  --policy.finetune_encoder=True \
  --policy.finetune_encoder_blocks=2 \
  --policy.state_dim=18 \
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.robot0_eye_in_hand","observation.images.robot1_eye_in_hand"]' \
  --policy.pretrained_path="$BASE_CKPT" \
  --policy.preference_pairs_path="$PREF_MANIFEST" \
  --policy.preference_clip_len=16 \
  --policy.preference_frame_stride=1 \
  --policy.preference_topk=3 \
  --policy.reward_pref_loss_weight="$REWARD_PREF_LOSS_WEIGHT" \
  --policy.reward_pref_warmup_steps="$REWARD_PREF_WARMUP_STEPS" \
  --policy.reward_pref_discount="$REWARD_PREF_DISCOUNT" \
  --policy.stop_grad_on_success="$STOP_GRAD_ON_SUCCESS" \
  --policy.bt_target_margin="$BT_TARGET_MARGIN" \
  --policy.mono_loss_weight="$MONO_LOSS_WEIGHT" \
  --policy.mono_warmup_steps="$MONO_WARMUP_STEPS" \
  --policy.mono_on_failure="$MONO_ON_FAILURE" \
  --policy.mono_on_success="$MONO_ON_SUCCESS" \
  --policy.mono_step_eps="$MONO_STEP_EPS" \
  --policy.tv_loss_weight="$TV_LOSS_WEIGHT" \
  --policy.tv_warmup_steps="$TV_WARMUP_STEPS" \
  --policy.tv_order="$TV_ORDER" \
  --policy.tv_on_failure="$TV_ON_FAILURE" \
  --policy.tv_on_success="$TV_ON_SUCCESS" \
  --policy.failure_sampling_mode="$FAILURE_SAMPLING_MODE" \
  --policy.failure_hybrid_n_recent="$FAILURE_HYBRID_N_RECENT" \
  --policy.success_sampling_mode="$SUCCESS_SAMPLING_MODE" \
  --policy.success_hybrid_n_recent="$SUCCESS_HYBRID_N_RECENT" \
  --policy.succ_anchor_mse_weight="$SUCC_ANCHOR_MSE_WEIGHT" \
  --policy.succ_anchor_mse_warmup_steps="$SUCC_ANCHOR_MSE_WARMUP_STEPS" \
  --batch_size=1 \
  --num_workers=0 \
  --steps=3 \
  --log_freq=1 \
  --eval_freq=10000 \
  --save_freq=10000 \
  --save_checkpoint=False \
  --wandb.enable=False \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.scheduler_decay_steps=100 \
  --policy.scheduler_peak_lr=5e-5 \
  --dataset.repo_id="$SUCCESS_REPO" \
  --val.repo_id="$SUCCESS_REPO"

echo
echo "[$(date)] CPU smoke test PASSED."
echo "All --policy.* flags parsed, dataset loaded, $((3)) steps completed."
echo "Safe to submit the _exaflops.sh sbatch job."
