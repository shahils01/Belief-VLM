set -euo pipefail

VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
DATASET_TYPE="${DATASET_TYPE:-nextqa_local}"   # hd_epic_local | nextqa_local
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/anshuln/NExTVQA/NExTVideo}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/anshuln/NExTVQA/NExT-QA/dataset/nextqa/train.csv}"
METADATA_ROOT="${METADATA_ROOT:-}"
VLM_CHECKPOINT="${VLM_CHECKPOINT:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-2048}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-Belief_VLM}"
WANDB_ENTITY="${WANDB_ENTITY:-i2rLAB}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-RL_Top-k}"
WANDB_TAGS="${WANDB_TAGS:-}"
DROP_MISSING_VIDEOS="${DROP_MISSING_VIDEOS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"

CMD=(
  accelerate launch
  --num_processes "$NUM_PROCESSES"
  --num_machines 1
  --mixed_precision "$ACCELERATE_MIXED_PRECISION"
  --dynamo_backend no
  train_ppo_vqa.py
  --dataset_type "$DATASET_TYPE"
  --video_root "$VIDEO_ROOT"
  --metadata_root "$METADATA_ROOT"
  --annotation_path "$ANNOTATION_PATH"
  --video_extension mp4
  --val_ratio 0.1
  --max_val_samples_per_split "$MAX_VAL_SAMPLES"
  --train_sampling_mode task_uniform
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --batch_size 4
  --num_workers 4
  --video_frames 8
  --mixed_precision bf16
  --allow_tf32
  --epochs 500
  --ppo_epochs 10
  --policy_lr 1e-4
  --vlm_lr 2e-5
  --vl_model_preset "$VL_MODEL_PRESET"
  --gradient_checkpointing
  --save_dir checkpoints_ppo_vqa_fulldataset
)

if [[ -n "$VLM_CHECKPOINT" ]]; then
  CMD+=(--vlm_checkpoint "$VLM_CHECKPOINT")
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

if [[ "$DROP_MISSING_VIDEOS" == "1" ]]; then
  CMD+=(--drop_missing_videos)
fi

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--wandb --wandb_project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_RUN_NAME" ]]; then
    CMD+=(--wandb_run_name "$WANDB_RUN_NAME")
  fi
  if [[ -n "$WANDB_TAGS" ]]; then
    CMD+=(--wandb_tags "$WANDB_TAGS")
  fi
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
