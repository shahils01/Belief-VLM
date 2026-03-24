#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
VIDEO_EXTENSION="${VIDEO_EXTENSION:-mp4}"
QUESTION_COLUMN="${QUESTION_COLUMN:-question}"
ANSWER_COLUMN="${ANSWER_COLUMN:-answer}"
VIDEO_ID_COLUMN="${VIDEO_ID_COLUMN:-video_id}"
PARTICIPANT_COLUMN="${PARTICIPANT_COLUMN:-participant_id}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-500}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_DIR="${SAVE_DIR:-/scratch/shahils/Belief-VLM/checkpoints_future_predictor}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_future_predictor/ckpt_epoch_197.pt}"
VL_CHECKPOINT="${VL_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt}"

VIDEO_FRAMES="${VIDEO_FRAMES:-8}"
FUTURE_FRAMES="${FUTURE_FRAMES:-1}"
FUTURE_OFFSET_SEC="${FUTURE_OFFSET_SEC:-0.0}"
FUTURE_DURATION_SEC="${FUTURE_DURATION_SEC:-2.0}"
VISION_LOSS_LAYERS="${VISION_LOSS_LAYERS:-6}"
PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-6}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

CMD=(
  accelerate launch --num_processes "$NUM_PROCESSES" train_future_predictor.py
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --video_extension "$VIDEO_EXTENSION"
  --question_column "$QUESTION_COLUMN"
  --answer_column "$ANSWER_COLUMN"
  --video_id_column "$VIDEO_ID_COLUMN"
  --participant_column "$PARTICIPANT_COLUMN"
  --video_frames "$VIDEO_FRAMES"
  --future_frames "$FUTURE_FRAMES"
  --future_offset_sec "$FUTURE_OFFSET_SEC"
  --future_duration_sec "$FUTURE_DURATION_SEC"
  --vision_loss_layers "$VISION_LOSS_LAYERS"
  --predictor_layers "$PREDICTOR_LAYERS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --max_grad_norm "$MAX_GRAD_NORM"
  --log_every "$LOG_EVERY"
  --mixed_precision "$MIXED_PRECISION"
  --allow_tf32
  --vl_model_preset "$VL_MODEL_PRESET"
  --gradient_checkpointing
  --save_dir "$SAVE_DIR"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

if [[ -n "$VL_CHECKPOINT" ]]; then
  CMD+=(--vl_checkpoint "$VL_CHECKPOINT")
fi

"${CMD[@]}"
