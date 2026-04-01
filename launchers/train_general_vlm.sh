#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-ai2d}"
DATASET_REPO="${DATASET_REPO:-}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-}"
EVAL_SPLIT="${EVAL_SPLIT:-}"
ANNOTATION_PATH="${ANNOTATION_PATH:-}"
STREAMING="${STREAMING:-0}"
MEDIA_ROOT="${MEDIA_ROOT:-}"
VIDEO_ROOT="${VIDEO_ROOT:-}"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
VL_MODEL_NAME="${VL_MODEL_NAME:-}"
PEFT="${PEFT:-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints_general_vlm}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
VIDEO_FRAMES="${VIDEO_FRAMES:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
SEED="${SEED:-42}"
WANDB="${WANDB:-0}"

CMD=(
  python /home/i2r/shahil_ws/Belief-VLM/train_general_vlm.py
  --dataset_name "$DATASET_NAME"
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --video_frames "$VIDEO_FRAMES"
  --epochs "$EPOCHS"
  --lr "$LR"
  --num_workers "$NUM_WORKERS"
  --mixed_precision "$MIXED_PRECISION"
  --save_dir "$OUTPUT_DIR"
  --peft "$PEFT"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --max_eval_samples "$MAX_EVAL_SAMPLES"
  --seed "$SEED"
  --vl_model_preset "$VL_MODEL_PRESET"
)

if [[ -n "$DATASET_REPO" ]]; then CMD+=(--dataset_repo "$DATASET_REPO"); fi
if [[ -n "$DATASET_CONFIG" ]]; then CMD+=(--dataset_config "$DATASET_CONFIG"); fi
if [[ -n "$TRAIN_SPLIT" ]]; then CMD+=(--train_split "$TRAIN_SPLIT"); fi
if [[ -n "$EVAL_SPLIT" ]]; then CMD+=(--eval_split "$EVAL_SPLIT"); fi
if [[ -n "$ANNOTATION_PATH" ]]; then CMD+=(--annotation_path "$ANNOTATION_PATH"); fi
if [[ -n "$MEDIA_ROOT" ]]; then CMD+=(--media_root "$MEDIA_ROOT"); fi
if [[ -n "$VIDEO_ROOT" ]]; then CMD+=(--video_root "$VIDEO_ROOT"); fi
if [[ -n "$VL_MODEL_NAME" ]]; then CMD+=(--vl_model_name "$VL_MODEL_NAME"); fi
if [[ "$STREAMING" == "1" ]]; then CMD+=(--streaming); fi
if [[ "$WANDB" == "1" ]]; then CMD+=(--wandb); fi

"${CMD[@]}"
