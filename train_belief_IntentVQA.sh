#!/usr/bin/env bash
set -euo pipefail

DATASET_SOURCE="${DATASET_SOURCE:-hd_epic}"   # hd_epic | nextvqa
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/anshuln/NExTVQA/NExT-QA/dataset/nextqa/train.csv}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/anshuln/NExTVQA/NExTVideo/}"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"

BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
VIDEO_FRAMES="${VIDEO_FRAMES:-8}"

# IntentQA-style settings
TOPK_NODES="${TOPK_NODES:-8}"
TRIPLET_MARGIN="${TRIPLET_MARGIN:-0.2}"
TRIPLET_WEIGHT="${TRIPLET_WEIGHT:-0.2}"
WUPS_T1="${WUPS_T1:-0.9}"
WUPS_T2="${WUPS_T2:-0.1}"
COMMONSENSE_WEIGHT="${COMMONSENSE_WEIGHT:-0.5}"

SAVE_DIR="${SAVE_DIR:-/scratch/shahils/Belief-VLM/checkpoints_belief_intent_vqa}"

accelerate launch --num_processes 1 train_belief_IntentVQA.py \
  --dataset_source "$DATASET_SOURCE" \
  --annotation_path "$ANNOTATION_PATH" \
  --video_root "$VIDEO_ROOT" \
  --video_frames "$VIDEO_FRAMES" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --vl_model_preset "$VL_MODEL_PRESET" \
  --topk_nodes "$TOPK_NODES" \
  --triplet_margin "$TRIPLET_MARGIN" \
  --triplet_weight "$TRIPLET_WEIGHT" \
  --wups_t1 "$WUPS_T1" \
  --wups_t2 "$WUPS_T2" \
  --commonsense_weight "$COMMONSENSE_WEIGHT" \
  --mixed_precision bf16 \
  --allow_tf32 \
  --gradient_checkpointing \
  --save_dir "$SAVE_DIR"

