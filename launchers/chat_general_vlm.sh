#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_general_vlm/ckpt_epoch_20.pt}"
QUESTION="${QUESTION:-"explain this image?"}"
IMAGE_PATH="${IMAGE_PATH:-/home/shahils/Desktop/marl_ws/Multi-Agent-Transformer/training_pipelines.jpg}"
VIDEO_PATH="${VIDEO_PATH:-}"
TASK_NAME="${TASK_NAME:-chat}"
SAMPLE_ID="${SAMPLE_ID:-interactive}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
USE_MEMORY_RETRIEVAL="${USE_MEMORY_RETRIEVAL:-1}"
MEMORY_TOP_K="${MEMORY_TOP_K:-2}"
MEMORY_INDEX_BACKEND="${MEMORY_INDEX_BACKEND:-auto}"
MEMORY_SAME_TASK_FIRST="${MEMORY_SAME_TASK_FIRST:-1}"
MEMORY_LAYER_IDX="${MEMORY_LAYER_IDX:-}"
MEMORY_INJECT_OFFSET="${MEMORY_INJECT_OFFSET:-}"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-}"
VL_MODEL_NAME="${VL_MODEL_NAME:-}"

CMD=(
  python chat_general_vlm.py
  --checkpoint "$CHECKPOINT"
  --question "$QUESTION"
  --task_name "$TASK_NAME"
  --sample_id "$SAMPLE_ID"
  --max_new_tokens "$MAX_NEW_TOKENS"
)

if [[ -n "$IMAGE_PATH" ]]; then CMD+=(--image_path "$IMAGE_PATH"); fi
if [[ -n "$VIDEO_PATH" ]]; then CMD+=(--video_path "$VIDEO_PATH"); fi
if [[ "$USE_MEMORY_RETRIEVAL" == "1" ]]; then CMD+=(--use_memory_retrieval); fi
if [[ "$MEMORY_SAME_TASK_FIRST" == "1" ]]; then CMD+=(--memory_same_task_first); fi
CMD+=(--memory_top_k "$MEMORY_TOP_K" --memory_index_backend "$MEMORY_INDEX_BACKEND")
if [[ -n "$MEMORY_LAYER_IDX" ]]; then CMD+=(--memory_layer_idx "$MEMORY_LAYER_IDX"); fi
if [[ -n "$MEMORY_INJECT_OFFSET" ]]; then CMD+=(--memory_inject_offset "$MEMORY_INJECT_OFFSET"); fi
if [[ -n "$VL_MODEL_PRESET" ]]; then CMD+=(--vl_model_preset "$VL_MODEL_PRESET"); fi
if [[ -n "$VL_MODEL_NAME" ]]; then CMD+=(--vl_model_name "$VL_MODEL_NAME"); fi

"${CMD[@]}"
