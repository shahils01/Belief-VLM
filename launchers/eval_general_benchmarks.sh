#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-ai2d}"
BENCHMARK_NAME="${BENCHMARK_NAME:-}"
CHECKPOINT="${CHECKPOINT:-}"
DATASET_REPO="${DATASET_REPO:-}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
EVAL_SPLIT="${EVAL_SPLIT:-}"
ANNOTATION_PATH="${ANNOTATION_PATH:-}"
STREAMING="${STREAMING:-0}"
MEDIA_ROOT="${MEDIA_ROOT:-}"
VIDEO_ROOT="${VIDEO_ROOT:-}"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
VL_MODEL_NAME="${VL_MODEL_NAME:-}"
VIDEO_FRAMES="${VIDEO_FRAMES:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
EVAL_MODE="${EVAL_MODE:-auto}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"
USE_MEMORY_RETRIEVAL="${USE_MEMORY_RETRIEVAL:-0}"
MEMORY_TOP_K="${MEMORY_TOP_K:-2}"
MEMORY_INDEX_BACKEND="${MEMORY_INDEX_BACKEND:-auto}"
MEMORY_SAME_TASK_FIRST="${MEMORY_SAME_TASK_FIRST:-1}"
MEMORY_LAYER_IDX="${MEMORY_LAYER_IDX:-}"
MEMORY_INJECT_OFFSET="${MEMORY_INJECT_OFFSET:-}"

CMD=(
  python /home/i2r/shahil_ws/Belief-VLM/eval_general_benchmarks.py
  --dataset_name "$DATASET_NAME"
  --video_frames "$VIDEO_FRAMES"
  --max_samples "$MAX_SAMPLES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --eval_mode "$EVAL_MODE"
  --print_samples "$PRINT_SAMPLES"
  --progress_every "$PROGRESS_EVERY"
  --vl_model_preset "$VL_MODEL_PRESET"
)

if [[ -n "$BENCHMARK_NAME" ]]; then CMD+=(--benchmark_name "$BENCHMARK_NAME"); fi
if [[ -n "$CHECKPOINT" ]]; then CMD+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "$DATASET_REPO" ]]; then CMD+=(--dataset_repo "$DATASET_REPO"); fi
if [[ -n "$DATASET_CONFIG" ]]; then CMD+=(--dataset_config "$DATASET_CONFIG"); fi
if [[ -n "$EVAL_SPLIT" ]]; then CMD+=(--eval_split "$EVAL_SPLIT"); fi
if [[ -n "$ANNOTATION_PATH" ]]; then CMD+=(--annotation_path "$ANNOTATION_PATH"); fi
if [[ -n "$MEDIA_ROOT" ]]; then CMD+=(--media_root "$MEDIA_ROOT"); fi
if [[ -n "$VIDEO_ROOT" ]]; then CMD+=(--video_root "$VIDEO_ROOT"); fi
if [[ -n "$VL_MODEL_NAME" ]]; then CMD+=(--vl_model_name "$VL_MODEL_NAME"); fi
if [[ -n "$SAVE_PREDICTIONS" ]]; then CMD+=(--save_predictions "$SAVE_PREDICTIONS"); fi
if [[ "$STREAMING" == "1" ]]; then CMD+=(--streaming); fi
if [[ "$USE_MEMORY_RETRIEVAL" == "1" ]]; then CMD+=(--use_memory_retrieval); fi
if [[ "$MEMORY_SAME_TASK_FIRST" == "1" ]]; then CMD+=(--memory_same_task_first); fi
CMD+=(--memory_top_k "$MEMORY_TOP_K" --memory_index_backend "$MEMORY_INDEX_BACKEND")
if [[ -n "$MEMORY_LAYER_IDX" ]]; then CMD+=(--memory_layer_idx "$MEMORY_LAYER_IDX"); fi
if [[ -n "$MEMORY_INJECT_OFFSET" ]]; then CMD+=(--memory_inject_offset "$MEMORY_INJECT_OFFSET"); fi

"${CMD[@]}"
