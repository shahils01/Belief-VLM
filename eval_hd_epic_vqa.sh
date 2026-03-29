CHECKPOINT="${CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_vlm_hd_epic_ddp/ckpt_epoch_23.pt}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
USE_DB_PRIOR="${USE_DB_PRIOR:-0}"
DB_TOP_K="${DB_TOP_K:-1}"
DB_PRIOR_PREFIX="${DB_PRIOR_PREFIX:-Belief prior:}"
DB_MEMORY_ANNOTATION_PATH="${DB_MEMORY_ANNOTATION_PATH:-}"
RETRIEVAL_EMBEDDER_MODEL="${RETRIEVAL_EMBEDDER_MODEL:-}"

CMD=(
  python eval_hd_epic_vqa.py
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --max_samples "$MAX_SAMPLES"
  --print_samples "$PRINT_SAMPLES"
  --progress_every "$PROGRESS_EVERY"
  --vl_model_preset "$VL_MODEL_PRESET"
  --video_frames "20"
  --mixed_precision bf16
  --allow_tf32
)

if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

if [[ -n "$SAVE_PREDICTIONS" ]]; then
  CMD+=(--save_predictions "$SAVE_PREDICTIONS")
fi

if [[ "$USE_DB_PRIOR" == "1" ]]; then
  CMD+=(--use_db_prior --db_top_k "$DB_TOP_K" --db_prior_prefix "$DB_PRIOR_PREFIX")
  if [[ -n "$DB_MEMORY_ANNOTATION_PATH" ]]; then
    CMD+=(--db_memory_annotation_path "$DB_MEMORY_ANNOTATION_PATH")
  fi
  if [[ -n "$RETRIEVAL_EMBEDDER_MODEL" ]]; then
    CMD+=(--retrieval_embedder_model "$RETRIEVAL_EMBEDDER_MODEL")
  fi
fi

"${CMD[@]}"
