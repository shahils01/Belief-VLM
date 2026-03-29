VL_MODEL_PRESET="${VL_MODEL_PRESET:-custom}"
VL_MODEL_NAME="${VL_MODEL_NAME:-/scratch/shahils/hf_models/InternVL3_5-2B-HF}"
CHECKPOINT="${CHECKPOINT:-checkpoints_ppo_vqa_fulldataset/ckpt_epoch_33.pt}"
EVAL_MODE="${EVAL_MODE:-ppo}"
ANNOTATION_PATH="$(printf "%s," /scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_*.json | sed 's/,$//')" 
# ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"
VAL_RATIO="${VAL_RATIO:-1.0}"
USE_DB_PRIOR="${USE_DB_PRIOR:-0}"
DB_MODALITY="${DB_MODALITY:-text}"
DB_TOP_K="${DB_TOP_K:-1}"
DB_PRIOR_PREFIX="${DB_PRIOR_PREFIX:-Belief prior:}"
DB_MEMORY_ANNOTATION_PATH="${DB_MEMORY_ANNOTATION_PATH:-}"
RETRIEVAL_EMBEDDER_MODEL="${RETRIEVAL_EMBEDDER_MODEL:-}"
DB_INDEX_BACKEND="${DB_INDEX_BACKEND:-auto}"
DB_BUILD_BATCH_SIZE="${DB_BUILD_BATCH_SIZE:-4}"

CMD=(
  python eval_ppo_vqa.py
  --eval_mode "$EVAL_MODE"
  --checkpoint "$CHECKPOINT"
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --max_samples "$MAX_SAMPLES"
  --max_val_samples_per_split "$MAX_VAL_SAMPLES"
  --print_samples "$PRINT_SAMPLES"
  --progress_every "$PROGRESS_EVERY"
  --video_frames 8
)

if [[ -n "$SAVE_PREDICTIONS" ]]; then
  CMD+=(--save_predictions "$SAVE_PREDICTIONS")
fi

if [[ -n "$VAL_RATIO" ]]; then
  CMD+=(--val_ratio "$VAL_RATIO")
fi

if [[ "$USE_DB_PRIOR" == "1" ]]; then
  CMD+=(--use_db_prior --db_modality "$DB_MODALITY" --db_top_k "$DB_TOP_K" --db_prior_prefix "$DB_PRIOR_PREFIX" --db_index_backend "$DB_INDEX_BACKEND" --db_build_batch_size "$DB_BUILD_BATCH_SIZE")
  if [[ -n "$DB_MEMORY_ANNOTATION_PATH" ]]; then
    CMD+=(--db_memory_annotation_path "$DB_MEMORY_ANNOTATION_PATH")
  fi
  if [[ -n "$RETRIEVAL_EMBEDDER_MODEL" ]]; then
    CMD+=(--retrieval_embedder_model "$RETRIEVAL_EMBEDDER_MODEL")
  fi
fi

"${CMD[@]}"
