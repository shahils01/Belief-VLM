VL_MODEL_PRESET="${VL_MODEL_PRESET:-custom}"
VL_MODEL_NAME="${VL_MODEL_NAME:-/scratch/shahils/hf_models/InternVL3_5-2B-HF}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"
VLM_CHECKPOINT="${VLM_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-200000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-50}"
USE_DB_PRIOR="${USE_DB_PRIOR:-0}"
DB_MODALITY="${DB_MODALITY:-text}"
DB_TOP_K="${DB_TOP_K:-2}"
DB_PRIOR_PREFIX="${DB_PRIOR_PREFIX:-Belief prior:}"
DB_MEMORY_ANNOTATION_PATH="${DB_MEMORY_ANNOTATION_PATH:-}"
RETRIEVAL_EMBEDDER_MODEL="${RETRIEVAL_EMBEDDER_MODEL:-}"
DB_INDEX_BACKEND="${DB_INDEX_BACKEND:-auto}"
DB_BUILD_BATCH_SIZE="${DB_BUILD_BATCH_SIZE:-4}"

CMD=(
  accelerate launch --num_processes 4 train_ppo_vqa.py
  --dataset_type hd_epic_local
  --video_root "$VIDEO_ROOT"
  --metadata_root "$METADATA_ROOT"
  --annotation_path "$ANNOTATION_PATH"
  --video_extension mp4
  --val_ratio 0.1
  --max_val_samples_per_split "$MAX_VAL_SAMPLES"
  --train_sampling_mode task_uniform
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --batch_size 8
  --grad_accum_steps 8
  --num_workers 1
  --video_frames 8
  --mixed_precision bf16
  --allow_tf32
  --epochs 5000
  --ppo_epochs 10
  --policy_lr 1e-4
  --vlm_lr 1e-4
  --vl_model_preset "$VL_MODEL_PRESET"
  --vl_model_name "$VL_MODEL_NAME"
  --resume_checkpoint checkpoints_ppo_vqa_fulldataset/ckpt_epoch_34.pt
  --gradient_checkpointing
  --save_dir checkpoints_ppo_vqa_fulldataset
)

if [[ -n "$VLM_CHECKPOINT" ]]; then
  CMD+=(--vlm_checkpoint "$VLM_CHECKPOINT")
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

  # --grad_accum_steps 8
