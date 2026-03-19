VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
FUTURE_OFFSET_SEC="${FUTURE_OFFSET_SEC:-2.0}"
FUTURE_DURATION_SEC="${FUTURE_DURATION_SEC:-0.0}"
VIDEO_FRAMES="${VIDEO_FRAMES:-8}"
FUTURE_FRAMES="${FUTURE_FRAMES:-8}"

accelerate launch --num_processes 1 train_future_predictor.py \
  --annotation_path "$ANNOTATION_PATH" \
  --video_root "$VIDEO_ROOT" \
  --video_frames "$VIDEO_FRAMES" \
  --future_frames "$FUTURE_FRAMES" \
  --future_offset_sec "$FUTURE_OFFSET_SEC" \
  --future_duration_sec "$FUTURE_DURATION_SEC" \
  --batch_size 2 \
  --num_workers 4 \
  --epochs 10 \
  --log_every 20 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --vl_model_preset "$VL_MODEL_PRESET" \
  --gradient_checkpointing \
  --save_dir checkpoints_future_predictor
