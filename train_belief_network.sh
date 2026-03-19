VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
FUTURE_OFFSET_SEC="${FUTURE_OFFSET_SEC:-0.0}"
FUTURE_DURATION_SEC="${FUTURE_DURATION_SEC:-0.0}"
VIDEO_FRAMES="${VIDEO_FRAMES:-20}"

accelerate launch --num_processes 1 train_belief_network.py \
  --annotation_path "$ANNOTATION_PATH" \
  --video_root "$VIDEO_ROOT" \
  --video_frames "$VIDEO_FRAMES" \
  --batch_size 16 \
  --num_workers 4 \
  --epochs 100 \
  --log_every 20 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --vl_model_preset "$VL_MODEL_PRESET" \
  --gradient_checkpointing \
  --save_dir /scratch/shahils/Belief-VLM/checkpoints_future_belief_predictor \
  # --resume_checkpoint /scratch/shahils/Belief-VLM/checkpoints_future_predictor/ckpt_epoch_3.pt