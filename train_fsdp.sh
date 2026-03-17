VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_1b}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"

accelerate launch --num_processes 4 train.py \
  --dataset_type hd_epic_local \
  --video_root "$VIDEO_ROOT" \
  --metadata_root "$METADATA_ROOT" \
  --annotation_path "$ANNOTATION_PATH" \
  --video_extension mp4 \
  --val_ratio 0.01 \
  --batch_size 1 \
  --num_workers 0 \
  --video_frames 8 \
  --grad_accum_steps 16 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --epochs 3 \
  --log_every 20 \
  --vl_model_preset "$VL_MODEL_PRESET" \
  --peft lora \
  --gradient_checkpointing \
  --fsdp \
  --save_dir checkpoints_belief_hd_epic_fsdp
