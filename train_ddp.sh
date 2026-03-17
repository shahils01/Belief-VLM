VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_1b}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"
QUESTION_COLUMN="${QUESTION_COLUMN:-question}"
ANSWER_COLUMN="${ANSWER_COLUMN:-answer}"
VIDEO_ID_COLUMN="${VIDEO_ID_COLUMN:-video_id}"
PARTICIPANT_COLUMN="${PARTICIPANT_COLUMN:-participant_id}"

CMD=(
  accelerate launch --num_processes 2 train.py
  --dataset_type hd_epic_local
  --video_root "$VIDEO_ROOT"
  --metadata_root "$METADATA_ROOT"
  --annotation_path "$ANNOTATION_PATH"
  --video_extension mp4
  --question_column "$QUESTION_COLUMN"
  --answer_column "$ANSWER_COLUMN"
  --video_id_column "$VIDEO_ID_COLUMN"
  --participant_column "$PARTICIPANT_COLUMN"
  --val_ratio 0.01
  --batch_size 1
  --num_workers 0
  --video_frames 8
  --grad_accum_steps 16
  --mixed_precision bf16
  --allow_tf32
  --epochs 3
  --log_every 20
  --vl_model_preset "$VL_MODEL_PRESET"
  --peft qlora
  --gradient_checkpointing
  --save_dir checkpoints_belief_hd_epic_ddp
)

"${CMD[@]}"
