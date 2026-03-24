VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"
QUESTION_COLUMN="${QUESTION_COLUMN:-question}"
ANSWER_COLUMN="${ANSWER_COLUMN:-answer}"
VIDEO_ID_COLUMN="${VIDEO_ID_COLUMN:-video_id}"
PARTICIPANT_COLUMN="${PARTICIPANT_COLUMN:-participant_id}"
DEBUG_GENERATE="${DEBUG_GENERATE:-0}"
DEBUG_GENERATE_EVERY="${DEBUG_GENERATE_EVERY:-0}"
PREDICTIVE_MODULE="${PREDICTIVE_MODULE:-future}"
USE_FUTURE_PREDICTOR="${USE_FUTURE_PREDICTOR:-1}"
FUTURE_PREDICTOR_CHECKPOINT="${FUTURE_PREDICTOR_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_future_predictor/ckpt_epoch_197.pt}"
FUTURE_FRAMES="${FUTURE_FRAMES:-1}"
FINETUNE_FUTURE_PREDICTOR="${FINETUNE_FUTURE_PREDICTOR:-1}"
FUTURE_AUX_WEIGHT="${FUTURE_AUX_WEIGHT:-0.1}"
FUTURE_OFFSET_SEC="${FUTURE_OFFSET_SEC:-0.0}"
FUTURE_DURATION_SEC="${FUTURE_DURATION_SEC:-0.0}"
USE_BELIEF_NETWORK="${USE_BELIEF_NETWORK:-0}"
BELIEF_NETWORK_CHECKPOINT="${BELIEF_NETWORK_CHECKPOINT:-}"
FINETUNE_BELIEF_NETWORK="${FINETUNE_BELIEF_NETWORK:-0}"
BELIEF_AUX_WEIGHT="${BELIEF_AUX_WEIGHT:-0.1}"

CMD=(
  accelerate launch --num_processes 6 train.py
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
  --batch_size 2
  --num_workers 4
  --video_frames 8
  --grad_accum_steps 16
  --mixed_precision bf16
  --allow_tf32
  --epochs 100
  --log_every 1
  --vl_model_preset "$VL_MODEL_PRESET"
  --gradient_checkpointing
  --save_dir checkpoints_future_full_finetune_scratch_temp
  --wandb
  --wandb_run_name "future_full_ft"
)

if [[ "$DEBUG_GENERATE" == "1" ]]; then
  CMD+=(--debug_generate --debug_generate_every "$DEBUG_GENERATE_EVERY")
fi

if [[ "$PREDICTIVE_MODULE" == "future" || "$USE_FUTURE_PREDICTOR" == "1" ]]; then
  CMD+=(--use_future_predictor --future_predictor_checkpoint "$FUTURE_PREDICTOR_CHECKPOINT" --future_frames "$FUTURE_FRAMES")
  if [[ "$FINETUNE_FUTURE_PREDICTOR" == "1" ]]; then
    CMD+=(--finetune_future_predictor --future_aux_weight "$FUTURE_AUX_WEIGHT" --future_offset_sec "$FUTURE_OFFSET_SEC" --future_duration_sec "$FUTURE_DURATION_SEC")
  fi
fi

if [[ "$PREDICTIVE_MODULE" == "belief" || "$USE_BELIEF_NETWORK" == "1" ]]; then
  CMD+=(--use_belief_network --belief_network_checkpoint "$BELIEF_NETWORK_CHECKPOINT")
  if [[ "$FINETUNE_BELIEF_NETWORK" == "1" ]]; then
    CMD+=(--finetune_belief_network --belief_aux_weight "$BELIEF_AUX_WEIGHT")
  fi
fi

"${CMD[@]}"

  # --peft qlora
  # --resume_checkpoint "/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt"
  # --load_model_only