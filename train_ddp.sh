VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"
QUESTION_COLUMN="${QUESTION_COLUMN:-question}"
ANSWER_COLUMN="${ANSWER_COLUMN:-answer}"
VIDEO_ID_COLUMN="${VIDEO_ID_COLUMN:-video_id}"
PARTICIPANT_COLUMN="${PARTICIPANT_COLUMN:-participant_id}"
DEBUG_GENERATE="${DEBUG_GENERATE:-0}"
DEBUG_GENERATE_EVERY="${DEBUG_GENERATE_EVERY:-0}"
USE_BELIEF_MODEL="${USE_BELIEF_MODEL:-1}"
BELIEF_NUM_TOKENS="${BELIEF_NUM_TOKENS:-4}"

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
  --val_ratio 0.1
  --batch_size 2
  --num_workers 4
  --video_frames 8
  --grad_accum_steps 16
  --mixed_precision bf16
  --allow_tf32
  --epochs 200
  --log_every 1
  --train_objective mc
  --belief_fusion_scope vision_text
  --belief_use_recurrence
  --belief_temporal_chunks 2
  --mc_loss_weight 1.0
  --lm_loss_weight 1.0
  --belief_aux_loss_weight 0.25
  --choice_micro_batch_size 2
  --lr_scheduler cosine
  --warmup_ratio 0.03
  --save_best_metric val_mc_acc
  --vl_model_preset "$VL_MODEL_PRESET"
  --gradient_checkpointing
  --save_dir checkpoints_belief_hd_epic_ddp_ca_belief
  --resume_checkpoint /scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt
  --load_model_only
  --wandb
)

if [[ "$DEBUG_GENERATE" == "1" ]]; then
  CMD+=(--debug_generate --debug_generate_every "$DEBUG_GENERATE_EVERY")
fi

if [[ "$USE_BELIEF_MODEL" == "1" ]]; then
  CMD+=(
    --use_belief_model
    --belief_num_tokens "$BELIEF_NUM_TOKENS"
  )
fi

"${CMD[@]}"
