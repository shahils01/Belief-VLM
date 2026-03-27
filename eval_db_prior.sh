CHECKPOINT="${CHECKPOINT:-checkpoints_db_prior_rlPi_lora/ckpt_epoch_8.pt}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
VAL_RATIO="${VAL_RATIO:-0.1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

python eval_db_prior.py \
  --checkpoint "$CHECKPOINT" \
  --annotation_path "$ANNOTATION_PATH" \
  --video_root "$VIDEO_ROOT" \
  --val_ratio "$VAL_RATIO" \
  --max_samples_per_split "$MAX_SAMPLES" \
  --use_rl_answer_head
