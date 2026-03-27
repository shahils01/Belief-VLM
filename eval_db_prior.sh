CHECKPOINT="${CHECKPOINT:-checkpoints_db_prior_rl2_lora/ckpt_epoch_0.pt}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/3d_perception_fixture_location.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
VIDEO_EXTENSION="${VIDEO_EXTENSION:-mp4}"
METADATA_ROOT="${METADATA_ROOT:-}"
VAL_RATIO="${VAL_RATIO:-0.1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-20}"
USE_RL_ANSWER_HEAD="${USE_RL_ANSWER_HEAD:-1}"
USE_RL_PRIOR_SELECTOR="${USE_RL_PRIOR_SELECTOR:-1}"

CMD=(
  python eval_db_prior.py
  --checkpoint "$CHECKPOINT"
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --video_extension "$VIDEO_EXTENSION"
  --metadata_root "$METADATA_ROOT"
  --val_ratio "$VAL_RATIO"
  --max_samples_per_split "$MAX_SAMPLES"
  --print_samples "$PRINT_SAMPLES"
)

if [ "$USE_RL_ANSWER_HEAD" = "1" ]; then
  CMD+=(--use_rl_answer_head)
fi

if [ "$USE_RL_PRIOR_SELECTOR" = "1" ]; then
  CMD+=(--use_rl_prior_selector)
fi

"${CMD[@]}"
