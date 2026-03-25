CHECKPOINT="${CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_ppo_vqa_01/ckpt_epoch_59.pt}"
EVAL_MODE="${EVAL_MODE:-ppo}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_how_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"
VAL_RATIO="${VAL_RATIO:-0.1}"

CMD=(
  python eval_ppo_vqa.py
  --eval_mode "$EVAL_MODE"
  --checkpoint "$CHECKPOINT"
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --max_samples "$MAX_SAMPLES"
  --print_samples "$PRINT_SAMPLES"
  --progress_every "$PROGRESS_EVERY"
)

if [[ -n "$SAVE_PREDICTIONS" ]]; then
  CMD+=(--save_predictions "$SAVE_PREDICTIONS")
fi

if [[ -n "$VAL_RATIO" ]]; then
  CMD+=(--val_ratio "$VAL_RATIO")
fi

"${CMD[@]}"
