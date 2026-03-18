CHECKPOINT="${CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp/ckpt_epoch_23.pt}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"

CMD=(
  python eval_hd_epic_vqa.py
  --checkpoint "$CHECKPOINT"
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --max_samples "$MAX_SAMPLES"
  --print_samples "$PRINT_SAMPLES"
)

if [[ -n "$SAVE_PREDICTIONS" ]]; then
  CMD+=(--save_predictions "$SAVE_PREDICTIONS")
fi

"${CMD[@]}"
