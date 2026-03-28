VL_MODEL_PRESET="${VL_MODEL_PRESET:-custom}"
VL_MODEL_NAME="${VL_MODEL_NAME:-/scratch/shahils/hf_models/InternVL3_5-2B-HF}"
VLM_CHECKPOINT="${VLM_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt}"
CHECKPOINT="${CHECKPOINT:-checkpoints_ppo_vqa_fulldataset_4b/ckpt_epoch_24.pt}"
EVAL_MODE="${EVAL_MODE:-ppo}"
ANNOTATION_PATH="$(printf "%s," /scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/recipe_*.json | sed 's/,$//')" 
# ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"
VAL_RATIO="${VAL_RATIO:-}"

CMD=(
  python eval_ppo_vqa.py
  --eval_mode "$EVAL_MODE"
  --checkpoint "$CHECKPOINT"
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --max_samples "$MAX_SAMPLES"
  --print_samples "$PRINT_SAMPLES"
  --progress_every "$PROGRESS_EVERY"
  --video_frames 8
)

if [[ -n "$SAVE_PREDICTIONS" ]]; then
  CMD+=(--save_predictions "$SAVE_PREDICTIONS")
fi

if [[ -n "$VAL_RATIO" ]]; then
  CMD+=(--val_ratio "$VAL_RATIO")
fi

"${CMD[@]}"
