CHECKPOINT="${CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_ca_belief/ckpt_epoch_70.pt}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
USE_BELIEF_MODEL="${USE_BELIEF_MODEL:-1}"
BELIEF_NUM_TOKENS="${BELIEF_NUM_TOKENS:-4}"

CMD=(
  python eval_hd_epic_vqa.py
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --max_samples "$MAX_SAMPLES"
  --print_samples "$PRINT_SAMPLES"
  --progress_every "$PROGRESS_EVERY"
  --vl_model_preset "$VL_MODEL_PRESET"
  --video_frames "20"
  --mixed_precision bf16
  --allow_tf32
)

if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(
    --checkpoint "$CHECKPOINT"
    )
fi

if [[ "$USE_BELIEF_MODEL" == "1" ]]; then
  CMD+=(
    --use_belief_model
    --belief_num_tokens "$BELIEF_NUM_TOKENS"
  )
fi

if [[ -n "$SAVE_PREDICTIONS" ]]; then
  CMD+=(--save_predictions "$SAVE_PREDICTIONS")
fi

"${CMD[@]}"

# /scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_28.pt
