CHECKPOINT="${CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_future_full_finetune/ckpt_epoch_21.pt}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
PREDICTIVE_MODULE="${PREDICTIVE_MODULE:-future}"
USE_FUTURE_PREDICTOR="${USE_FUTURE_PREDICTOR:-1}"
FUTURE_PREDICTOR_CHECKPOINT="${FUTURE_PREDICTOR_CHECKPOINT:-}"
FUTURE_FRAMES="${FUTURE_FRAMES:-2}"
USE_BELIEF_NETWORK="${USE_BELIEF_NETWORK:-0}"
BELIEF_NETWORK_CHECKPOINT="${BELIEF_NETWORK_CHECKPOINT:-}"

CMD=(
  python eval_hd_epic_vqa.py
  --annotation_path "$ANNOTATION_PATH"
  --video_root "$VIDEO_ROOT"
  --max_samples "$MAX_SAMPLES"
  --print_samples "$PRINT_SAMPLES"
  --progress_every "$PROGRESS_EVERY"
  --vl_model_preset "$VL_MODEL_PRESET"
  --video_frames "10"
  --mixed_precision bf16
  --allow_tf32
)

if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

if [[ "$PREDICTIVE_MODULE" == "future" || "$USE_FUTURE_PREDICTOR" == "1" ]]; then
  CMD+=(--use_future_predictor --future_predictor_checkpoint "$FUTURE_PREDICTOR_CHECKPOINT" --future_frames "$FUTURE_FRAMES")
fi

if [[ "$PREDICTIVE_MODULE" == "belief" || "$USE_BELIEF_NETWORK" == "1" ]]; then
  CMD+=(--use_belief_network --belief_network_checkpoint "$BELIEF_NETWORK_CHECKPOINT")
fi

if [[ -n "$SAVE_PREDICTIONS" ]]; then
  CMD+=(--save_predictions "$SAVE_PREDICTIONS")
fi

"${CMD[@]}"

# /scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_28.pt
