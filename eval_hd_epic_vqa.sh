CHECKPOINT="${CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_bundles/ckpt_epoch_38.pt}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRINT_SAMPLES="${PRINT_SAMPLES:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
USE_FUTURE_PREDICTOR="${USE_FUTURE_PREDICTOR:-0}"
FUTURE_PREDICTOR_CHECKPOINT="${FUTURE_PREDICTOR_CHECKPOINT:-}"
FUTURE_FRAMES="${FUTURE_FRAMES:-8}"
USE_BELIEF_MODEL="${USE_BELIEF_MODEL:-1}"
BELIEF_NUM_TOKENS="${BELIEF_NUM_TOKENS:-4}"
BELIEF_TARGET_FRAMES="${BELIEF_TARGET_FRAMES:-2}"
BELIEF_AUX_WEIGHT="${BELIEF_AUX_WEIGHT:-0.2}"
BELIEF_FUTURE_WEIGHT="${BELIEF_FUTURE_WEIGHT:-1.0}"
BELIEF_RECON_WEIGHT="${BELIEF_RECON_WEIGHT:-0.5}"
BELIEF_KL_WEIGHT="${BELIEF_KL_WEIGHT:-0.001}"

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

if [[ "$USE_FUTURE_PREDICTOR" == "1" ]]; then
  CMD+=(--use_future_predictor --future_predictor_checkpoint "$FUTURE_PREDICTOR_CHECKPOINT" --future_frames "$FUTURE_FRAMES")
fi

if [[ "$USE_BELIEF_MODEL" == "1" ]]; then
  CMD+=(
    --use_belief_model
    --belief_num_tokens "$BELIEF_NUM_TOKENS"
    --belief_target_frames "$BELIEF_TARGET_FRAMES"
    --belief_aux_weight "$BELIEF_AUX_WEIGHT"
    --belief_future_weight "$BELIEF_FUTURE_WEIGHT"
    --belief_reconstruction_weight "$BELIEF_RECON_WEIGHT"
    --belief_kl_weight "$BELIEF_KL_WEIGHT"
  )
fi

if [[ -n "$SAVE_PREDICTIONS" ]]; then
  CMD+=(--save_predictions "$SAVE_PREDICTIONS")
fi

"${CMD[@]}"

# /scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_28.pt
