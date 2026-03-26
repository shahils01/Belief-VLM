VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
DATASET_TYPE="${DATASET_TYPE:-hd_epic_local}"   # hd_epic_local | nextqa_local
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/anshuln/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/anshuln/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"
VLM_CHECKPOINT="${VLM_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-2048}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-vlm-ppo-vqa}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

CMD=(
  accelerate launch --num_processes 4 train_ppo_vqa.py
  --dataset_type "$DATASET_TYPE"
  --video_root "$VIDEO_ROOT"
  --metadata_root "$METADATA_ROOT"
  --annotation_path "$ANNOTATION_PATH"
  --video_extension mp4
  --val_ratio 0.1
  --max_val_samples_per_split "$MAX_VAL_SAMPLES"
  --train_sampling_mode task_uniform
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --batch_size 64
  --num_workers 4
  --video_frames 8
  --mixed_precision bf16
  --allow_tf32
  --epochs 500
  --ppo_epochs 10
  --policy_lr 1e-4
  --vlm_lr 2e-5
  --vl_model_preset "$VL_MODEL_PRESET"
  --gradient_checkpointing
  --save_dir checkpoints_ppo_vqa_fulldataset
  --resume_checkpoint checkpoints_ppo_vqa_01/ckpt_epoch_62.pt
)

if [[ -n "$VLM_CHECKPOINT" ]]; then
  CMD+=(--vlm_checkpoint "$VLM_CHECKPOINT")
fi

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--wandb --wandb_project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_RUN_NAME" ]]; then
    CMD+=(--wandb_run_name "$WANDB_RUN_NAME")
  fi
  if [[ -n "$WANDB_TAGS" ]]; then
    CMD+=(--wandb_tags "$WANDB_TAGS")
  fi
fi

"${CMD[@]}"
