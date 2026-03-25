VL_MODEL_PRESET="${VL_MODEL_PRESET:-internvl3_5_2b}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"
VLM_CHECKPOINT="${VLM_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-2048}"

CMD=(
  accelerate launch --num_processes 4 train_ppo_vqa.py
  --dataset_type hd_epic_local
  --video_root "$VIDEO_ROOT"
  --metadata_root "$METADATA_ROOT"
  --annotation_path "$ANNOTATION_PATH"
  --video_extension mp4
  --val_ratio 0.1
  --train_sampling_mode task_uniform
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --batch_size 64
  --num_workers 4
  --video_frames 8
  --mixed_precision bf16
  --allow_tf32
  --epochs 500
  --max_train_steps "$MAX_TRAIN_STEPS"
  --save_every_steps "$SAVE_EVERY_STEPS"
  --eval_every_steps "$EVAL_EVERY_STEPS"
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

"${CMD[@]}"
