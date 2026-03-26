VL_MODEL_PRESET="${VL_MODEL_PRESET:-custom}"
VL_MODEL_NAME="${VL_MODEL_NAME:-/scratch/shahils/hf_models/InternVL3_5-2B-HF}"
VIDEO_ROOT="${VIDEO_ROOT:-/scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/scratch/shahils/hd_epic_dataset/hd-epic-annotations/vqa-benchmark/}"
METADATA_ROOT="${METADATA_ROOT:-/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data}"
VLM_CHECKPOINT="${VLM_CHECKPOINT:-/scratch/shahils/Belief-VLM/checkpoints_belief_hd_epic_ddp_07/ckpt_epoch_99.pt}"
ANSWER_HEAD_CHECKPOINT="${ANSWER_HEAD_CHECKPOINT:-checkpoints_ppo_vqa_fulldataset_01/ckpt_epoch_241.pt}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-5000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-128}"

CMD=(
  accelerate launch --num_processes 2 train_db_prior.py
  --dataset_type hd_epic_local
  --video_root "$VIDEO_ROOT"
  --metadata_root "$METADATA_ROOT"
  --annotation_path "$ANNOTATION_PATH"
  --video_extension mp4
  --val_ratio 0.1
  --max_val_samples_per_split "$MAX_VAL_SAMPLES"
  --train_sampling_mode task_uniform
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --batch_size 4
  --num_workers 2
  --video_frames 5
  --mixed_precision bf16
  --allow_tf32
  --epochs 50000
  --ppo_epochs 4
  --policy_lr 1e-4
  --selector_lr 1e-4
  --vlm_lr 2e-5
  --use_rl_answer_head
  --prior_top_k 4
  --retrieval_embedder_model sentence-transformers/all-MiniLM-L6-v2
  --vl_model_preset "$VL_MODEL_PRESET"
  --vl_model_name "$VL_MODEL_NAME"
  --train_vlm_with_rl
  --peft lora
  --gradient_checkpointing
  --grad_accum_steps 64
  --save_dir checkpoints_db_prior_rlPi_lora
  --wandb
  --wandb_run_name vdb_rlPi_lora
)

if [[ -n "$VLM_CHECKPOINT" ]]; then
  CMD+=(--vlm_checkpoint "$VLM_CHECKPOINT")
fi

if [[ -n "$ANSWER_HEAD_CHECKPOINT" ]]; then
  CMD+=(--answer_head_checkpoint "$ANSWER_HEAD_CHECKPOINT")
fi

"${CMD[@]}"


# Notes:
# 1) wandb_run_name: vdb_rl2 :: 2 Rl policies (one for topK and one for answer head)
# 2) wandb_run_name: vdb_rl :: RL policy for topK
# 3) wandb_run_name: vdb_rlPi :: RL policy as answer head
# 4) wandb_run_name: vdb :: No RL policies