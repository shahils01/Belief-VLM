VL_MODEL_PRESET="${VL_MODEL_PRESET:-llava_onevision_0p5b}"
DATASET_NAME="${DATASET_NAME:-wofmanaf/ego4d-video}"
VIDEO_ROOT="${VIDEO_ROOT:-}"

# Required for this dataset in practice:
# rows reference filenames like EGO_225484.npy, but the HF repo stores large archive parts.
# Point VIDEO_ROOT at the locally extracted .npy directory.

CMD=(
  accelerate launch --num_processes 4 train.py
  --dataset_name "$DATASET_NAME"
  --dataset_split train
  --val_ratio 0.01
  --batch_size 1
  --num_workers 0
  --video_frames 8
  --grad_accum_steps 16
  --mixed_precision bf16
  --allow_tf32
  --epochs 3
  --log_every 20
  --vl_model_preset "$VL_MODEL_PRESET"
  --peft lora
  --gradient_checkpointing
  --fsdp
  --save_dir checkpoints_belief_ego4d_fsdp
)

if [ -n "$VIDEO_ROOT" ]; then
  CMD+=(--video_root "$VIDEO_ROOT")
fi

"${CMD[@]}"
