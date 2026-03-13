VL_MODEL_PRESET="${VL_MODEL_PRESET:-llava_onevision_0p5b}"

accelerate launch --num_processes 4 train.py \
  --dataset_name HuggingFaceM4/something_something_v2 \
  --train_split train \
  --val_split validation \
  --batch_size 1 \
  --num_workers 0 \
  --video_frames 8 \
  --grad_accum_steps 16 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --epochs 5 \
  --log_every 20 \
  --vl_model_preset "$VL_MODEL_PRESET" \
  --value_pooling hidden_mean \
  --text_prompt_template "Classify the human action shown in this video. Text context: {text}" \
  --peft lora \
  --gradient_checkpointing \
  --fsdp \
  --save_dir checkpoints_belief_fsdp
