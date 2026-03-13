python train.py \
  --dataset_name HuggingFaceM4/something_something_v2 \
  --train_split train \
  --val_split validation \
  --batch_size 1 \
  --video_frames 8 \
  --epochs 1 \
  --vl_model_preset llava_onevision_0p5b \
  --text_prompt_template "Classify the human action shown in this video. Text context: {text}" \
  --save_dir checkpoints_belief_local
