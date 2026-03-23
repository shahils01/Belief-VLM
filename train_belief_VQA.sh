accelerate launch --num_processes 1 Belief_aware_VLM/Belief-VLM/train_beleif_VQA.py \
  --annotation_path /path/to/fine_grained_why_recognition.json \
  --video_root /path/to/HD-EPIC/Videos \
  --future_offset_sec 0.5 \
  --future_decode_frames 3 \
  --future_pick_indices "0,2" \
  --vl_model_preset internvl3_5_2b \
  --batch_size 2 \
  --epochs 10 \
  --mixed_precision bf16
