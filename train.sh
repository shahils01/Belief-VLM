python train.py \
  --dataset_name wofmanaf/ego4d-video \
  --dataset_split train \
  --val_ratio 0.01 \
  --batch_size 1 \
  --video_frames 8 \
  --epochs 1 \
  --vl_model_preset llava_onevision_0p5b \
  --save_dir checkpoints_belief_ego4d_local
