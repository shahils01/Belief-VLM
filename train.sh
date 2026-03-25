python train.py \
  --dataset_type hd_epic_local \
  --video_root /scratch/shahils/hd_epic_dataset/videos/HD-EPIC/Videos \
  --metadata_root "/scratch/shahils/hd_epic_dataset/HD-EPIC Intermediate Data" \
  --val_ratio 0.01 \
  --batch_size 1 \
  --video_frames 8 \
  --epochs 1 \
  --vl_model_preset internvl3_5_1b \
  --save_dir checkpoints_vlm_hd_epic_local
