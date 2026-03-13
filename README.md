# Belief-VLM

`Belief-VLM` is now configured for supervised multimodal fine-tuning on `wofmanaf/ego4d-video`.
The training objective is causal language modeling over the assistant response in each `conversations` sample, conditioned on the video plus the user prompt.

## Data format
The current loader targets the Hugging Face dataset `wofmanaf/ego4d-video`.
Each sample is expected to contain:
- `id`
- `video`: a path-like string such as `EGO_1.npy`
- `conversations`: user/assistant turns

The loader resolves the `.npy` video file either from `--video_root` or by downloading it from the dataset repo with `huggingface_hub`.
It samples `--video_frames` frames uniformly and masks the prompt tokens so the loss is applied only to the assistant answer.

## Loss
Training uses standard causal LM cross-entropy with label masking:

```text
L = CrossEntropy(next_token_logits, assistant_tokens_only)
```

This is the correct objective for `ego4d-video` because the supervision is free-form text, not class IDs.

## Train/val split
The dataset exposes only a `train` split on Hugging Face.
This code creates a deterministic validation holdout from that split using `--val_ratio` and the sample `id`.

## Training
Example DDP run:

```bash
accelerate launch --num_processes 2 train.py \
  --dataset_name wofmanaf/ego4d-video \
  --dataset_split train \
  --val_ratio 0.01 \
  --batch_size 1 \
  --video_frames 8 \
  --mixed_precision bf16 \
  --grad_accum_steps 16 \
  --vl_model_preset llava_onevision_0p5b \
  --peft qlora \
  --gradient_checkpointing \
  --save_dir checkpoints_belief_ego4d
```

If you already downloaded the `.npy` files locally, pass `--video_root /path/to/ego4d_npy` to avoid per-sample hub fetches.

## Notes
- `wofmanaf/ego4d-video` is not a classification dataset, so the old label-classifier path has been replaced with generative SFT.
- The loader expects each `.npy` file to contain a time-major frame array. If the actual file layout differs, the frame decoder in `data_loading.py` will need one more adjustment.
- For large models, `--peft qlora` or `--peft lora` is the practical starting point.
