# Belief-VLM

`Belief-VLM` is now a supervised video-text classification pipeline for human-belief or action prediction tasks.
The model takes a decoded video clip plus a text prompt, pools the VLM representation, and trains a classification head against dataset labels.

## What changed
- Removed the robot observation, graph, GNN, reward, TD, and contrastive training path from the main training pipeline.
- Replaced the old WebDataset trajectory loader with a Hugging Face video dataset loader.
- Switched the training objective to supervised classification with cross-entropy loss.

## Recommended loss
For Something-Something V2, use standard multiclass cross-entropy:

```text
L = CrossEntropy(logits, class_label)
```

This is the right default because each clip has one action label. If you see overconfidence, add light `--label_smoothing 0.05`.

## Dataset support
The default configuration targets the Hugging Face dataset `HuggingFaceM4/something_something_v2`.
The loader expects:
- a decoded `video` column
- a text column such as `text`
- an integer `label` column

You can override those with `--video_column`, `--text_column`, and `--label_column`.

## Training
Single node, 2 GPUs:

```bash
accelerate launch --num_processes 2 train.py \
  --dataset_name HuggingFaceM4/something_something_v2 \
  --train_split train \
  --val_split validation \
  --batch_size 2 \
  --video_frames 8 \
  --mixed_precision bf16 \
  --grad_accum_steps 8 \
  --vl_model_preset llava_onevision_0p5b \
  --peft qlora \
  --gradient_checkpointing \
  --text_prompt_template "Classify the human action shown in this video. Text context: {text}" \
  --save_dir checkpoints_belief
```

## Notes
- Something-Something V2 is a single-label action dataset, so classification is simpler and better aligned than TD/value regression.
- The text field in this dataset is usually a template-style action description. Keep it if you want a video+text model; drop or randomize it only if you want to force a mostly-video classifier.
- For large VLMs, start with `--peft qlora` or `--peft lora` instead of full fine-tuning.
- Video decoding through Hugging Face datasets typically needs `datasets[video]` and a backend such as `torchcodec`.
