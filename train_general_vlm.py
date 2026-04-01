import argparse
import json
import os
import random
import time
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset

from datasets_registry import get_adapter, list_adapters
from general_data_utils import (
    build_multimodal_sft_example,
    collate_sft_batch,
    decode_media,
)
from train import _apply_peft, _configure_memory_optimizations, _resolve_vl_model_preset, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generalized multimodal SFT training for InternVL benchmarks.")
    parser.add_argument("--dataset_name", type=str, default="ai2d", choices=list_adapters())
    parser.add_argument("--dataset_repo", type=str, default="")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--dataset_revision", type=str, default="main")
    parser.add_argument("--train_split", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trust_remote_code_dataset", action="store_true", default=False)
    parser.add_argument("--annotation_path", type=str, default="")
    parser.add_argument("--dataset_media_type", type=str, default="", choices=["", "image", "video"])
    parser.add_argument("--media_root", type=str, default="")
    parser.add_argument("--video_root", type=str, default="")
    parser.add_argument("--media_column", type=str, default="media")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--video_column", type=str, default="video")
    parser.add_argument("--question_column", type=str, default="question")
    parser.add_argument("--answer_column", type=str, default="answer")
    parser.add_argument("--options_column", type=str, default="options")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--video_frames", type=int, default=8)

    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")

    parser.add_argument("--vl_backend", type=str, default="internvl", choices=["internvl"])
    parser.add_argument(
        "--vl_model_preset",
        type=str,
        default="internvl3_5_2b",
        choices=["custom", "internvl3_5_1b", "internvl3_5_2b", "internvl3_5_4b", "internvl3_5_8b"],
    )
    parser.add_argument("--vl_model_name", type=str, default="OpenGVLab/InternVL3_5-2B-HF")
    parser.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--vl_max_text_len", type=int, default=256)
    parser.add_argument("--freeze_vl", action="store_true")
    parser.add_argument("--peft", type=str, default="lora", choices=["none", "lora", "qlora"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_general_vlm")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--load_model_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="general-vlm")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    return parser.parse_args()


def _build_quant_config(args):
    if args.peft != "qlora":
        return None
    from transformers import BitsAndBytesConfig

    if args.vl_dtype == "float16":
        compute_dtype = torch.float16
    elif args.vl_dtype == "float32":
        compute_dtype = torch.float32
    else:
        compute_dtype = torch.bfloat16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SFTCollator:
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args

    def __call__(self, batch):
        items = []
        for sample in batch:
            media = decode_media(
                sample["media_type"],
                sample["media"],
                video_frames=self.args.video_frames,
                metadata=sample.get("metadata") or {},
            )
            packed = build_multimodal_sft_example(
                processor=self.processor,
                media_type=sample["media_type"],
                media=media,
                prompt=sample["prompt"],
                answer=sample["target_text"],
                vl_backend=self.args.vl_backend,
                max_text_len=self.args.vl_max_text_len,
            )
            model_inputs = {
                key: value
                for key, value in packed.items()
                if torch.is_tensor(value) and key != "labels"
            }
            items.append(
                {
                    "id": sample["id"],
                    "task_name": sample["task_name"],
                    "prompt": sample["prompt"],
                    "target_text": sample["target_text"],
                    "metadata": sample.get("metadata", {}),
                    "choices": sample.get("choices"),
                    "correct_idx": sample.get("correct_idx"),
                    "media_type": sample["media_type"],
                    "labels": packed["labels"],
                    "inputs": model_inputs,
                }
            )
        return collate_sft_batch(items)


def _load_checkpoint(model, optimizer, accelerator, args):
    if not args.resume_checkpoint:
        return 0, 0
    ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    accelerator.unwrap_model(model).load_state_dict(state_dict, strict=False)
    if not args.load_model_only and isinstance(ckpt, dict):
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))
    return 0, 0


def _save_checkpoint(model, optimizer, accelerator, args, epoch, global_step):
    if not accelerator.is_main_process:
        return
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
    torch.save(
        {
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    _set_seed(args.seed)
    _resolve_vl_model_preset(args)
    args.quantization_config = _build_quant_config(args)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)
    if accelerator.is_main_process and args.wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.wandb_run_name or None,
                config=vars(args),
            )
        except Exception:
            accelerator.print("wandb initialization failed; continuing without wandb.")

    adapter = get_adapter(args.dataset_name)
    model = build_model(args, device=accelerator.device)
    model = _apply_peft(model, args)
    _configure_memory_optimizations(model, args)
    processor = accelerator.unwrap_model(model).backbone.processor

    train_dataset = adapter.build_train_dataset(args)
    shuffle = not isinstance(train_dataset, IterableDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=SFTCollator(processor, args),
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    start_epoch, global_step = _load_checkpoint(model, optimizer, accelerator, args)

    accelerator.print(f"dataset={args.dataset_name} trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        start_time = time.time()
        for step, batch in enumerate(train_loader, start=1):
            inputs = {k: v.to(accelerator.device) if torch.is_tensor(v) else v for k, v in batch["inputs"].items()}
            labels = batch["labels"].to(accelerator.device)
            with accelerator.accumulate(model):
                outputs = model(inputs, labels=labels)
                loss = outputs["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += float(loss.detach().item()) * labels.size(0)
            running_count += int(labels.size(0))
            global_step += 1
            if step % args.log_every == 0:
                avg_loss = running_loss / max(1, running_count)
                elapsed = time.time() - start_time
                accelerator.print(
                    f"epoch={epoch} step={step} global_step={global_step} loss={avg_loss:.4f} "
                    f"samples={running_count} sec={elapsed:.1f}"
                )
        _save_checkpoint(model, optimizer, accelerator, args, epoch + 1, global_step)

    if accelerator.is_main_process and args.wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
