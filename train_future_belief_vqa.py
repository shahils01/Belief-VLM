import argparse
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from future_belief_vqa import build_future_belief_vqa_loader, build_future_vqa_option_batch
from model import ModelConfig, MultimodalBeliefModel
from train import _configure_memory_optimizations, _load_checkpoint_state, _resolve_vl_model_preset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a direct VLM multiple-choice predictor on future HD-EPIC VQA clips."
    )
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--metadata_root", type=str, default="")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--video_id_column", type=str, default="video_id")
    parser.add_argument("--video_path_column", type=str, default="video_path")
    parser.add_argument("--participant_column", type=str, default="participant_id")
    parser.add_argument("--question_column", type=str, default="question")
    parser.add_argument("--answer_column", type=str, default="answer")
    parser.add_argument("--options_column", type=str, default="options")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--future_frames", type=int, default=2)
    parser.add_argument("--future_offset_sec", type=float, default=0.1)
    parser.add_argument("--future_duration_sec", type=float, default=0.001)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_future_belief_vqa")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")

    parser.add_argument("--vl_backend", type=str, default="internvl", choices=["internvl"])
    parser.add_argument("--vl_model_name", type=str, default="OpenGVLab/InternVL3_5-1B-HF")
    parser.add_argument(
        "--vl_model_preset",
        type=str,
        default="internvl3_5_1b",
        choices=["custom", "internvl3_5_1b", "internvl3_5_2b", "internvl3_5_4b", "internvl3_5_8b"],
    )
    parser.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--vl_max_text_len", type=int, default=256)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
    parser.add_argument("--vl_checkpoint", type=str, default="")
    parser.add_argument("--freeze_vl", action="store_true")
    return parser.parse_args()


def build_vlm(args, device):
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        freeze_vl=args.freeze_vl,
        quantization_config=None,
        use_cache=not args.disable_vl_cache,
    )
    model = MultimodalBeliefModel(cfg, device=device)
    _configure_memory_optimizations(model, args)
    if args.vl_checkpoint:
        ckpt = torch.load(args.vl_checkpoint, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        try:
            model.load_state_dict(state)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
    return model


def _compute_choice_scores(logits, labels, batch_size: int, num_options: int):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    seq_loss = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return -seq_loss.view(batch_size, num_options)


def run_epoch(model, loader, optimizer, accelerator, args, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0.0
    base_model = accelerator.unwrap_model(model)
    processor = base_model.backbone.processor

    for step, batch in enumerate(loader, start=1):
        collated, labels, num_options = build_future_vqa_option_batch(processor, batch, args)
        labels = labels.to(accelerator.device)
        flat_labels = collated["labels"].to(accelerator.device)

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                outputs = model(collated["inputs"], labels=flat_labels)
                choice_scores = _compute_choice_scores(outputs["logits"], flat_labels, labels.shape[0], num_options)
                loss = F.cross_entropy(choice_scores, labels)
                if train:
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        preds = choice_scores.argmax(dim=-1)
        batch_size = labels.shape[0]
        batch_stats = torch.stack(
            [
                loss.detach() * batch_size,
                (preds == labels).sum().detach().to(loss.dtype),
                torch.tensor(batch_size, device=accelerator.device, dtype=loss.dtype),
            ]
        )
        batch_stats = accelerator.gather_for_metrics(batch_stats.unsqueeze(0)).sum(dim=0)
        total_loss += float(batch_stats[0].item())
        total_correct += float(batch_stats[1].item())
        total_examples += float(batch_stats[2].item())

        if accelerator.is_main_process and args.log_every > 0 and step % args.log_every == 0:
            phase = "train" if train else "val"
            denom = max(total_examples, 1.0)
            accelerator.print(
                f"{phase} step={step} "
                f"loss={total_loss / denom:.4f} "
                f"acc={total_correct / denom:.4f}"
            )

    denom = max(total_examples, 1.0)
    return {
        "loss": total_loss / denom,
        "accuracy": total_correct / denom,
    }


def main():
    args = parse_args()
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    model = build_vlm(args, accelerator.device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    accelerator.print(f"parameters total={total_params:,} trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader = build_future_belief_vqa_loader(args, args.batch_size, args.num_workers, is_train=True)
    val_loader = build_future_belief_vqa_loader(args, args.batch_size, args.num_workers, is_train=False)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    start_epoch = 0
    best_val = float("-inf")
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _load_checkpoint_state(model, ckpt["model"], accelerator)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val_accuracy", float("-inf")))

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, accelerator, args, True)
        accelerator.print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f}"
        )

        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, accelerator, args, False)
        accelerator.print(
            f"epoch={epoch} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_accuracy": best_val,
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(state, ckpt_path)
            accelerator.print(f"saved {ckpt_path}")

            if val_metrics["accuracy"] > best_val:
                best_val = val_metrics["accuracy"]
                state["best_val_accuracy"] = best_val
                best_path = os.path.join(args.save_dir, "ckpt_best.pt")
                torch.save(state, best_path)
                accelerator.print(f"saved {best_path}")


if __name__ == "__main__":
    main()
