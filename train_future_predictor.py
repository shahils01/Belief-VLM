import argparse
import os

import torch
from accelerate import Accelerator

from data_loading import build_future_prediction_loader
from future_prediction import FuturePredictionLoss, FuturePredictionTransformer, FuturePredictorConfig
from model import ModelConfig, MultimodalBeliefModel
from train import _configure_memory_optimizations, _load_checkpoint_state, _resolve_vl_model_preset


def parse_args():
    parser = argparse.ArgumentParser(description="Train future embedding predictor on HD-EPIC VQA clips.")
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

    parser.add_argument("--video_frames", type=int, default=8)
    parser.add_argument("--future_frames", type=int, default=8)
    parser.add_argument("--future_offset_sec", type=float, default=0.0)
    parser.add_argument("--future_duration_sec", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_future_predictor")
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

    parser.add_argument("--predictor_hidden_dim", type=int, default=1024)
    parser.add_argument("--predictor_layers", type=int, default=2)
    parser.add_argument("--predictor_heads", type=int, default=8)
    parser.add_argument("--predictor_dropout", type=float, default=0.1)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--cosine_weight", type=float, default=1.0)
    return parser.parse_args()


def build_visual_backbone(args, device):
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        freeze_vl=True,
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
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def encode_batch(backbone_model, batch, device):
    with torch.no_grad():
        context_embeddings = backbone_model.backbone.extract_clip_embeddings(batch["context_frames"]).to(device)
        future_embeddings = backbone_model.backbone.extract_clip_embeddings(batch["future_frames"]).to(device)
    return context_embeddings, future_embeddings


def run_epoch(visual_backbone, predictor, loss_fn, loader, optimizer, accelerator, args, train):
    predictor.train() if train else predictor.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_examples = 0

    for step, batch in enumerate(loader, start=1):
        context_embeddings, future_embeddings = encode_batch(visual_backbone, batch, accelerator.device)
        with accelerator.accumulate(predictor):
            with torch.set_grad_enabled(train):
                pred_future = predictor(context_embeddings, future_frames=future_embeddings.shape[1])
                metrics = loss_fn(pred_future, future_embeddings)
                loss = metrics["loss"]
                if train:
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(predictor.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = context_embeddings.size(0)
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_mse += metrics["mse"].detach().item() * batch_size
        total_cosine += metrics["cosine"].detach().item() * batch_size

        if args.log_every > 0 and step % args.log_every == 0:
            phase = "train" if train else "val"
            print(
                f"{phase} step={step} "
                f"loss={total_loss / max(total_examples, 1):.4f} "
                f"mse={total_mse / max(total_examples, 1):.4f} "
                f"cosine={total_cosine / max(total_examples, 1):.4f}"
            )

    denom = max(total_examples, 1)
    return {
        "loss": total_loss / denom,
        "mse": total_mse / denom,
        "cosine": total_cosine / denom,
    }


def main():
    args = parse_args()
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    visual_backbone = build_visual_backbone(args, accelerator.device)

    probe_frames = build_future_prediction_loader(args, "train", batch_size=1, num_workers=0, is_train=True)
    probe_batch = next(iter(probe_frames))
    probe_embeddings = encode_batch(visual_backbone, probe_batch, accelerator.device)[0]
    embed_dim = int(probe_embeddings.shape[-1])
    args.predictor_embed_dim = embed_dim
    predictor_cfg = FuturePredictorConfig(
        embed_dim=embed_dim,
        hidden_dim=args.predictor_hidden_dim,
        num_layers=args.predictor_layers,
        num_heads=args.predictor_heads,
        dropout=args.predictor_dropout,
        max_context_frames=args.video_frames,
        max_future_frames=args.future_frames,
    )
    predictor = FuturePredictionTransformer(predictor_cfg).to(accelerator.device)
    loss_fn = FuturePredictionLoss(mse_weight=args.mse_weight, cosine_weight=args.cosine_weight)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = build_future_prediction_loader(args, "train", args.batch_size, args.num_workers, is_train=True)
    val_loader = build_future_prediction_loader(args, "validation", args.batch_size, args.num_workers, is_train=False)

    predictor, optimizer, train_loader, val_loader = accelerator.prepare(predictor, optimizer, train_loader, val_loader)

    start_epoch = 0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _load_checkpoint_state(predictor, ckpt["predictor"], accelerator)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(visual_backbone, predictor, loss_fn, train_loader, optimizer, accelerator, args, True)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_mse={train_metrics['mse']:.4f} train_cosine={train_metrics['cosine']:.4f}"
        )

        with torch.no_grad():
            val_metrics = run_epoch(visual_backbone, predictor, loss_fn, val_loader, optimizer, accelerator, args, False)
        print(
            f"epoch={epoch} val_loss={val_metrics['loss']:.4f} "
            f"val_mse={val_metrics['mse']:.4f} val_cosine={val_metrics['cosine']:.4f}"
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(
                {
                    "predictor": accelerator.unwrap_model(predictor).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
