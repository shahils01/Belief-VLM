import argparse
import os

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from Belief_Network import AttentionConfig, RecursiveBeliefNetwork
from data_loading import (
    _get_first,
    _load_records,
    _resolve_hd_epic_clip_window,
    _resolve_hd_epic_video_path,
    _stable_fold,
    decode_mp4_frames,
)
from model import ModelConfig, MultimodalBeliefModel
from train import _configure_memory_optimizations, _load_checkpoint_state, _resolve_vl_model_preset


def parse_args():
    parser = argparse.ArgumentParser(description="Train recursive belief network with ELBO on HD-EPIC clips.")
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--video_id_column", type=str, default="video_id")
    parser.add_argument("--video_path_column", type=str, default="video_path")
    parser.add_argument("--participant_column", type=str, default="participant_id")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--video_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_belief_network")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")

    parser.add_argument("--vl_backend", type=str, default="internvl", choices=["internvl"])
    parser.add_argument("--vl_model_name", type=str, default="OpenGVLab/InternVL3_5-2B-HF")
    parser.add_argument(
        "--vl_model_preset",
        type=str,
        default="internvl3_5_2b",
        choices=["custom", "internvl3_5_1b", "internvl3_5_2b", "internvl3_5_4b", "internvl3_5_8b"],
    )
    parser.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
    parser.add_argument("--vl_checkpoint", type=str, default="")

    parser.add_argument("--state_dim", type=int, default=12)
    parser.add_argument("--belief_dim", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--proj_dropout", type=float, default=0.2)

    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--temporal_nce_weight", type=float, default=1.0)
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


class BeliefVideoDataset(Dataset):
    def __init__(self, records, args, is_train: bool):
        self.args = args
        selected = []
        for idx, record in enumerate(records):
            sample_id = str(_get_first(record, [args.id_column, "id", "sample_id", "uid", "video_id"]) or idx)
            val_ratio = max(0.0, min(0.5, float(args.val_ratio)))
            if val_ratio > 0.0:
                fold = _stable_fold(sample_id, args.seed)
                in_val = fold < val_ratio
                if is_train and in_val:
                    continue
                if (not is_train) and (not in_val):
                    continue
            selected.append(record)
            if args.max_samples_per_split > 0 and len(selected) >= args.max_samples_per_split:
                break
        self.records = selected

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sample_id = str(_get_first(record, [self.args.id_column, "id", "sample_id", "uid", "video_id"]) or index)
        video_path = _resolve_hd_epic_video_path(self.args, record)
        start_time_sec, end_time_sec = _resolve_hd_epic_clip_window(record)
        frames = decode_mp4_frames(
            video_path,
            self.args.video_frames,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
        )
        return {"id": sample_id, "frames": frames}


def collate_video_batch(batch):
    return {
        "ids": [item["id"] for item in batch],
        "frames": [item["frames"] for item in batch],
    }


def build_belief_loader(args, batch_size: int, num_workers: int, is_train: bool):
    records = _load_records(args)
    dataset = BeliefVideoDataset(records=records, args=args, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        collate_fn=collate_video_batch,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(visual_backbone, belief_net, loader, optimizer, accelerator, args, train):
    belief_net.train() if train else belief_net.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_nce = 0.0
    total_examples = 0

    for step, batch in enumerate(loader, start=1):
        with torch.no_grad():
            visual_embeddings = visual_backbone.backbone.extract_clip_embeddings(batch["frames"]).to(accelerator.device)
            state_seq = torch.zeros(
                visual_embeddings.shape[0],
                visual_embeddings.shape[1],
                args.state_dim,
                device=accelerator.device,
                dtype=visual_embeddings.dtype,
            )

        with accelerator.accumulate(belief_net):
            with torch.set_grad_enabled(train):
                outputs = belief_net(visual_seq=visual_embeddings, state_seq=state_seq)
                loss = outputs["loss"]
                if train:
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(belief_net.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = visual_embeddings.size(0)
        total_examples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_recon += float(outputs["recon_loss"].detach().item()) * batch_size
        total_kl += float(outputs["kl_loss"].detach().item()) * batch_size
        total_nce += float(outputs["temporal_nce_loss"].detach().item()) * batch_size

        if args.log_every > 0 and step % args.log_every == 0:
            phase = "train" if train else "val"
            denom = max(total_examples, 1)
            print(
                f"{phase} step={step} "
                f"loss={total_loss / denom:.4f} "
                f"recon={total_recon / denom:.4f} "
                f"kl={total_kl / denom:.4f} "
                f"nce={total_nce / denom:.4f}"
            )

    denom = max(total_examples, 1)
    return {
        "loss": total_loss / denom,
        "recon_loss": total_recon / denom,
        "kl_loss": total_kl / denom,
        "temporal_nce_loss": total_nce / denom,
    }


def main():
    args = parse_args()
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    visual_backbone = build_visual_backbone(args, accelerator.device)

    probe_loader = build_belief_loader(args, batch_size=1, num_workers=0, is_train=True)
    probe_batch = next(iter(probe_loader))
    probe_visual = visual_backbone.backbone.extract_clip_embeddings(probe_batch["frames"]).to(accelerator.device)
    visual_dim = int(probe_visual.shape[-1])

    belief_cfg = AttentionConfig(
        num_attention_heads=args.num_attention_heads,
        state_dim=args.state_dim,
        belief_dim=args.belief_dim,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        attention_dropout=args.attention_dropout,
        proj_dropout=args.proj_dropout,
    )
    belief_net = RecursiveBeliefNetwork(
        config=belief_cfg,
        visual_dim=visual_dim,
        beta=args.beta,
        recon_weight=args.recon_weight,
        temporal_nce_weight=args.temporal_nce_weight,
        device=accelerator.device,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(belief_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = build_belief_loader(args, args.batch_size, args.num_workers, is_train=True)
    val_loader = build_belief_loader(args, args.batch_size, args.num_workers, is_train=False)

    belief_net, optimizer, train_loader, val_loader = accelerator.prepare(
        belief_net, optimizer, train_loader, val_loader
    )

    start_epoch = 0
    best_val = float("inf")
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _load_checkpoint_state(belief_net, ckpt["belief_net"], accelerator)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val_loss", float("inf")))

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(visual_backbone, belief_net, train_loader, optimizer, accelerator, args, True)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_recon={train_metrics['recon_loss']:.4f} "
            f"train_kl={train_metrics['kl_loss']:.4f} "
            f"train_nce={train_metrics['temporal_nce_loss']:.4f}"
        )

        with torch.no_grad():
            val_metrics = run_epoch(visual_backbone, belief_net, val_loader, optimizer, accelerator, args, False)
        print(
            f"epoch={epoch} val_loss={val_metrics['loss']:.4f} "
            f"val_recon={val_metrics['recon_loss']:.4f} "
            f"val_kl={val_metrics['kl_loss']:.4f} "
            f"val_nce={val_metrics['temporal_nce_loss']:.4f}"
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state = {
                "belief_net": accelerator.unwrap_model(belief_net).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val,
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(state, ckpt_path)
            print(f"saved {ckpt_path}")

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_path = os.path.join(args.save_dir, "ckpt_best.pt")
                state["best_val_loss"] = best_val
                torch.save(state, best_path)
                print(f"saved {best_path}")


if __name__ == "__main__":
    main()
