import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from data_loading import (
    _get_first,
    _load_records,
    _normalize_text,
    _resolve_hd_epic_clip_window,
    _resolve_hd_epic_video_path,
    _stable_fold,
    decode_mp4_frames,
)
from model import ModelConfig, MultimodalBeliefModel
from train import _configure_memory_optimizations, _load_checkpoint_state, _resolve_vl_model_preset


def parse_args():
    parser = argparse.ArgumentParser(description="Train simple future-observation VQA model on HD-EPIC.")
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--video_extension", type=str, default="mp4")
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

    parser.add_argument("--future_offset_sec", type=float, default=0.5)
    parser.add_argument("--future_window_sec", type=float, default=0.5)
    parser.add_argument("--future_decode_frames", type=int, default=3)
    parser.add_argument("--future_pick_indices", type=str, default="0,2")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_future_vqa")
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
    parser.add_argument("--vl_max_text_len", type=int, default=128)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
    parser.add_argument("--vl_checkpoint", type=str, default="")

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_heads", type=int, default=8)
    return parser.parse_args()


def _parse_choices(raw_choices):
    if isinstance(raw_choices, (list, tuple)):
        return [_normalize_text(x) for x in raw_choices if _normalize_text(x)]
    if isinstance(raw_choices, str):
        text = raw_choices.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [_normalize_text(x) for x in parsed if _normalize_text(x)]
            except Exception:
                pass
        if "\n" in text:
            return [_normalize_text(x) for x in text.split("\n") if _normalize_text(x)]
        if "|" in text:
            return [_normalize_text(x) for x in text.split("|") if _normalize_text(x)]
        return [_normalize_text(text)]
    return []


def _resolve_label(record, args, choices):
    idx = _get_first(record, ["correct_idx", "answer_idx", "label_idx"])
    if idx not in (None, ""):
        idx = int(idx)
        if 0 <= idx < len(choices):
            return idx
        if 1 <= idx <= len(choices):
            return idx - 1
    answer = _normalize_text(_get_first(record, [args.answer_column, "answer"]))
    if answer:
        for i, choice in enumerate(choices):
            if choice.strip().lower() == answer.strip().lower():
                return i
    return None


def _parse_pick_indices(text: str):
    idxs = []
    for token in text.split(","):
        token = token.strip()
        if token:
            idxs.append(int(token))
    if not idxs:
        idxs = [0, 2]
    return idxs


def _select_future_frames(decoded_frames, pick_indices):
    if not decoded_frames:
        raise RuntimeError("No future frames decoded.")
    selected = []
    max_idx = len(decoded_frames) - 1
    for idx in pick_indices:
        idx = min(max(idx, 0), max_idx)
        selected.append(decoded_frames[idx])
    return selected


@dataclass
class FutureVQASample:
    sample_id: str
    question: str
    choices: List[str]
    label: int
    future_frames: List


class HD_EPICFutureVQADataset(Dataset):
    def __init__(self, records, args, is_train: bool):
        self.args = args
        self.pick_indices = _parse_pick_indices(args.future_pick_indices)
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

        question = _normalize_text(
            _get_first(record, [self.args.question_column, "question", "prompt", "instruction", "query"])
        )
        choices = _parse_choices(
            _get_first(record, [self.args.options_column, "options", "choices", "answer_options"])
        )
        if not choices:
            raise RuntimeError(f"Sample {sample_id} has no answer options.")
        label = _resolve_label(record, self.args, choices)
        if label is None:
            raise RuntimeError(f"Sample {sample_id} has no valid label index.")

        video_path = _resolve_hd_epic_video_path(self.args, record)
        _, end_time_sec = _resolve_hd_epic_clip_window(record)
        if end_time_sec is None:
            raise RuntimeError(f"Sample {sample_id} missing clip end time; cannot extract future observations.")

        future_start = float(end_time_sec) + float(self.args.future_offset_sec)
        future_end = future_start + max(float(self.args.future_window_sec), 0.1)
        decoded = decode_mp4_frames(
            video_path,
            num_frames=max(int(self.args.future_decode_frames), 3),
            start_time_sec=future_start,
            end_time_sec=future_end,
        )
        future_frames = _select_future_frames(decoded, self.pick_indices)

        return FutureVQASample(
            sample_id=sample_id,
            question=question,
            choices=choices,
            label=label,
            future_frames=future_frames,
        )



def collate_future_vqa_batch(batch: List[FutureVQASample]):
    return {
        "ids": [x.sample_id for x in batch],
        "questions": [x.question for x in batch],
        "choices": [x.choices for x in batch],
        "labels": torch.tensor([x.label for x in batch], dtype=torch.long),
        "future_frames": [x.future_frames for x in batch],
    }


def build_loader(args, batch_size, num_workers, is_train):
    records = _load_records(args)
    dataset = HD_EPICFutureVQADataset(records=records, args=args, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        collate_fn=collate_future_vqa_batch,
        pin_memory=torch.cuda.is_available(),
    )


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
    for p in model.parameters():
        p.requires_grad = False
    return model


def encode_texts(visual_backbone, texts, device, max_text_len):
    tokenizer = visual_backbone.backbone.tokenizer
    token_embed = visual_backbone.backbone.model.get_input_embeddings()
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        embeds = token_embed(input_ids)
        mask = attention_mask.unsqueeze(-1).to(dtype=embeds.dtype)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled


def flatten_choices(choice_lists):
    flat = []
    counts = []
    for choices in choice_lists:
        flat.extend(choices)
        counts.append(len(choices))
    return flat, counts


def pack_choice_embeddings(choice_embeds, counts, device):
    max_choices = max(counts)
    dim = choice_embeds.shape[-1]
    packed = torch.zeros(len(counts), max_choices, dim, device=device, dtype=choice_embeds.dtype)
    mask = torch.zeros(len(counts), max_choices, device=device, dtype=torch.bool)
    cur = 0
    for i, c in enumerate(counts):
        packed[i, :c] = choice_embeds[cur : cur + c]
        mask[i, :c] = True
        cur += c
    return packed, mask


class FutureObservationVQANet(nn.Module):
    """
    g(ans | v'):
    - v' are future-frame embeddings (e.g., 1st and 3rd frame after T+0.5s)
    - temporal pooling over v'
    - fuse with question embedding
    - score each answer-option embedding
    """

    def __init__(self, visual_dim, text_dim, hidden_dim=768, dropout=0.1, attn_heads=8):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.option_proj = nn.Linear(text_dim, hidden_dim)

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.score_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, future_visual_seq, question_embed, choice_embed, choice_mask):
        v = self.visual_proj(future_visual_seq)
        attended, _ = self.temporal_attn(v, v, v)

        q = self.pool_query.expand(attended.size(0), -1, -1)
        pooled, _ = self.temporal_attn(q, attended, attended)
        pooled = pooled.squeeze(1)

        fused = F.normalize(self.dropout(pooled + self.text_proj(question_embed)), dim=-1)
        options = F.normalize(self.option_proj(choice_embed), dim=-1)

        logits = torch.einsum("bd,bcd->bc", fused, options) * self.score_temp.exp()
        logits = logits.masked_fill(~choice_mask, -1e4)
        return logits


def run_epoch(backbone, model, loader, optimizer, accelerator, args, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for step, batch in enumerate(loader, start=1):
        labels = batch["labels"].to(accelerator.device)

        with torch.no_grad():
            future_embeds = backbone.backbone.extract_clip_embeddings(batch["future_frames"]).to(accelerator.device)
            question_embeds = encode_texts(
                backbone, batch["questions"], accelerator.device, args.vl_max_text_len
            )
            flat_choices, counts = flatten_choices(batch["choices"])
            choice_embeds_flat = encode_texts(
                backbone, flat_choices, accelerator.device, args.vl_max_text_len
            )
            choice_embeds, choice_mask = pack_choice_embeddings(choice_embeds_flat, counts, accelerator.device)

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                logits = model(
                    future_visual_seq=future_embeds,
                    question_embed=question_embeds,
                    choice_embed=choice_embeds,
                    choice_mask=choice_mask,
                )
                loss = F.cross_entropy(logits, labels)
                if train:
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        preds = torch.argmax(logits, dim=-1)
        batch_size = labels.shape[0]
        total += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_acc += float(preds.eq(labels).float().sum().item())

        if args.log_every > 0 and step % args.log_every == 0:
            phase = "train" if train else "val"
            print(
                f"{phase} step={step} "
                f"loss={total_loss / max(total, 1):.4f} "
                f"acc={total_acc / max(total, 1):.4f}"
            )

    denom = max(total, 1)
    return {"loss": total_loss / denom, "acc": total_acc / denom}


def main():
    args = parse_args()
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    backbone = build_visual_backbone(args, accelerator.device)

    probe_loader = build_loader(args, batch_size=1, num_workers=0, is_train=True)
    probe = next(iter(probe_loader))
    probe_visual = backbone.backbone.extract_clip_embeddings(probe["future_frames"]).to(accelerator.device)
    probe_text = encode_texts(backbone, probe["questions"], accelerator.device, args.vl_max_text_len)
    visual_dim = int(probe_visual.shape[-1])
    text_dim = int(probe_text.shape[-1])

    model = FutureObservationVQANet(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        attn_heads=args.attn_heads,
    ).to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = build_loader(args, args.batch_size, args.num_workers, is_train=True)
    val_loader = build_loader(args, args.batch_size, args.num_workers, is_train=False)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    start_epoch = 0
    best_val_acc = -1.0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _load_checkpoint_state(model, ckpt["model"], accelerator)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(backbone, model, train_loader, optimizer, accelerator, args, train=True)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f}"
        )
        with torch.no_grad():
            val_metrics = run_epoch(backbone, model, val_loader, optimizer, accelerator, args, train=False)
        print(
            f"epoch={epoch} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(state, ckpt_path)
            print(f"saved {ckpt_path}")

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_path = os.path.join(args.save_dir, "ckpt_best.pt")
                state["best_val_acc"] = best_val_acc
                torch.save(state, best_path)
                print(f"saved {best_path}")


if __name__ == "__main__":
    main()
