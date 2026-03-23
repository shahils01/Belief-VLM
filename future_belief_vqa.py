from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_loading import (
    _get_first,
    _load_records,
    _normalize_text,
    _resolve_hd_epic_future_window,
    _resolve_hd_epic_video_path,
    _stable_fold,
    decode_mp4_frames,
)


def _coerce_options(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_normalize_text(item) for item in value if _normalize_text(item)]
    text = _normalize_text(value)
    if not text:
        return []
    return [item.strip() for item in text.split("\n") if item.strip()]


def _resolve_correct_choice_index(record, num_choices: int) -> int:
    correct_idx = _get_first(record, ["correct_idx", "answer_idx", "label_idx"])
    if correct_idx in (None, ""):
        raise RuntimeError("Record is missing `correct_idx`/`answer_idx`/`label_idx` for multiple-choice belief training.")
    idx = int(correct_idx)
    if 1 <= idx <= num_choices:
        idx -= 1
    if idx < 0 or idx >= num_choices:
        raise RuntimeError(f"Choice index {idx} is out of range for {num_choices} options.")
    return idx


class FutureBeliefVQADataset(Dataset):
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
        video_path = _resolve_hd_epic_video_path(self.args, record)
        future_start_sec, future_end_sec = _resolve_hd_epic_future_window(
            record,
            future_offset_sec=self.args.future_offset_sec,
            future_duration_sec=self.args.future_duration_sec,
        )
        future_frames = decode_mp4_frames(
            video_path,
            self.args.future_frames,
            start_time_sec=future_start_sec,
            end_time_sec=future_end_sec,
        )

        question = _normalize_text(
            _get_first(record, [self.args.question_column, "question", "prompt", "instruction", "query"])
        )
        options = _coerce_options(_get_first(record, [self.args.options_column, "options", "choices", "answer_options"]))
        if not question:
            raise RuntimeError("Record is missing a question field.")
        if len(options) < 2:
            raise RuntimeError("Belief VQA training expects multiple-choice options in the annotation record.")
        label = _resolve_correct_choice_index(record, len(options))

        return {
            "id": str(_get_first(record, [self.args.id_column, "id", "sample_id", "uid", "video_id"]) or index),
            "future_frames": future_frames,
            "question": question,
            "options": options,
            "label": label,
        }


def collate_future_belief_vqa(batch):
    option_counts = {len(item["options"]) for item in batch}
    if len(option_counts) != 1:
        raise RuntimeError("All samples in a batch must have the same number of answer options.")
    return {
        "ids": [item["id"] for item in batch],
        "future_frames": [item["future_frames"] for item in batch],
        "questions": [item["question"] for item in batch],
        "options": [item["options"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }


def build_future_belief_vqa_loader(args, batch_size: int, num_workers: int, is_train: bool):
    records = _load_records(args)
    dataset = FutureBeliefVQADataset(records=records, args=args, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        collate_fn=collate_future_belief_vqa,
        pin_memory=torch.cuda.is_available(),
    )


@dataclass
class FutureBeliefVQAConfig:
    visual_dim: int
    text_dim: int
    hidden_dim: int = 1024
    num_attention_heads: int = 8
    num_fusion_layers: int = 2
    dropout: float = 0.1
    future_frames: int = 2


class FutureBeliefVQAModel(nn.Module):
    def __init__(self, cfg: FutureBeliefVQAConfig):
        super().__init__()
        self.cfg = cfg
        self.visual_proj = nn.Linear(cfg.visual_dim, cfg.hidden_dim)
        self.question_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        self.option_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        self.future_pos_embed = nn.Parameter(torch.zeros(1, cfg.future_frames, cfg.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_attention_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_fusion_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def _pool_contextual_text_embeddings(self, backbone_model, texts: Sequence[str], device: torch.device):
        tokenizer = backbone_model.backbone.tokenizer
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = backbone_model.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Backbone LM did not return hidden_states for contextual text encoding.")
        embeddings = hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (embeddings * mask).sum(dim=1) / denom

    def encode_batch(self, backbone_model, batch, device: torch.device):
        with torch.no_grad():
            future_visual = backbone_model.backbone.extract_clip_embeddings(batch["future_frames"], normalize=False)
        future_visual = future_visual.to(device)
        question_embeddings = self._pool_contextual_text_embeddings(backbone_model, batch["questions"], device)

        flat_options = [option for options in batch["options"] for option in options]
        option_embeddings = self._pool_contextual_text_embeddings(backbone_model, flat_options, device)
        num_options = len(batch["options"][0])
        option_embeddings = option_embeddings.view(len(batch["options"]), num_options, -1)
        return future_visual, question_embeddings, option_embeddings

    def forward(self, backbone_model, batch, device: torch.device):
        future_visual, question_embeddings, option_embeddings = self.encode_batch(backbone_model, batch, device)
        future_hidden = self.visual_proj(future_visual)
        if future_hidden.shape[1] > self.future_pos_embed.shape[1]:
            raise RuntimeError(
                f"Configured future_frames={self.future_pos_embed.shape[1]} but got {future_hidden.shape[1]} frames."
            )
        future_hidden = future_hidden + self.future_pos_embed[:, : future_hidden.shape[1]]
        question_token = self.question_proj(question_embeddings).unsqueeze(1)
        fused = self.fusion(torch.cat([question_token, future_hidden], dim=1))
        context = self.norm(fused[:, 0])

        option_hidden = self.option_proj(option_embeddings)
        context = context.unsqueeze(1).expand_as(option_hidden)
        logits = self.classifier(torch.cat([context, option_hidden], dim=-1)).squeeze(-1)

        labels = batch.get("labels")
        outputs = {"logits": logits}
        if labels is not None:
            labels = labels.to(device)
            outputs["loss"] = F.cross_entropy(logits, labels)
        return outputs
