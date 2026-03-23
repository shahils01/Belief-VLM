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
    num_belief_tokens: int = 4


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, context, context_mask=None):
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()
        norm_query = self.query_norm(query)
        norm_context = self.context_norm(context)
        attended, _ = self.cross_attn(
            norm_query,
            norm_context,
            norm_context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + attended
        query = query + self.mlp(self.mlp_norm(query))
        return query


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        norm_x = self.norm(x)
        attended, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attended
        x = x + self.mlp(self.mlp_norm(x))
        return x


class FutureBeliefVQAModel(nn.Module):
    def __init__(self, cfg: FutureBeliefVQAConfig):
        super().__init__()
        self.cfg = cfg
        self.visual_proj = nn.Linear(cfg.visual_dim, cfg.hidden_dim)
        self.question_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        self.option_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        self.future_pos_embed = nn.Parameter(torch.zeros(1, cfg.future_frames, cfg.hidden_dim))
        self.belief_tokens = nn.Parameter(torch.zeros(1, cfg.num_belief_tokens, cfg.hidden_dim))
        self.text_to_belief = nn.ModuleList(
            [CrossAttentionBlock(cfg.hidden_dim, cfg.num_attention_heads, cfg.dropout) for _ in range(cfg.num_fusion_layers)]
        )
        self.vision_to_belief = nn.ModuleList(
            [CrossAttentionBlock(cfg.hidden_dim, cfg.num_attention_heads, cfg.dropout) for _ in range(cfg.num_fusion_layers)]
        )
        self.belief_self_attn = nn.ModuleList(
            [SelfAttentionBlock(cfg.hidden_dim, cfg.num_attention_heads, cfg.dropout) for _ in range(cfg.num_fusion_layers)]
        )
        self.option_to_belief = CrossAttentionBlock(cfg.hidden_dim, cfg.num_attention_heads, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def _encode_contextual_text(self, backbone_model, texts: Sequence[str], device: torch.device):
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
        return embeddings, attention_mask.bool()

    def _pool_masked_tokens(self, embeddings, mask):
        mask_f = mask.unsqueeze(-1).to(embeddings.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (embeddings * mask_f).sum(dim=1) / denom

    def encode_batch(self, backbone_model, batch, device: torch.device):
        with torch.no_grad():
            future_visual = backbone_model.backbone.extract_clip_embeddings(batch["future_frames"], normalize=False)
        future_visual = future_visual.to(device)
        question_tokens, question_mask = self._encode_contextual_text(backbone_model, batch["questions"], device)

        flat_options = [option for options in batch["options"] for option in options]
        option_tokens, option_mask = self._encode_contextual_text(backbone_model, flat_options, device)
        num_options = len(batch["options"][0])
        option_tokens = option_tokens.view(len(batch["options"]), num_options, option_tokens.shape[1], option_tokens.shape[2])
        option_mask = option_mask.view(len(batch["options"]), num_options, option_mask.shape[1])
        return future_visual, question_tokens, question_mask, option_tokens, option_mask

    def forward(self, backbone_model, batch, device: torch.device):
        future_visual, question_tokens, question_mask, option_tokens, option_mask = self.encode_batch(
            backbone_model,
            batch,
            device,
        )
        future_hidden = self.visual_proj(future_visual)
        if future_hidden.shape[1] > self.future_pos_embed.shape[1]:
            raise RuntimeError(
                f"Configured future_frames={self.future_pos_embed.shape[1]} but got {future_hidden.shape[1]} frames."
            )
        future_hidden = future_hidden + self.future_pos_embed[:, : future_hidden.shape[1]]
        question_hidden = self.question_proj(question_tokens)

        belief = self.belief_tokens.expand(future_hidden.shape[0], -1, -1)
        for text_block, vision_block, self_block in zip(
            self.text_to_belief,
            self.vision_to_belief,
            self.belief_self_attn,
        ):
            belief = text_block(belief, question_hidden, question_mask)
            belief = vision_block(belief, future_hidden)
            belief = self_block(belief)
        belief = self.norm(belief)

        batch_size, num_options, option_seq_len, _ = option_tokens.shape
        option_hidden = self.option_proj(option_tokens.view(batch_size * num_options, option_seq_len, -1))
        option_mask = option_mask.view(batch_size * num_options, option_seq_len)
        repeated_belief = belief.unsqueeze(1).expand(-1, num_options, -1, -1).reshape(
            batch_size * num_options,
            belief.shape[1],
            belief.shape[2],
        )
        option_conditioned = self.option_to_belief(option_hidden, repeated_belief)
        option_summary = self._pool_masked_tokens(option_conditioned, option_mask)
        logits = self.classifier(option_summary).view(batch_size, num_options)

        labels = batch.get("labels")
        outputs = {"logits": logits}
        if labels is not None:
            labels = labels.to(device)
            outputs["loss"] = F.cross_entropy(logits, labels)
        return outputs
