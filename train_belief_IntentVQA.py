import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from NextVQA_data_loading import build_nextvqa_intent_loader
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
    parser = argparse.ArgumentParser(description="Train IntentQA-style context-aware VQA model in Belief-VLM.")
    parser.add_argument("--dataset_source", type=str, default="hd_epic", choices=["hd_epic", "nextvqa"])
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
    parser.add_argument("--action_column", type=str, default="action")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--video_frames", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_belief_intent_vqa")
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

    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--bert_max_len", type=int, default=128)
    parser.add_argument("--freeze_bert", action="store_true", default=True)
    parser.add_argument("--unfreeze_bert", dest="freeze_bert", action="store_false")

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--dgt_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--topk_nodes", type=int, default=8)

    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--triplet_weight", type=float, default=0.2)
    parser.add_argument("--wups_t1", type=float, default=0.9)
    parser.add_argument("--wups_t2", type=float, default=0.1)
    parser.add_argument("--commonsense_weight", type=float, default=0.5)
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


@dataclass
class IntentSample:
    sample_id: str
    action: str
    question: str
    choices: List[str]
    label: int
    answer_text: str
    frames: List


class IntentQADataset(Dataset):
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
        action = _normalize_text(
            _get_first(
                record,
                [self.args.action_column, "action", "action_id", "verb", "lemmatized_verb"],
            )
        )
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
        answer_text = choices[label]

        video_path = _resolve_hd_epic_video_path(self.args, record)
        start_time_sec, end_time_sec = _resolve_hd_epic_clip_window(record)
        frames = decode_mp4_frames(
            video_path,
            self.args.video_frames,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
        )
        return IntentSample(
            sample_id=sample_id,
            action=action,
            question=question,
            choices=choices,
            label=label,
            answer_text=answer_text,
            frames=frames,
        )


def collate_intent_batch(batch: List[IntentSample]):
    return {
        "ids": [x.sample_id for x in batch],
        "actions": [x.action for x in batch],
        "questions": [x.question for x in batch],
        "choices": [x.choices for x in batch],
        "labels": torch.tensor([x.label for x in batch], dtype=torch.long),
        "answer_texts": [x.answer_text for x in batch],
        "frames": [x.frames for x in batch],
    }


def build_loader(args, split_name: str, batch_size, num_workers, is_train):
    if args.dataset_source == "nextvqa":
        return build_nextvqa_intent_loader(
            args=args,
            split_name=split_name,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=is_train,
        )

    records = _load_records(args)
    dataset = IntentQADataset(records=records, args=args, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        collate_fn=collate_intent_batch,
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


def build_bert_encoder(args, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    encoder = AutoModel.from_pretrained(args.bert_model_name).to(device)
    if args.freeze_bert:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
    return tokenizer, encoder


def flatten_qa_texts(questions: List[str], choices: List[List[str]]) -> Tuple[List[str], List[int]]:
    texts = []
    counts = []
    for q, opts in zip(questions, choices):
        counts.append(len(opts))
        for a in opts:
            texts.append(f"{q} [SEP] {a}")
    return texts, counts


def encode_qa_features(tokenizer, encoder, texts, device, max_len: int, freeze: bool):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if freeze:
        with torch.no_grad():
            outputs = encoder(**inputs)
    else:
        outputs = encoder(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def pack_qa_features(flat_feats, counts, device):
    max_choices = max(counts)
    dim = flat_feats.shape[-1]
    packed = torch.zeros(len(counts), max_choices, dim, device=device, dtype=flat_feats.dtype)
    mask = torch.zeros(len(counts), max_choices, device=device, dtype=torch.bool)
    cur = 0
    for i, c in enumerate(counts):
        packed[i, :c] = flat_feats[cur : cur + c]
        mask[i, :c] = True
        cur += c
    return packed, mask


def _simple_wups(a: str, b: str) -> float:
    a_tokens = set(_normalize_text(a).lower().split())
    b_tokens = set(_normalize_text(b).lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    return inter / float(max(len(a_tokens), len(b_tokens)))


def mine_triplets(
    actions: List[str],
    answers: List[str],
    t1: float,
    t2: float,
) -> List[Tuple[int, int, int]]:
    triplets = []
    n = len(answers)
    for i in range(n):
        pos_idx = None
        neg_idx = None
        for j in range(n):
            if i == j:
                continue
            same_action = (actions[i] == actions[j]) if actions[i] and actions[j] else True
            if not same_action:
                continue
            w = _simple_wups(answers[i], answers[j])
            if pos_idx is None and w >= t1:
                pos_idx = j
            if neg_idx is None and w < t2:
                neg_idx = j
            if pos_idx is not None and neg_idx is not None:
                break
        if pos_idx is not None and neg_idx is not None:
            triplets.append((i, pos_idx, neg_idx))
    return triplets


class IntentQAModel(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int, dgt_layers: int, num_heads: int, dropout: float, topk_nodes: int):
        super().__init__()
        self.topk_nodes = topk_nodes
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dgt = nn.TransformerEncoder(enc_layer, num_layers=dgt_layers)
        self.mhsa = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.cs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, region_nodes, qa_feats, choice_mask, labels=None):
        # region_nodes: [B, R, Dv], qa_feats: [B, C, Dt]
        g_r = self.dgt(self.v_proj(region_nodes))
        q_a = self.t_proj(qa_feats)

        # Eq. (6): S_{r|q,A} = G_r * F_{q,A}^T
        sim = torch.einsum("brd,bcd->brc", g_r, q_a)  # [B, R, C]

        logits_vis = []
        gt_topk_feats = None
        for c in range(q_a.shape[1]):
            # Eq. (7): G_{r|q,A} = G_r + S_{r|q,A}F_{q,A}
            s_c = sim[:, :, c].unsqueeze(-1)
            g_rqa = g_r + s_c * q_a[:, c].unsqueeze(1)
            fused, _ = self.mhsa(g_rqa, g_rqa, g_rqa)
            pooled = fused.mean(dim=1)
            score = torch.sum(F.normalize(pooled, dim=-1) * F.normalize(q_a[:, c], dim=-1), dim=-1)
            logits_vis.append(score)
        logits_vis = torch.stack(logits_vis, dim=-1)
        logits_vis = logits_vis.masked_fill(~choice_mask, -1e4)

        # commonsense prior branch (text-only)
        logits_cs = self.cs_head(q_a).squeeze(-1).masked_fill(~choice_mask, -1e4)

        if labels is not None:
            # Top-k node features from anchor gt option (Eq. 9)
            batch_idx = torch.arange(g_r.shape[0], device=g_r.device)
            sim_gt = sim[batch_idx, :, labels]  # [B, R]
            k = min(self.topk_nodes, sim_gt.shape[1])
            topk_idx = torch.topk(sim_gt, k=k, dim=-1).indices  # [B, k]
            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, g_r.shape[-1])
            gt_topk_feats = torch.gather(g_r, dim=1, index=gather_idx)  # [B, k, D]
        return {
            "logits_vis": logits_vis,
            "logits_cs": logits_cs,
            "gt_topk_feats": gt_topk_feats,
        }


def triplet_loss_from_feats(feats, triplets, margin):
    if feats is None or len(triplets) == 0:
        return feats.new_zeros(()) if feats is not None else torch.tensor(0.0)
    losses = []
    for a_idx, p_idx, n_idx in triplets:
        a = feats[a_idx]
        p = feats[p_idx]
        n = feats[n_idx]

        p_align = (a @ p.transpose(0, 1)) @ p
        n_align = (a @ n.transpose(0, 1)) @ n
        d_ap = ((a - p_align) ** 2).mean()
        d_an = ((a - n_align) ** 2).mean()
        losses.append(F.relu(d_ap - d_an + margin))
    return torch.stack(losses).mean()


def extract_region_nodes(visual_backbone, batch_frames, device):
    # build pixel values frame-by-frame then extract ViT token embeddings
    clips = batch_frames
    clip_lengths = [len(c) for c in clips]
    flat_frames = [f for c in clips for f in c]
    with torch.no_grad():
        pixel_values = visual_backbone.backbone.build_pixel_values(flat_frames)
        image_tokens = visual_backbone.backbone.extract_image_tokens(pixel_values, normalize=False)  # [Nf, Nt, D]
    per_clip_nodes = []
    cursor = 0
    for length in clip_lengths:
        tokens = image_tokens[cursor : cursor + length]  # [T, Nt, D]
        cursor += length
        per_clip_nodes.append(tokens.reshape(tokens.shape[0] * tokens.shape[1], tokens.shape[2]))
    return torch.stack(per_clip_nodes, dim=0).to(device)


def run_epoch(
    visual_backbone,
    qa_tokenizer,
    qa_encoder,
    model,
    loader,
    optimizer,
    accelerator,
    args,
    train: bool,
):
    model.train() if train else model.eval()
    if args.freeze_bert:
        qa_encoder.eval()
    else:
        qa_encoder.train() if train else qa_encoder.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_triplet = 0.0
    total_acc = 0.0
    total = 0

    for step, batch in enumerate(loader, start=1):
        labels = batch["labels"].to(accelerator.device)
        actions = batch["actions"]
        answers = batch["answer_texts"]

        region_nodes = extract_region_nodes(visual_backbone, batch["frames"], accelerator.device)
        flat_qa, counts = flatten_qa_texts(batch["questions"], batch["choices"])
        qa_feats_flat = encode_qa_features(
            qa_tokenizer,
            qa_encoder,
            flat_qa,
            accelerator.device,
            max_len=args.bert_max_len,
            freeze=args.freeze_bert,
        )
        qa_feats, choice_mask = pack_qa_features(qa_feats_flat, counts, accelerator.device)

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                out = model(region_nodes=region_nodes, qa_feats=qa_feats, choice_mask=choice_mask, labels=labels)
                logits = out["logits_vis"] + args.commonsense_weight * out["logits_cs"]
                ce = F.cross_entropy(logits, labels)

                triplets = mine_triplets(actions, answers, t1=args.wups_t1, t2=args.wups_t2)
                tri = triplet_loss_from_feats(out["gt_topk_feats"], triplets, margin=args.triplet_margin).to(logits.device)
                loss = ce + args.triplet_weight * tri

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
        total_ce += float(ce.detach().item()) * batch_size
        total_triplet += float(tri.detach().item()) * batch_size
        total_acc += float(preds.eq(labels).float().sum().item())

        if args.log_every > 0 and step % args.log_every == 0:
            phase = "train" if train else "val"
            denom = max(total, 1)
            print(
                f"{phase} step={step} "
                f"loss={total_loss / denom:.4f} "
                f"ce={total_ce / denom:.4f} "
                f"triplet={total_triplet / denom:.4f} "
                f"acc={total_acc / denom:.4f}"
            )

    denom = max(total, 1)
    return {
        "loss": total_loss / denom,
        "ce": total_ce / denom,
        "triplet": total_triplet / denom,
        "acc": total_acc / denom,
    }


def main():
    args = parse_args()
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    visual_backbone = build_visual_backbone(args, accelerator.device)
    qa_tokenizer, qa_encoder = build_bert_encoder(args, accelerator.device)

    probe_loader = build_loader(args, split_name="train", batch_size=1, num_workers=0, is_train=True)
    probe = next(iter(probe_loader))
    probe_regions = extract_region_nodes(visual_backbone, probe["frames"], accelerator.device)
    probe_qa = encode_qa_features(
        qa_tokenizer,
        qa_encoder,
        [f"{probe['questions'][0]} [SEP] {probe['choices'][0][0]}"],
        accelerator.device,
        max_len=args.bert_max_len,
        freeze=args.freeze_bert,
    )
    visual_dim = int(probe_regions.shape[-1])
    text_dim = int(probe_qa.shape[-1])

    model = IntentQAModel(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        dgt_layers=args.dgt_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        topk_nodes=args.topk_nodes,
    ).to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = build_loader(args, split_name="train", batch_size=args.batch_size, num_workers=args.num_workers, is_train=True)
    val_loader = build_loader(args, split_name="validation", batch_size=args.batch_size, num_workers=args.num_workers, is_train=False)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    start_epoch = 0
    best_val_acc = -1.0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _load_checkpoint_state(model, ckpt["model"], accelerator)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            visual_backbone,
            qa_tokenizer,
            qa_encoder,
            model,
            train_loader,
            optimizer,
            accelerator,
            args,
            train=True,
        )
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_ce={train_metrics['ce']:.4f} train_triplet={train_metrics['triplet']:.4f} "
            f"train_acc={train_metrics['acc']:.4f}"
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                visual_backbone,
                qa_tokenizer,
                qa_encoder,
                model,
                val_loader,
                optimizer,
                accelerator,
                args,
                train=False,
            )
        print(
            f"epoch={epoch} val_loss={val_metrics['loss']:.4f} "
            f"val_ce={val_metrics['ce']:.4f} val_triplet={val_metrics['triplet']:.4f} "
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
