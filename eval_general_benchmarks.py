import argparse
import json
import os
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_registry import get_evaluator
from datasets_registry import get_adapter, list_adapters
from general_data_utils import (
    build_multimodal_prompt_only_example,
    build_multimodal_sft_example,
    decode_media,
)
from train import _apply_peft, _configure_memory_optimizations, _resolve_vl_model_preset, build_model
from vector_memory import OnlineVectorMemory


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generalized multimodal benchmarks with InternVL.")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--benchmark_name", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--dataset_repo", type=str, default="")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--dataset_revision", type=str, default="main")
    parser.add_argument("--annotation_path", type=str, default="")
    parser.add_argument("--train_split", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trust_remote_code_dataset", action="store_true", default=False)
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
    parser.add_argument("--video_frames", type=int, default=None)
    parser.add_argument("--vl_model_preset", type=str, default=None)
    parser.add_argument("--vl_model_name", type=str, default=None)
    parser.add_argument("--vl_backend", type=str, default=None)
    parser.add_argument("--vl_dtype", type=str, default=None)
    parser.add_argument("--vl_max_text_len", type=int, default=None)
    parser.add_argument("--peft", type=str, default=None)
    parser.add_argument("--freeze_vl", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "generate", "multiple_choice_nll"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--print_samples", type=int, default=10)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--save_predictions", type=str, default="")
    parser.add_argument("--use_memory_retrieval", action="store_true")
    parser.add_argument("--memory_top_k", type=int, default=2)
    parser.add_argument("--memory_index_backend", type=str, default="auto", choices=["auto", "faiss", "numpy"])
    parser.add_argument("--memory_same_task_first", action="store_true", default=True)
    parser.add_argument("--memory_layer_idx", type=int, default=None)
    parser.add_argument("--memory_inject_offset", type=int, default=None)
    return parser.parse_args()


class GatedMemoryFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.context_proj = nn.Linear(dim, dim)
        self.answer_proj = nn.Linear(dim, dim)
        self.reward_proj = nn.Linear(1, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.Sigmoid(),
        )

    def forward(self, query_state, memory_context, memory_answer, memory_reward):
        reward = memory_reward.unsqueeze(-1)
        gate = self.gate(torch.cat([query_state, memory_context, reward], dim=-1))
        return (
            self.context_proj(memory_context)
            + gate * self.answer_proj(memory_answer)
            + self.reward_proj(reward)
        )


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


def _merge_args(cli_args, ckpt_args):
    merged = dict(ckpt_args)
    for key, value in vars(cli_args).items():
        if value not in (None, ""):
            merged[key] = value
    merged.setdefault("dataset_name", "ai2d")
    merged.setdefault("vl_model_preset", "internvl3_5_2b")
    merged.setdefault("vl_model_name", "OpenGVLab/InternVL3_5-2B-HF")
    merged.setdefault("vl_backend", "internvl")
    merged.setdefault("vl_dtype", "bfloat16")
    merged.setdefault("vl_max_text_len", 256)
    merged.setdefault("video_frames", 8)
    merged.setdefault("peft", "none")
    merged.setdefault("freeze_vl", False)
    merged.setdefault("gradient_checkpointing", False)
    merged.setdefault("disable_vl_cache", False)
    merged.setdefault("allow_tf32", False)
    merged.setdefault("use_memory_retrieval", False)
    merged.setdefault("memory_top_k", 2)
    merged.setdefault("memory_index_backend", "auto")
    merged.setdefault("memory_same_task_first", True)
    merged.setdefault("memory_layer_idx", 1)
    merged.setdefault("memory_inject_offset", 0)
    return SimpleNamespace(**merged)


def _load_model(checkpoint_path, args, device):
    args.quantization_config = _build_quant_config(args)
    _resolve_vl_model_preset(args)
    model = build_model(args, device=device)
    model = _apply_peft(model, args)
    _configure_memory_optimizations(model, args)
    ckpt = {}
    memory_fusion = None
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        if args.use_memory_retrieval and isinstance(ckpt, dict) and ckpt.get("memory_fusion") is not None:
            hidden_dim = getattr(model.backbone.model.config, "hidden_size", None)
            if hidden_dim is None:
                raise RuntimeError("Could not determine hidden size for memory fusion.")
            memory_fusion = GatedMemoryFusion(int(hidden_dim)).to(device)
            memory_fusion.load_state_dict(ckpt["memory_fusion"], strict=False)
    model.eval()
    if memory_fusion is not None:
        memory_fusion.eval()
    return model, memory_fusion, ckpt


def _sequence_nll(logits, labels):
    seq_len = min(int(logits.shape[1]), int(labels.shape[1]))
    logits = logits[:, :seq_len, :]
    labels = labels[:, :seq_len]
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size(0), shift_labels.size(1))
    mask = shift_labels.ne(-100)
    denom = mask.sum(dim=1).clamp_min(1)
    return (token_losses * mask).sum(dim=1) / denom


def _decode_generated_text(model, generated_ids, prompt_input_ids):
    tokenizer = model.backbone.tokenizer
    prompt_len = int(prompt_input_ids.shape[-1])
    gen_tokens = generated_ids[0, prompt_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else {}
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    merged_args = _merge_args(args, ckpt_args)
    if merged_args.dataset_name not in list_adapters():
        raise RuntimeError(
            f"Unknown dataset adapter `{merged_args.dataset_name}`. Available: {', '.join(list_adapters())}"
        )
    model, memory_fusion, ckpt = _load_model(args.checkpoint, merged_args, device)
    adapter = get_adapter(merged_args.dataset_name)
    evaluator = get_evaluator(merged_args.benchmark_name or adapter.default_evaluator)
    dataset = adapter.build_eval_dataset(merged_args, split=merged_args.eval_split)
    memory = None
    if merged_args.use_memory_retrieval and isinstance(ckpt, dict) and ckpt.get("vector_memory") is not None:
        memory = OnlineVectorMemory.from_state_dict(ckpt["vector_memory"], merged_args)

    total = 0
    correct = 0
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []

    iterator = dataset if isinstance(dataset, torch.utils.data.IterableDataset) else dataset
    for sample in iterator:
        if merged_args.max_samples > 0 and total >= merged_args.max_samples:
            break
        media = decode_media(
            sample["media_type"],
            sample["media"],
            video_frames=merged_args.video_frames,
            metadata=sample.get("metadata") or {},
        )
        hook_handle = None
        if memory is not None and len(memory) > 0 and memory_fusion is not None:
            prompt_only = build_multimodal_prompt_only_example(
                processor=model.backbone.processor,
                media_type=sample["media_type"],
                media=media,
                prompt=evaluator.prepare_prompt(sample),
                vl_backend=merged_args.vl_backend,
                max_text_len=merged_args.vl_max_text_len,
            )
            prompt_inputs = {
                k: v.unsqueeze(0).to(device) if torch.is_tensor(v) and v.dim() > 0 else v
                for k, v in prompt_only.items()
                if torch.is_tensor(v)
            }
            with torch.no_grad():
                query_state = model.encode_inputs(
                    prompt_inputs,
                    pooling="last",
                    layer_idx=merged_args.memory_layer_idx,
                )["pooled_state"].float()
            retrieved = memory.retrieve_aggregates(
                query_state,
                [sample["id"]],
                [sample["task_name"]],
                top_k=merged_args.memory_top_k,
            )
            memory_context = torch.from_numpy(retrieved["context"]).to(device, dtype=query_state.dtype)
            memory_answer = torch.from_numpy(retrieved["answer"]).to(device, dtype=query_state.dtype)
            memory_reward = torch.from_numpy(retrieved["reward"]).to(device, dtype=query_state.dtype)
            fused_memory = memory_fusion(query_state, memory_context, memory_answer, memory_reward)
            hook_handle = model.inject_pooled_memory_context(
                fused_memory,
                max(int(merged_args.memory_layer_idx) + int(merged_args.memory_inject_offset), 0),
            )

        try:
            if merged_args.eval_mode == "multiple_choice_nll" or (
                merged_args.eval_mode == "auto" and evaluator.task_type == "multiple_choice"
            ):
                best_idx = None
                best_nll = None
                for idx, choice in enumerate(sample.get("choices") or []):
                    packed = build_multimodal_sft_example(
                        processor=model.backbone.processor,
                        media_type=sample["media_type"],
                        media=media,
                        prompt=evaluator.prepare_prompt(sample),
                        answer=choice,
                        vl_backend=merged_args.vl_backend,
                        max_text_len=merged_args.vl_max_text_len,
                    )
                    inputs = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) and v.dim() > 0 else v for k, v in packed.items() if k in {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos"}}
                    labels = packed["labels"].unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(inputs, labels=labels)
                    nll = float(_sequence_nll(outputs["logits"], labels)[0].item())
                    if best_nll is None or nll < best_nll:
                        best_nll = nll
                        best_idx = idx
                pred_text = sample["choices"][best_idx] if best_idx is not None else ""
                metrics = evaluator.score(pred_text, sample)
                metrics["pred_idx"] = best_idx
            else:
                packed = build_multimodal_prompt_only_example(
                    processor=model.backbone.processor,
                    media_type=sample["media_type"],
                    media=media,
                    prompt=evaluator.prepare_prompt(sample),
                    vl_backend=merged_args.vl_backend,
                    max_text_len=merged_args.vl_max_text_len,
                )
                inputs = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) and v.dim() > 0 else v for k, v in packed.items() if k in {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos"}}
                with torch.no_grad():
                    generated = model.generate(inputs, max_new_tokens=merged_args.max_new_tokens)
                pred_text = _decode_generated_text(model, generated, inputs["input_ids"])
                metrics = evaluator.score(pred_text, sample)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        total += 1
        correct += int(metrics["correct"])
        by_task[sample["task_name"]]["correct"] += int(metrics["correct"])
        by_task[sample["task_name"]]["total"] += 1
        predictions.append(
            {
                "id": sample["id"],
                "task_name": sample["task_name"],
                "prediction": metrics.get("prediction", pred_text),
                "target": metrics.get("target", sample.get("target_text", "")),
                "correct": bool(metrics["correct"]),
                "correct_idx": sample.get("correct_idx"),
                "pred_idx": metrics.get("pred_idx"),
            }
        )
        if total <= merged_args.print_samples:
            print(
                f"[{total}] task={sample['task_name']} id={sample['id']} "
                f"pred={predictions[-1]['prediction']} target={predictions[-1]['target']} correct={predictions[-1]['correct']}"
            )
        if merged_args.progress_every > 0 and total % merged_args.progress_every == 0:
            print(f"progress {total} accuracy={correct / max(1, total):.4f}")

    print(f"final accuracy={correct / max(1, total):.4f} ({correct}/{total})")
    print("per_task_accuracy:")
    for task_name in sorted(by_task):
        stats = by_task[task_name]
        print(f"  {task_name}: {stats['correct'] / max(1, stats['total']):.4f} ({stats['correct']}/{stats['total']})")
    if merged_args.save_predictions:
        os.makedirs(os.path.dirname(merged_args.save_predictions) or ".", exist_ok=True)
        with open(merged_args.save_predictions, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2)


if __name__ == "__main__":
    evaluate(parse_args())
