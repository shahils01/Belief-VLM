import argparse
import os
import random
import time

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset

from datasets_registry import get_adapter, list_adapters
from general_data_utils import (
    build_multimodal_prompt_only_example,
    build_multimodal_sft_example,
    decode_media,
    stack_inputs,
)
from train import _apply_peft, _configure_memory_optimizations, _resolve_vl_model_preset, build_model
from vector_memory import OnlineVectorMemory


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
    parser.add_argument("--use_memory_retrieval", action="store_true")
    parser.add_argument("--memory_top_k", type=int, default=2)
    parser.add_argument("--memory_index_backend", type=str, default="auto", choices=["auto", "faiss", "numpy"])
    parser.add_argument("--memory_same_task_first", action="store_true", default=True)
    parser.add_argument("--memory_layer_idx", type=int, default=1)
    parser.add_argument("--memory_inject_offset", type=int, default=0)
    parser.add_argument("--freeze_memory_prefix", action="store_true")

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


def _infer_hidden_dim_from_batch(model, batch, accelerator, layer_idx: int):
    prompt_inputs = {
        k: v.to(accelerator.device) if torch.is_tensor(v) else v
        for k, v in batch["prompt_inputs"].items()
    }
    with torch.no_grad():
        encoded = accelerator.unwrap_model(model).encode_inputs(
            prompt_inputs,
            pooling="last",
            layer_idx=layer_idx,
        )
    return int(encoded["pooled_state"].shape[-1])


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
            prompt_only = build_multimodal_prompt_only_example(
                processor=self.processor,
                media_type=sample["media_type"],
                media=media,
                prompt=sample["prompt"],
                vl_backend=self.args.vl_backend,
                max_text_len=self.args.vl_max_text_len,
            )
            model_inputs = {
                key: value
                for key, value in packed.items()
                if torch.is_tensor(value) and key != "labels"
            }
            prompt_inputs = {
                key: value
                for key, value in prompt_only.items()
                if torch.is_tensor(value)
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
                    "prompt_inputs": prompt_inputs,
                }
            )
        max_len = max(int(item["labels"].shape[0]) for item in items)
        labels = [
            torch.nn.functional.pad(item["labels"], (0, max_len - int(item["labels"].shape[0])), value=-100)
            for item in items
        ]
        return {
            "ids": [item["id"] for item in items],
            "task_names": [item["task_name"] for item in items],
            "prompts": [item["prompt"] for item in items],
            "target_text": [item["target_text"] for item in items],
            "metadata": [item["metadata"] for item in items],
            "choices": [item.get("choices") for item in items],
            "correct_idx": [item.get("correct_idx") for item in items],
            "media_type": [item["media_type"] for item in items],
            "inputs": stack_inputs([item["inputs"] for item in items]),
            "prompt_inputs": stack_inputs([item["prompt_inputs"] for item in items]),
            "labels": torch.stack(labels, dim=0),
        }


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


def _load_checkpoint(model, memory_fusion, optimizer, accelerator, args):
    if not args.resume_checkpoint:
        return 0, 0, None
    ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    accelerator.unwrap_model(model).load_state_dict(state_dict, strict=False)
    if memory_fusion is not None and isinstance(ckpt, dict) and ckpt.get("memory_fusion") is not None:
        accelerator.unwrap_model(memory_fusion).load_state_dict(ckpt["memory_fusion"], strict=False)
    if not args.load_model_only and isinstance(ckpt, dict):
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0)), ckpt.get("vector_memory")
    return 0, 0, ckpt.get("vector_memory") if isinstance(ckpt, dict) else None


def _save_checkpoint(model, memory_fusion, optimizer, accelerator, args, epoch, global_step, memory):
    if not accelerator.is_main_process:
        return
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
    torch.save(
        {
            "model": accelerator.unwrap_model(model).state_dict(),
            "memory_fusion": accelerator.unwrap_model(memory_fusion).state_dict() if memory_fusion is not None else None,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "vector_memory": memory.state_dict() if memory is not None else None,
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
    if args.use_memory_retrieval and args.freeze_memory_prefix:
        frozen = model.freeze_language_prefix(max(int(args.memory_layer_idx) + 1, 0))
        accelerator.print(f"froze {frozen} language layers for memory retrieval consistency")
    processor = accelerator.unwrap_model(model).backbone.processor
    memory_fusion = None
    memory = None

    train_dataset = adapter.build_train_dataset(args)
    shuffle = not isinstance(train_dataset, IterableDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=SFTCollator(processor, args),
    )

    if args.use_memory_retrieval:
        warmup_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=SFTCollator(processor, args),
        )
        warmup_batch = next(iter(warmup_loader))
        hidden_dim = _infer_hidden_dim_from_batch(model, warmup_batch, accelerator, args.memory_layer_idx)
        memory_fusion = GatedMemoryFusion(int(hidden_dim)).to(accelerator.device)
        memory = OnlineVectorMemory(
            dim=int(hidden_dim),
            prior_prefix="Belief prior:",
            backend=args.memory_index_backend,
            same_task_first=args.memory_same_task_first,
        )

    optimizer = torch.optim.AdamW(
        (
            list(param for param in model.parameters() if param.requires_grad)
            + list(memory_fusion.parameters() if memory_fusion is not None else [])
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if memory_fusion is not None:
        model, memory_fusion, optimizer, train_loader = accelerator.prepare(model, memory_fusion, optimizer, train_loader)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    start_epoch, global_step, memory_state = _load_checkpoint(model, memory_fusion, optimizer, accelerator, args)
    if memory is not None and memory_state is not None:
        memory = OnlineVectorMemory.from_state_dict(memory_state, args)

    accelerator.print(f"dataset={args.dataset_name} trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        start_time = time.time()
        for step, batch in enumerate(train_loader, start=1):
            inputs = {k: v.to(accelerator.device) if torch.is_tensor(v) else v for k, v in batch["inputs"].items()}
            prompt_inputs = {k: v.to(accelerator.device) if torch.is_tensor(v) else v for k, v in batch["prompt_inputs"].items()}
            labels = batch["labels"].to(accelerator.device)
            injected = False
            hook_handle = None
            query_state = None
            response_state = None
            if memory is not None:
                with torch.no_grad():
                    query_state = accelerator.unwrap_model(model).encode_inputs(
                        prompt_inputs,
                        pooling="last",
                        layer_idx=args.memory_layer_idx,
                    )["pooled_state"].float()
                    response_state = accelerator.unwrap_model(model).encode_inputs(
                        inputs,
                        pooling="last",
                        layer_idx=args.memory_layer_idx,
                    )["pooled_state"].float()
                if len(memory) > 0:
                    retrieved = memory.retrieve_aggregates(
                        query_state,
                        batch["ids"],
                        batch["task_names"],
                        top_k=args.memory_top_k,
                    )
                    memory_context = torch.from_numpy(retrieved["context"]).to(accelerator.device, dtype=query_state.dtype)
                    memory_answer = torch.from_numpy(retrieved["answer"]).to(accelerator.device, dtype=query_state.dtype)
                    memory_reward = torch.from_numpy(retrieved["reward"]).to(accelerator.device, dtype=query_state.dtype)
                    fused_memory = memory_fusion(query_state.to(accelerator.device), memory_context, memory_answer, memory_reward)
                    inject_layer_idx = max(int(args.memory_layer_idx) + int(args.memory_inject_offset), 0)
                    hook_handle = accelerator.unwrap_model(model).inject_pooled_memory_context(fused_memory, inject_layer_idx)
                    injected = True
            with accelerator.accumulate(model):
                try:
                    outputs = model(inputs, labels=labels)
                    loss = outputs["loss"]
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm > 0:
                        params = list(model.parameters())
                        if memory_fusion is not None:
                            params += list(memory_fusion.parameters())
                        accelerator.clip_grad_norm_(params, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()
            if memory is not None and query_state is not None and response_state is not None:
                memory.add(
                    embeddings=query_state,
                    sample_ids=batch["ids"],
                    task_names=batch["task_names"],
                    answer_texts=batch["target_text"],
                    answer_embeddings=response_state,
                    rewards=torch.ones(query_state.size(0), dtype=query_state.dtype, device=query_state.device),
                )
            running_loss += float(loss.detach().item()) * labels.size(0)
            running_count += int(labels.size(0))
            global_step += 1
            if step % args.log_every == 0:
                avg_loss = running_loss / max(1, running_count)
                elapsed = time.time() - start_time
                extra = ""
                if memory is not None:
                    extra = f" memory_size={len(memory)} injected={int(injected)}"
                accelerator.print(
                    f"epoch={epoch} step={step} global_step={global_step} loss={avg_loss:.4f} "
                    f"samples={running_count} sec={elapsed:.1f}{extra}"
                )
        _save_checkpoint(model, memory_fusion, optimizer, accelerator, args, epoch + 1, global_step, memory)

    if accelerator.is_main_process and args.wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
