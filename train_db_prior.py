import argparse
import functools
import inspect
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

try:
    from accelerate import DataLoaderConfiguration
except Exception:
    DataLoaderConfiguration = None

try:
    from accelerate.utils import DistributedDataParallelKwargs
except Exception:
    DistributedDataParallelKwargs = None

try:
    from accelerate.utils import FullyShardedDataParallelPlugin
except Exception:
    try:
        from accelerate.utils import FSDPPlugin as FullyShardedDataParallelPlugin
    except Exception:
        FullyShardedDataParallelPlugin = None

try:
    from torch.distributed.fsdp import CPUOffload
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
except Exception:
    CPUOffload = None
    size_based_auto_wrap_policy = None

from belief_db import BeliefVectorDB
from data_loading import _load_records, build_rl_vqa_loader
from train import (
    _apply_peft,
    _configure_memory_optimizations,
    _count_parameters,
    _load_checkpoint_state,
    _resolve_vl_model_preset,
    build_model,
)
from train_ppo_vqa import PPOAnswerPolicy, _build_quant_config, _masked_logits


class PriorSelectorPolicy(nn.Module):
    def __init__(self, hidden_dim: int, candidate_dim: int, dropout: float):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_proj = nn.Linear(candidate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, query_state, candidate_embeddings, candidate_counts):
        query_hidden = self.query_proj(query_state).unsqueeze(1)
        candidate_hidden = self.candidate_proj(candidate_embeddings)
        hidden = torch.tanh(query_hidden + candidate_hidden)
        hidden = self.dropout(hidden)
        logits = self.scorer(hidden).squeeze(-1)
        candidate_ids = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
        mask = candidate_ids < candidate_counts.unsqueeze(1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        values = self.value_head(query_state).squeeze(-1)
        return logits, values


def parse_args():
    parser = argparse.ArgumentParser(description="Train retrieval-augmented prior selection and optional RL answer head.")
    parser.add_argument("--dataset_type", type=str, default="hd_epic_local", choices=["hd_epic_local"])
    parser.add_argument("--dataset_name", type=str, default="hd_epic_local")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--video_root", type=str, default="")
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--annotation_path", type=str, default="")
    parser.add_argument("--metadata_root", type=str, default="")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--video_id_column", type=str, default="video_id")
    parser.add_argument("--video_path_column", type=str, default="video_path")
    parser.add_argument("--participant_column", type=str, default="participant_id")
    parser.add_argument("--question_column", type=str, default="question")
    parser.add_argument("--answer_column", type=str, default="answer")
    parser.add_argument("--options_column", type=str, default="options")
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--max_val_samples_per_split", type=int, default=0)
    parser.add_argument("--train_sampling_mode", type=str, default="task_uniform", choices=["flat", "task_uniform"])
    parser.add_argument("--train_samples_per_epoch", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--video_frames", type=int, default=8)

    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--fsdp_min_num_params", type=int, default=1_000_000)
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--fsdp_use_orig_params", action="store_true")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
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
    parser.add_argument("--freeze_vl", action="store_true")
    parser.add_argument("--vlm_checkpoint", type=str, default="")
    parser.add_argument("--train_vlm_with_rl", action="store_true")
    parser.add_argument("--state_pooling", type=str, default="last", choices=["last", "mean"])

    parser.add_argument("--peft", type=str, default="none", choices=["none", "lora", "qlora"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--policy_lr", type=float, default=1e-4)
    parser.add_argument("--selector_lr", type=float, default=1e-4)
    parser.add_argument("--vlm_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--max_choice_options", type=int, default=5)
    parser.add_argument("--policy_dropout", type=float, default=0.2)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_db_prior")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--answer_head_checkpoint", type=str, default="")
    parser.add_argument("--load_model_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_rl_answer_head", action="store_true")
    parser.add_argument("--use_rl_prior_selector", action="store_true")
    parser.add_argument("--prior_top_k", type=int, default=4)
    parser.add_argument("--prior_prompt_prefix", type=str, default="Belief prior:")
    parser.add_argument("--retrieval_embedder_model", type=str, default="")
    parser.add_argument("--retrieval_hash_dim", type=int, default=512)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vlm-db-prior")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="")
    return parser.parse_args()


def _text_with_media_prompt(processor, prompt: str, vl_backend: str):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor does not expose a tokenizer.")

    def _add_media_token(text: str) -> str:
        vocab = tokenizer.get_vocab()
        if any(token in text for token in ("<video>", "<image>", "<img>")):
            return text
        if vl_backend == "internvl":
            for token in ("<video>", "<image>", "<img>"):
                if token in vocab:
                    return f"{token}\n{text}"
        if "<video>" in vocab:
            return f"<video>\n{text}"
        if "<image>" in vocab:
            return f"<image>\n{text}"
        return text

    prompt_text = f"User: {prompt}\nAssistant:"
    # Use explicit media tokens here. Re-tokenizing chat-template output without
    # re-running the full processor can drop InternVL's expected video token
    # alignment, which leads to image feature/token count mismatches.
    return _add_media_token(prompt_text)


def _text_with_media_full(processor, prompt: str, answer: str, vl_backend: str):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor does not expose a tokenizer.")

    def _add_media_token(text: str) -> str:
        vocab = tokenizer.get_vocab()
        if any(token in text for token in ("<video>", "<image>", "<img>")):
            return text
        if vl_backend == "internvl":
            for token in ("<video>", "<image>", "<img>"):
                if token in vocab:
                    return f"{token}\n{text}"
        if "<video>" in vocab:
            return f"<video>\n{text}"
        if "<image>" in vocab:
            return f"<image>\n{text}"
        return text

    prompt_text = f"User: {prompt}\nAssistant:"
    full_text = f"{prompt_text} {answer}".strip()
    return _add_media_token(prompt_text), _add_media_token(full_text)


def _tokenize_prompt_batch(processor, prompts, vl_backend):
    tokenizer = getattr(processor, "tokenizer", None)
    texts = [_text_with_media_prompt(processor, prompt, vl_backend) for prompt in prompts]
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def _combine_inputs_with_prompts(batch_inputs, prompts, processor, args):
    text_inputs = _tokenize_prompt_batch(processor, prompts, args.vl_backend)
    combined = {}
    for key, value in batch_inputs.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        combined[key] = value
    combined.update(text_inputs)
    return combined


def _slice_single_inputs(batch_inputs, idx: int, batch_size: int, video_frames: int):
    single = {}
    frame_like = {"pixel_values", "pixel_values_videos", "video_values", "video", "videos"}
    for key, value in batch_inputs.items():
        if not torch.is_tensor(value):
            single[key] = value
            continue
        if key in {"input_ids", "attention_mask"} and value.dim() >= 2:
            single[key] = value[idx : idx + 1]
        elif key in frame_like and value.dim() == 4:
            start = idx * video_frames
            end = start + video_frames
            single[key] = value[start:end]
        elif value.dim() > 0 and value.shape[0] == batch_size:
            single[key] = value[idx : idx + 1]
        else:
            single[key] = value
    return single


def _score_answers_with_vlm(model, processor, batch_inputs, prior_prompts, choices_batch, args):
    preds = []
    batch_size = len(prior_prompts)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor does not expose a tokenizer.")

    for idx in range(batch_size):
        single_inputs = _slice_single_inputs(batch_inputs, idx, batch_size, args.video_frames)
        choice_losses = []
        for answer in choices_batch[idx]:
            prompt_text, full_text = _text_with_media_full(processor, prior_prompts[idx], answer, args.vl_backend)
            tokenized_full = tokenizer(
                full_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            prompt_ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"][0]
            input_ids = tokenized_full["input_ids"]
            attention_mask = tokenized_full["attention_mask"]
            labels = input_ids.clone()
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            labels[labels == pad_token_id] = -100
            prompt_len = min(int(prompt_ids.numel()), int(labels.shape[1]))
            labels[:, :prompt_len] = -100

            model_inputs = dict(single_inputs)
            model_inputs["input_ids"] = input_ids
            model_inputs["attention_mask"] = attention_mask
            with torch.no_grad():
                outputs = model(model_inputs, labels=labels)
            logits = outputs["logits"][:, : labels.shape[1], :]
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
            loss = (token_losses * mask).sum(dim=1) / denom
            choice_losses.append(float(loss.item()))
        preds.append(int(torch.tensor(choice_losses).argmin().item()))
    return torch.tensor(preds, dtype=torch.long, device=next(model.parameters()).device)


def _extract_state(model, batch_inputs, args):
    ctx = nullcontext() if args.train_vlm_with_rl else torch.no_grad()
    with ctx:
        encoded = model(batch_inputs, return_hidden_states=True, pooling=args.state_pooling)
    pooled = encoded["pooled_state"]
    if not args.train_vlm_with_rl:
        pooled = pooled.detach()
    return pooled.float()


def _prior_selector_forward(policy, query_state, candidate_embeddings, candidate_counts):
    logits, values = policy(query_state, candidate_embeddings, candidate_counts)
    dist = torch.distributions.Categorical(logits=logits)
    return dist, values


def _answer_forward(policy, state, num_choices):
    logits, values = policy(state)
    masked_logits = _masked_logits(logits, num_choices.to(logits.device))
    return torch.distributions.Categorical(logits=masked_logits), values


def _compose_prior_prompt(base_prompt: str, prior_text: str, prefix: str):
    prior_text = prior_text.strip()
    if not prior_text:
        return base_prompt
    return f"{prefix}\n{prior_text}\n\n{base_prompt}"


def run_epoch(model, processor, belief_db, selector_policy, answer_policy, loader, optimizer, accelerator, args, train, global_step):
    model.train(args.train_vlm_with_rl and train)
    if not args.train_vlm_with_rl:
        model.eval()
    if selector_policy is not None:
        selector_policy.train() if train else selector_policy.eval()
    if answer_policy is not None:
        answer_policy.train() if train else answer_policy.eval()

    total_reward = 0.0
    total_examples = 0
    total_selector_loss = 0.0
    total_answer_loss = 0.0
    step = 0
    epoch_start_time = time.perf_counter()

    for batch in loader:
        step += 1
        num_choices = batch["num_choices"].to(accelerator.device)
        correct_idx = batch["correct_idx"].to(accelerator.device)
        base_inputs = batch["inputs"]

        query_state = _extract_state(model, base_inputs, args)
        retrieval = belief_db.retrieve(
            task_names=batch["task_names"],
            prompts=batch["prompts"],
            top_k=args.prior_top_k,
            exclude_ids=batch["ids"],
        )
        candidate_embeddings = retrieval.candidate_embeddings.to(accelerator.device)
        candidate_counts = retrieval.candidate_counts.to(accelerator.device)

        if selector_policy is not None:
            with torch.no_grad():
                selector_dist, selector_values = _prior_selector_forward(
                    selector_policy, query_state, candidate_embeddings, candidate_counts
                )
                selector_actions = selector_dist.sample() if train else torch.argmax(selector_dist.logits, dim=-1)
                old_selector_log_probs = selector_dist.log_prob(selector_actions)
        else:
            selector_values = torch.zeros(query_state.shape[0], device=accelerator.device)
            selector_actions = torch.zeros(query_state.shape[0], dtype=torch.long, device=accelerator.device)
            old_selector_log_probs = torch.zeros(query_state.shape[0], device=accelerator.device)

        chosen_priors = []
        for row, action in enumerate(selector_actions.tolist()):
            valid_idx = min(int(action), len(retrieval.candidate_texts[row]) - 1)
            chosen_priors.append(retrieval.candidate_texts[row][valid_idx])
        prior_prompts = [
            _compose_prior_prompt(prompt, prior, args.prior_prompt_prefix)
            for prompt, prior in zip(batch["prompts"], chosen_priors)
        ]

        prior_inputs = _combine_inputs_with_prompts(base_inputs, prior_prompts, processor, args)
        prior_state = _extract_state(model, prior_inputs, args)

        if answer_policy is not None:
            with torch.no_grad():
                answer_dist, answer_values = _answer_forward(answer_policy, prior_state, num_choices)
                answer_actions = answer_dist.sample() if train else torch.argmax(answer_dist.logits, dim=-1)
                old_answer_log_probs = answer_dist.log_prob(answer_actions)
                rewards = (answer_actions == correct_idx).float() * float(args.reward_scale)
        else:
            answer_values = torch.zeros(query_state.shape[0], device=accelerator.device)
            old_answer_log_probs = torch.zeros(query_state.shape[0], device=accelerator.device)
            answer_actions = _score_answers_with_vlm(model, processor, prior_inputs, prior_prompts, batch["choices"], args)
            rewards = (answer_actions == correct_idx).float() * float(args.reward_scale)

        selector_returns = rewards
        selector_adv = selector_returns - selector_values.detach()
        answer_returns = rewards
        answer_adv = answer_returns - answer_values.detach()

        selector_loss_value = 0.0
        answer_loss_value = 0.0
        if train:
            train_modules = [module for module in (selector_policy, answer_policy) if module is not None]
            sync_module = train_modules[0] if train_modules else model
            with accelerator.accumulate(sync_module):
                for _ in range(args.ppo_epochs):
                    query_state_step = _extract_state(model, base_inputs, args)
                    total_loss = torch.zeros((), device=accelerator.device)

                    if selector_policy is not None:
                        selector_dist, selector_values_step = _prior_selector_forward(
                            selector_policy, query_state_step, candidate_embeddings, candidate_counts
                        )
                        selector_log_probs = selector_dist.log_prob(selector_actions)
                        selector_ratio = torch.exp(selector_log_probs - old_selector_log_probs)
                        unclipped = selector_ratio * selector_adv
                        clipped = torch.clamp(selector_ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * selector_adv
                        selector_policy_loss = -torch.min(unclipped, clipped).mean()
                        selector_value_loss = F.mse_loss(selector_values_step, selector_returns)
                        selector_entropy = selector_dist.entropy().mean()
                        total_loss = total_loss + selector_policy_loss + args.value_coef * selector_value_loss - args.entropy_coef * selector_entropy
                        selector_loss_value = float(selector_policy_loss.detach().item())

                    if answer_policy is not None:
                        prior_state_step = _extract_state(model, prior_inputs, args)
                        answer_dist, answer_values_step = _answer_forward(answer_policy, prior_state_step, num_choices)
                        answer_log_probs = answer_dist.log_prob(answer_actions)
                        answer_ratio = torch.exp(answer_log_probs - old_answer_log_probs)
                        unclipped = answer_ratio * answer_adv
                        clipped = torch.clamp(answer_ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * answer_adv
                        answer_policy_loss = -torch.min(unclipped, clipped).mean()
                        answer_value_loss = F.mse_loss(answer_values_step, answer_returns)
                        answer_entropy = answer_dist.entropy().mean()
                        total_loss = total_loss + answer_policy_loss + args.value_coef * answer_value_loss - args.entropy_coef * answer_entropy
                        answer_loss_value = float(answer_policy_loss.detach().item())

                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(total_loss)
                    if args.max_grad_norm > 0:
                        for module in train_modules:
                            accelerator.clip_grad_norm_(module.parameters(), args.max_grad_norm)
                        if args.train_vlm_with_rl:
                            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

        batch_size = int(correct_idx.numel())
        total_examples += batch_size
        total_reward += float(rewards.sum().item())
        total_selector_loss += selector_loss_value * batch_size
        total_answer_loss += answer_loss_value * batch_size
        if train:
            global_step += 1

        if args.log_every > 0 and step % args.log_every == 0:
            elapsed = max(time.perf_counter() - epoch_start_time, 1e-6)
            phase = "train" if train else "val"
            accelerator.print(
                f"{phase} step={step} reward={total_reward / max(total_examples,1):.4f} "
                f"selector_loss={total_selector_loss / max(total_examples,1):.4f} "
                f"answer_loss={total_answer_loss / max(total_examples,1):.4f} "
                f"samples_per_sec={total_examples / elapsed:.2f} sec_per_step={elapsed / max(step,1):.2f}"
            )

    metrics = {
        "reward": total_reward / max(total_examples, 1),
        "selector_loss": total_selector_loss / max(total_examples, 1),
        "answer_loss": total_answer_loss / max(total_examples, 1),
    }
    return metrics, global_step


def _save_checkpoint(model, selector_policy, answer_policy, optimizer, accelerator, args, epoch, global_step):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        }
        if selector_policy is not None:
            payload["selector_policy"] = accelerator.unwrap_model(selector_policy).state_dict()
        if answer_policy is not None:
            payload["answer_policy"] = accelerator.unwrap_model(answer_policy).state_dict()
        ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
        torch.save(payload, ckpt_path)
        accelerator.print(f"saved {ckpt_path}")


def main():
    args = parse_args()
    if not args.use_rl_answer_head and not args.use_rl_prior_selector:
        raise RuntimeError("Enable at least one of --use_rl_answer_head or --use_rl_prior_selector.")
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.peft == "qlora" and args.fsdp:
        raise RuntimeError("FSDP + QLoRA is not supported.")
    if args.train_vlm_with_rl and not args.ddp_find_unused_parameters:
        args.ddp_find_unused_parameters = True
    args.quantization_config = _build_quant_config(args)

    fsdp_plugin = None
    if args.fsdp:
        if FullyShardedDataParallelPlugin is None:
            raise RuntimeError("FSDP requested but accelerate FSDP plugin is unavailable.")
        fsdp_kwargs = {}
        use_orig_params = args.fsdp_use_orig_params or (args.peft != "none")
        if size_based_auto_wrap_policy is not None:
            fsdp_kwargs["auto_wrap_policy"] = functools.partial(
                size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params
            )
        else:
            try:
                params = inspect.signature(FullyShardedDataParallelPlugin).parameters
                if "min_num_params" in params:
                    fsdp_kwargs["min_num_params"] = args.fsdp_min_num_params
            except Exception:
                pass
        try:
            params = inspect.signature(FullyShardedDataParallelPlugin).parameters
            if "use_orig_params" in params:
                fsdp_kwargs["use_orig_params"] = use_orig_params
        except Exception:
            pass
        if args.fsdp_cpu_offload:
            if CPUOffload is None:
                raise RuntimeError("FSDP CPU offload requested but torch.distributed.fsdp is unavailable.")
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_kwargs)

    ddp_kwargs = None
    if not args.fsdp and DistributedDataParallelKwargs is not None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.ddp_find_unused_parameters)

    dataloader_config = None
    if DataLoaderConfiguration is not None:
        dataloader_config = DataLoaderConfiguration(split_batches=False, dispatch_batches=False)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
        dataloader_config=dataloader_config,
        log_with="wandb" if args.wandb else None,
    )

    if args.wandb:
        init_kwargs = {"wandb": {}}
        if args.wandb_entity:
            init_kwargs["wandb"]["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["wandb"]["name"] = args.wandb_run_name
        if args.wandb_tags:
            init_kwargs["wandb"]["tags"] = [item.strip() for item in args.wandb_tags.split(",") if item.strip()]
        accelerator.init_trackers(project_name=args.wandb_project, init_kwargs=init_kwargs)

    records = _load_records(args)
    belief_db = BeliefVectorDB.from_records(records, args, split_name=args.train_split)
    train_loader = build_rl_vqa_loader(args, args.train_split, args.batch_size, args.num_workers, is_train=True)
    val_loader = (
        build_rl_vqa_loader(args, args.val_split, args.batch_size, args.num_workers, is_train=False)
        if args.val_ratio > 0
        else None
    )

    model = build_model(args, device=accelerator.device)
    processor = model.backbone.processor
    model = _apply_peft(model, args)
    _configure_memory_optimizations(model, args)
    if args.vlm_checkpoint:
        vlm_ckpt = torch.load(args.vlm_checkpoint, map_location="cpu")
        state_dict = vlm_ckpt["model"] if isinstance(vlm_ckpt, dict) and "model" in vlm_ckpt else vlm_ckpt
        _load_checkpoint_state(model, state_dict, accelerator)

    probe_batch = next(iter(build_rl_vqa_loader(args, args.train_split, batch_size=1, num_workers=0, is_train=True)))
    with torch.no_grad():
        probe_state = model(probe_batch["inputs"], return_hidden_states=True, pooling=args.state_pooling)["pooled_state"]
    selector_policy = (
        PriorSelectorPolicy(
            hidden_dim=int(probe_state.shape[-1]),
            candidate_dim=belief_db.embedder.dim,
            dropout=args.policy_dropout,
        )
        if args.use_rl_prior_selector
        else None
    )
    answer_policy = (
        PPOAnswerPolicy(
            hidden_dim=int(probe_state.shape[-1]),
            action_dim=args.max_choice_options,
            dropout=args.policy_dropout,
        )
        if args.use_rl_answer_head
        else None
    )

    if not args.train_vlm_with_rl:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    param_groups = []
    if selector_policy is not None:
        param_groups.append({"params": [p for p in selector_policy.parameters() if p.requires_grad], "lr": args.selector_lr})
    if answer_policy is not None:
        param_groups.append({"params": [p for p in answer_policy.parameters() if p.requires_grad], "lr": args.policy_lr})
    if args.train_vlm_with_rl:
        param_groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": args.vlm_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    prepare_items = [model, optimizer, train_loader]
    if selector_policy is not None:
        prepare_items.insert(1, selector_policy)
    if answer_policy is not None:
        insert_idx = 2 if selector_policy is not None else 1
        prepare_items.insert(insert_idx, answer_policy)
    prepared = accelerator.prepare(*prepare_items)

    idx = 0
    model = prepared[idx]
    idx += 1
    if selector_policy is not None:
        selector_policy = prepared[idx]
        idx += 1
    if answer_policy is not None:
        answer_policy = prepared[idx]
        idx += 1
    optimizer = prepared[idx]
    idx += 1
    train_loader = prepared[idx]
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    total_params, trainable_params = _count_parameters(model)
    accelerator.print(f"vlm parameters total={total_params:,} trainable={trainable_params:,}")

    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _load_checkpoint_state(model, ckpt["model"], accelerator)
        if selector_policy is not None and "selector_policy" in ckpt:
            accelerator.unwrap_model(selector_policy).load_state_dict(ckpt["selector_policy"])
        if answer_policy is not None and "answer_policy" in ckpt:
            accelerator.unwrap_model(answer_policy).load_state_dict(ckpt["answer_policy"])
        if not args.load_model_only:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))
    elif args.answer_head_checkpoint and answer_policy is not None:
        answer_ckpt = torch.load(args.answer_head_checkpoint, map_location="cpu")
        policy_state = answer_ckpt.get("policy")
        if policy_state is None:
            raise RuntimeError(
                f"Checkpoint {args.answer_head_checkpoint} does not contain a 'policy' state for the RL answer head."
            )
        accelerator.unwrap_model(answer_policy).load_state_dict(policy_state, strict=False)

    for epoch in range(start_epoch, args.epochs):
        train_sampler = getattr(train_loader, "sampler", None)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if val_loader is not None:
            val_sampler = getattr(val_loader, "sampler", None)
            if hasattr(val_sampler, "set_epoch"):
                val_sampler.set_epoch(epoch)

        train_metrics, global_step = run_epoch(
            model, processor, belief_db, selector_policy, answer_policy, train_loader, optimizer, accelerator, args, True, global_step
        )
        accelerator.print(f"epoch={epoch} train_reward={train_metrics['reward']:.4f}")
        if val_loader is not None:
            with torch.no_grad():
                val_metrics, _ = run_epoch(
                    model, processor, belief_db, selector_policy, answer_policy, val_loader, optimizer, accelerator, args, False, global_step
                )
            accelerator.print(f"epoch={epoch} val_reward={val_metrics['reward']:.4f}")

        _save_checkpoint(model, selector_policy, answer_policy, optimizer, accelerator, args, epoch, global_step)

    if args.wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
