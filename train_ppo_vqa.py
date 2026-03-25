import argparse
import functools
import inspect
import os
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

from data_loading import build_rl_vqa_loader
from train import (
    _apply_peft,
    _configure_memory_optimizations,
    _count_parameters,
    _load_checkpoint_state,
    _resolve_vl_model_preset,
    build_model,
)


class PPOAnswerPolicy(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        hidden = self.backbone(state)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO policy over VLM embeddings for multiple-choice VQA.")

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
    parser.add_argument("--save_dir", type=str, default="checkpoints_ppo_vqa")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--load_model_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vlm-ppo-vqa")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="")
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


def _masked_logits(logits, num_choices):
    action_dim = logits.shape[-1]
    choice_ids = torch.arange(action_dim, device=logits.device).unsqueeze(0)
    mask = choice_ids < num_choices.unsqueeze(1)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _extract_state(model, batch_inputs, args):
    ctx = nullcontext() if args.train_vlm_with_rl else torch.no_grad()
    with ctx:
        encoded = model(batch_inputs, return_hidden_states=True, pooling=args.state_pooling)
    pooled_state = encoded["pooled_state"]
    if not args.train_vlm_with_rl:
        pooled_state = pooled_state.detach()
    return pooled_state.float()


def _policy_forward(model, policy, batch_inputs, num_choices, args):
    state = _extract_state(model, batch_inputs, args)
    logits, values = policy(state)
    masked_logits = _masked_logits(logits, num_choices.to(logits.device))
    dist = torch.distributions.Categorical(logits=masked_logits)
    return dist, values


def _evaluate_policy(model, policy, loader, accelerator, args):
    model.eval()
    policy.eval()
    total = 0
    correct = 0
    for batch in loader:
        num_choices = batch["num_choices"].to(accelerator.device)
        correct_idx = batch["correct_idx"].to(accelerator.device)
        with torch.no_grad():
            dist, _ = _policy_forward(model, policy, batch["inputs"], num_choices, args)
            actions = torch.argmax(dist.logits, dim=-1)
        correct += int((actions == correct_idx).sum().item())
        total += int(correct_idx.numel())
    accuracy = correct / max(total, 1)
    return accuracy


def run_epoch(model, policy, loader, optimizer, accelerator, args, train, global_step):
    model.train(args.train_vlm_with_rl and train)
    if not args.train_vlm_with_rl:
        model.eval()
    policy.train() if train else policy.eval()

    total_reward = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_examples = 0
    step = 0

    for batch in loader:
        step += 1
        num_choices = batch["num_choices"].to(accelerator.device)
        if int(num_choices.max().item()) > args.max_choice_options:
            raise RuntimeError(
                f"Encountered {int(num_choices.max().item())} choices, but policy head supports only "
                f"{args.max_choice_options}. Increase --max_choice_options."
            )
        correct_idx = batch["correct_idx"].to(accelerator.device)

        with torch.no_grad():
            rollout_dist, rollout_values = _policy_forward(model, policy, batch["inputs"], num_choices, args)
            if train:
                actions = rollout_dist.sample()
            else:
                actions = torch.argmax(rollout_dist.logits, dim=-1)
            old_log_probs = rollout_dist.log_prob(actions)
            rewards = (actions == correct_idx).float() * float(args.reward_scale)
            returns = rewards
            advantages = returns - rollout_values.detach()

        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0

        if train:
            with accelerator.accumulate(policy):
                for _ in range(args.ppo_epochs):
                    dist, values = _policy_forward(model, policy, batch["inputs"], num_choices, args)
                    log_probs = dist.log_prob(actions)
                    ratio = torch.exp(log_probs - old_log_probs)
                    unclipped = ratio * advantages
                    clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(values, returns)
                    entropy = dist.entropy().mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                        if args.train_vlm_with_rl:
                            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    policy_loss_value = float(policy_loss.detach().item())
                    value_loss_value = float(value_loss.detach().item())
                    entropy_value = float(entropy.detach().item())
        else:
            with torch.no_grad():
                dist, values = _policy_forward(model, policy, batch["inputs"], num_choices, args)
                log_probs = dist.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
                policy_loss_value = float((-torch.min(unclipped, clipped).mean()).item())
                value_loss_value = float(F.mse_loss(values, returns).item())
                entropy_value = float(dist.entropy().mean().item())

        batch_size = int(correct_idx.numel())
        total_examples += batch_size
        total_reward += float(rewards.sum().item())
        total_policy_loss += policy_loss_value * batch_size
        total_value_loss += value_loss_value * batch_size
        total_entropy += entropy_value * batch_size

        if args.log_every > 0 and step % args.log_every == 0:
            avg_reward = total_reward / max(total_examples, 1)
            avg_policy_loss = total_policy_loss / max(total_examples, 1)
            avg_value_loss = total_value_loss / max(total_examples, 1)
            avg_entropy = total_entropy / max(total_examples, 1)
            phase = "train" if train else "val"
            accelerator.print(
                f"{phase} step={step} reward={avg_reward:.4f} policy_loss={avg_policy_loss:.4f} "
                f"value_loss={avg_value_loss:.4f} entropy={avg_entropy:.4f}"
            )
            if args.wandb:
                metrics = {
                    f"{phase}/reward": avg_reward,
                    f"{phase}/policy_loss": avg_policy_loss,
                    f"{phase}/value_loss": avg_value_loss,
                    f"{phase}/entropy": avg_entropy,
                }
                if train:
                    metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                    accelerator.log(metrics, step=global_step + step)
                else:
                    accelerator.log(metrics, step=global_step)

    metrics = {
        "reward": total_reward / max(total_examples, 1),
        "policy_loss": total_policy_loss / max(total_examples, 1),
        "value_loss": total_value_loss / max(total_examples, 1),
        "entropy": total_entropy / max(total_examples, 1),
    }
    return metrics, (global_step + step if train else global_step)


def main():
    args = parse_args()
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

    train_loader = build_rl_vqa_loader(args, args.train_split, args.batch_size, args.num_workers, is_train=True)
    val_loader = (
        build_rl_vqa_loader(args, args.val_split, args.batch_size, args.num_workers, is_train=False)
        if args.val_ratio > 0
        else None
    )

    model = build_model(args, device=accelerator.device)
    model = _apply_peft(model, args)
    _configure_memory_optimizations(model, args)
    if args.vlm_checkpoint:
        vlm_ckpt = torch.load(args.vlm_checkpoint, map_location="cpu")
        state_dict = vlm_ckpt["model"] if isinstance(vlm_ckpt, dict) and "model" in vlm_ckpt else vlm_ckpt
        _load_checkpoint_state(model, state_dict, accelerator)

    probe_batch = next(iter(build_rl_vqa_loader(args, args.train_split, batch_size=1, num_workers=0, is_train=True)))
    with torch.no_grad():
        probe_state = model(probe_batch["inputs"], return_hidden_states=True, pooling=args.state_pooling)["pooled_state"]
    policy = PPOAnswerPolicy(
        hidden_dim=int(probe_state.shape[-1]),
        action_dim=args.max_choice_options,
        dropout=args.policy_dropout,
    )

    if not args.train_vlm_with_rl:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    param_groups = [{"params": [p for p in policy.parameters() if p.requires_grad], "lr": args.policy_lr}]
    if args.train_vlm_with_rl:
        param_groups.append(
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.vlm_lr}
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    model, policy, optimizer, train_loader = accelerator.prepare(model, policy, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    total_params, trainable_params = _count_parameters(policy)
    accelerator.print(f"policy parameters total={total_params:,} trainable={trainable_params:,}")
    if args.train_vlm_with_rl:
        total_params, trainable_params = _count_parameters(model)
        accelerator.print(f"vlm parameters total={total_params:,} trainable={trainable_params:,}")

    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _load_checkpoint_state(model, ckpt["model"], accelerator)
        accelerator.unwrap_model(policy).load_state_dict(ckpt["policy"])
        if not args.load_model_only:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))

    for epoch in range(start_epoch, args.epochs):
        train_metrics, global_step = run_epoch(
            model=model,
            policy=policy,
            loader=train_loader,
            optimizer=optimizer,
            accelerator=accelerator,
            args=args,
            train=True,
            global_step=global_step,
        )
        accelerator.print(
            f"epoch={epoch} train_reward={train_metrics['reward']:.4f} "
            f"train_policy_loss={train_metrics['policy_loss']:.4f}"
        )
        if args.wandb:
            accelerator.log(
                {
                    "train/epoch_reward": train_metrics["reward"],
                    "train/epoch_policy_loss": train_metrics["policy_loss"],
                    "train/epoch_value_loss": train_metrics["value_loss"],
                    "train/epoch_entropy": train_metrics["entropy"],
                },
                step=global_step,
            )

        if val_loader is not None:
            with torch.no_grad():
                val_metrics, _ = run_epoch(
                    model=model,
                    policy=policy,
                    loader=val_loader,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    args=args,
                    train=False,
                    global_step=global_step,
                )
                val_acc = _evaluate_policy(model, policy, val_loader, accelerator, args)
            accelerator.print(
                f"epoch={epoch} val_reward={val_metrics['reward']:.4f} val_acc={val_acc:.4f}"
            )
            if args.wandb:
                accelerator.log(
                    {
                        "val/epoch_reward": val_metrics["reward"],
                        "val/epoch_policy_loss": val_metrics["policy_loss"],
                        "val/epoch_value_loss": val_metrics["value_loss"],
                        "val/epoch_entropy": val_metrics["entropy"],
                        "val/accuracy": val_acc,
                    },
                    step=global_step,
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_policy = accelerator.unwrap_model(policy)
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(
                {
                    "model": unwrapped_model.state_dict(),
                    "policy": unwrapped_policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": vars(args),
                },
                ckpt_path,
            )
            accelerator.print(f"saved {ckpt_path}")

    if args.wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
