import argparse
import functools
import inspect
import os

import torch
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

from data_loading import build_train_loader
from model import ModelConfig, MultimodalBeliefModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", type=str, default="hd_epic_local", choices=["hd_epic_local"])
    parser.add_argument("--dataset_name", type=str, default="hd_epic_local")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--dataset_revision", type=str, default="main")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trust_remote_code_dataset", action="store_true", default=False)
    parser.add_argument("--no_trust_remote_code_dataset", dest="trust_remote_code_dataset", action="store_false")
    parser.add_argument("--video_column", type=str, default="video")
    parser.add_argument("--video_root", type=str, default="")
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--annotation_path", type=str, default="")
    parser.add_argument("--metadata_root", type=str, default="")
    parser.add_argument("--conversations_column", type=str, default="conversations")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--video_id_column", type=str, default="video_id")
    parser.add_argument("--video_path_column", type=str, default="video_path")
    parser.add_argument("--participant_column", type=str, default="participant_id")
    parser.add_argument("--question_column", type=str, default="question")
    parser.add_argument("--answer_column", type=str, default="answer")
    parser.add_argument("--options_column", type=str, default="options")
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--shuffle_buffer", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=1)
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
    parser.add_argument("--detect_anomaly", action="store_true")

    parser.add_argument(
        "--vl_backend",
        type=str,
        default="internvl",
        choices=["internvl"],
    )
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

    parser.add_argument("--peft", type=str, default="none", choices=["none", "lora", "qlora"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_belief_sft")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--load_model_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_max_new_tokens", type=int, default=64)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="belief-vlm")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="")

    return parser.parse_args()


def _resolve_vl_model_preset(args):
    if args.vl_model_preset == "internvl3_5_1b":
        args.vl_backend = "internvl"
        args.vl_model_name = "OpenGVLab/InternVL3_5-1B-HF"
    elif args.vl_model_preset == "internvl3_5_2b":
        args.vl_backend = "internvl"
        args.vl_model_name = "OpenGVLab/InternVL3_5-2B-HF"
    elif args.vl_model_preset == "internvl3_5_4b":
        args.vl_backend = "internvl"
        args.vl_model_name = "OpenGVLab/InternVL3_5-4B-HF"
    elif args.vl_model_preset == "internvl3_5_8b":
        args.vl_backend = "internvl"
        args.vl_model_name = "OpenGVLab/InternVL3_5-8B-HF"


def _parse_lora_targets(args):
    if args.lora_target_modules:
        return [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _apply_peft(model, args):
    if args.peft == "none":
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise RuntimeError("PEFT requested but `peft` is not installed.") from e

    for param in model.backbone.model.parameters():
        param.requires_grad = False

    if args.peft == "qlora":
        model.backbone.model = prepare_model_for_kbit_training(model.backbone.model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=_parse_lora_targets(args),
        task_type="CAUSAL_LM",
    )
    model.backbone.model = get_peft_model(model.backbone.model, lora_cfg)
    return model


def build_model(args, device):
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        freeze_vl=args.freeze_vl,
        quantization_config=getattr(args, "quantization_config", None),
        use_cache=not args.disable_vl_cache,
    )
    return MultimodalBeliefModel(cfg, device=device)


def _configure_memory_optimizations(model, args):
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if hasattr(model.backbone.model, "config") and hasattr(model.backbone.model.config, "use_cache"):
        model.backbone.model.config.use_cache = False if args.disable_vl_cache or args.gradient_checkpointing else model.backbone.model.config.use_cache

    if args.gradient_checkpointing:
        fn = getattr(model.backbone.model, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                fn()
        enable_inputs = getattr(model.backbone.model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            try:
                enable_inputs()
            except Exception:
                pass


def _count_parameters(model):
    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return total, trainable


def _load_checkpoint_state(model, ckpt_state, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    try:
        unwrapped.load_state_dict(ckpt_state)
        return
    except RuntimeError:
        incompatible = unwrapped.load_state_dict(ckpt_state, strict=False)
        accelerator.print(
            "Non-strict checkpoint load complete: "
            f"missing_keys={len(getattr(incompatible, 'missing_keys', []))} "
            f"unexpected_keys={len(getattr(incompatible, 'unexpected_keys', []))}"
        )



def run_epoch(model, loader, optimizer, accelerator, args, train, global_step):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_examples = 0
    step = 0

    for batch in loader:
        step += 1
        labels = batch["labels"].to(accelerator.device)
        inputs = {
            key: value.to(accelerator.device) if torch.is_tensor(value) else value
            for key, value in batch["inputs"].items()
        }

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                outputs = model(inputs, labels=labels)
                # debug_ids = accelerator.unwrap_model(model).generate(
                #     inputs, max_new_tokens=args.eval_max_new_tokens
                # )
                # debug_text = accelerator.unwrap_model(model).backbone.tokenizer.batch_decode(
                #     debug_ids, skip_special_tokens=False
                # )
                # accelerator.print(debug_text)
                loss = outputs["loss"]
                if train:
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size

        if args.log_every > 0 and step % args.log_every == 0:
            avg_loss = total_loss / max(total_examples, 1)
            phase = "train" if train else "val"
            accelerator.print(f"{phase} step={step} loss={avg_loss:.4f}")
            if args.wandb:
                metrics = {f"{phase}/loss": avg_loss}
                if train:
                    metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                    accelerator.log(metrics, step=global_step + step)
                else:
                    accelerator.log(metrics, step=global_step)

    avg_loss = total_loss / max(total_examples, 1)
    return avg_loss, (global_step + step if train else global_step)


def main():
    args = parse_args()
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    if args.peft == "qlora" and args.fsdp:
        raise RuntimeError("FSDP + QLoRA is not supported.")

    if args.peft == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("QLoRA requested but bitsandbytes/transformers are not available.") from e
        if args.vl_dtype == "float16":
            compute_dtype = torch.float16
        elif args.vl_dtype == "float32":
            compute_dtype = torch.float32
        else:
            compute_dtype = torch.bfloat16
        args.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        args.quantization_config = None

    fsdp_plugin = None
    if args.fsdp:
        if FullyShardedDataParallelPlugin is None:
            raise RuntimeError("FSDP requested but accelerate FSDP plugin is unavailable.")
        fsdp_kwargs = {}
        use_orig_params = args.fsdp_use_orig_params or (args.peft != "none")
        if size_based_auto_wrap_policy is not None:
            fsdp_kwargs["auto_wrap_policy"] = functools.partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)
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
        dataloader_config = DataLoaderConfiguration(
            split_batches=False,
            dispatch_batches=False,
        )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
        dataloader_config=dataloader_config,
        log_with="wandb" if args.wandb else None,
    )

    if args.wandb:
        init_kwargs = {"wandb": {"project": args.wandb_project}}
        if args.wandb_entity:
            init_kwargs["wandb"]["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["wandb"]["name"] = args.wandb_run_name
        if args.wandb_tags:
            init_kwargs["wandb"]["tags"] = [item.strip() for item in args.wandb_tags.split(",") if item.strip()]
        accelerator.init_trackers(project_name=args.wandb_project, init_kwargs=init_kwargs)

    accelerator.print(
        f"dataset_type={args.dataset_type} dataset={args.dataset_name} "
        f"train_alias={args.train_split} val_alias={args.val_split}"
    )
    train_loader = build_train_loader(args, args.train_split, args.batch_size, args.num_workers, is_train=True)
    val_loader = build_train_loader(args, args.val_split, args.batch_size, args.num_workers, is_train=False) if args.val_ratio > 0 else None

    model = build_model(args, device=accelerator.device)
    model = _apply_peft(model, args)
    _configure_memory_optimizations(model, args)

    total_params, trainable_params = _count_parameters(model)
    accelerator.print(f"parameters total={total_params:,} trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _load_checkpoint_state(model, ckpt["model"], accelerator)
        if not args.load_model_only:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))
            accelerator.print(f"resumed checkpoint={args.resume_checkpoint} start_epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            accelerator=accelerator,
            args=args,
            train=True,
            global_step=global_step,
        )
        accelerator.print(f"epoch={epoch} train_loss={train_loss:.4f}")
        if args.wandb:
            accelerator.log({"train/epoch_loss": train_loss}, step=global_step)

        if val_loader is not None:
            with torch.no_grad():
                val_loss, _ = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    args=args,
                    train=False,
                    global_step=global_step,
                )
            accelerator.print(f"epoch={epoch} val_loss={val_loss:.4f}")
            if args.wandb:
                accelerator.log({"val/epoch_loss": val_loss}, step=global_step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
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
