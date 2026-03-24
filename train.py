import argparse
import functools
import inspect
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

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
    parser.add_argument("--use_future_predictor", action="store_true")
    parser.add_argument("--future_predictor_checkpoint", type=str, default="")
    parser.add_argument("--future_frames", type=int, default=0)
    parser.add_argument("--use_belief_model", action="store_true")
    parser.add_argument("--belief_hidden_dim", type=int, default=1024)
    parser.add_argument("--belief_latent_dim", type=int, default=512)
    parser.add_argument("--belief_num_layers", type=int, default=2)
    parser.add_argument("--belief_num_heads", type=int, default=8)
    parser.add_argument("--belief_num_tokens", type=int, default=4)
    parser.add_argument("--belief_dropout", type=float, default=0.1)
    parser.add_argument("--belief_fusion_scope", type=str, default="vision_text", choices=["vision_only", "vision_text", "late_prefix"])
    parser.add_argument("--belief_use_recurrence", action="store_true")
    parser.add_argument("--belief_temporal_chunks", type=int, default=4)
    parser.add_argument("--disable_late_belief_prefix", action="store_true")

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
    parser.add_argument("--train_objective", type=str, default="hybrid", choices=["generative", "mc", "hybrid"])
    parser.add_argument("--lm_loss_weight", type=float, default=1.0)
    parser.add_argument("--mc_loss_weight", type=float, default=1.0)
    parser.add_argument("--belief_aux_loss_weight", type=float, default=0.25)
    parser.add_argument("--mc_train_num_choices", type=int, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_best_metric", type=str, default="val_mc_acc", choices=["val_mc_acc", "val_loss"])
    parser.add_argument("--save_dir", type=str, default="checkpoints_belief_sft")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--load_model_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_max_new_tokens", type=int, default=64)
    parser.add_argument("--debug_generate", action="store_true")
    parser.add_argument("--debug_generate_every", type=int, default=0)

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
        future_predictor_checkpoint=args.future_predictor_checkpoint if args.use_future_predictor else "",
        future_predictor_bundle=getattr(args, "future_predictor_bundle", None) if args.use_future_predictor else None,
        future_context_frames=args.video_frames if (args.use_future_predictor or args.use_belief_model) else 0,
        future_frames=args.future_frames if args.use_future_predictor else 0,
        use_belief_model=args.use_belief_model,
        belief_hidden_dim=args.belief_hidden_dim,
        belief_latent_dim=args.belief_latent_dim,
        belief_num_layers=args.belief_num_layers,
        belief_num_heads=args.belief_num_heads,
        belief_num_tokens=args.belief_num_tokens,
        belief_dropout=args.belief_dropout,
        belief_max_text_tokens=args.vl_max_text_len,
        belief_fusion_scope=args.belief_fusion_scope,
        belief_use_recurrence=args.belief_use_recurrence,
        belief_temporal_chunks=args.belief_temporal_chunks,
        disable_late_belief_prefix=args.disable_late_belief_prefix,
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


def _restore_bundled_future_predictor_args(args, ckpt):
    if not isinstance(ckpt, dict):
        return
    bundle = ckpt.get("future_predictor")
    if bundle is None:
        return
    args.use_future_predictor = True
    if not getattr(args, "future_predictor_checkpoint", ""):
        args.future_predictor_bundle = bundle
    bundle_args = bundle.get("args", {})
    if int(getattr(args, "future_frames", 0) or 0) <= 0:
        args.future_frames = int(bundle_args.get("future_frames", args.future_frames or 0))


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


def _reshape_choice_scores(flat_scores, choice_group_sizes):
    max_choices = int(choice_group_sizes.max().item())
    grouped = flat_scores.new_full((choice_group_sizes.shape[0], max_choices), -1e9)
    start = 0
    for row, group_size in enumerate(choice_group_sizes.tolist()):
        grouped[row, :group_size] = flat_scores[start:start + group_size]
        start += group_size
    return grouped


def _compute_choice_losses(model, choice_inputs, choice_labels, choice_group_sizes, correct_choice_idx):
    outputs = model(choice_inputs, labels=choice_labels, return_aux=True)
    choice_nll = _sequence_nll(outputs["logits"], choice_labels)
    choice_scores = _reshape_choice_scores(-choice_nll, choice_group_sizes)
    mc_loss = F.cross_entropy(choice_scores, correct_choice_idx)
    mc_acc = (choice_scores.argmax(dim=1) == correct_choice_idx).float().mean()
    result = {
        "mc_loss": mc_loss,
        "mc_acc": mc_acc,
        "choice_scores": choice_scores,
    }
    belief_summary = outputs.get("belief_summary")
    choice_text_summary = outputs.get("choice_text_summary")
    if belief_summary is not None and choice_text_summary is not None:
        belief_summary = F.normalize(belief_summary.float(), dim=-1)
        choice_text_summary = F.normalize(choice_text_summary.float(), dim=-1)
        flat_aux_scores = (belief_summary * choice_text_summary).sum(dim=-1)
        aux_scores = _reshape_choice_scores(flat_aux_scores, choice_group_sizes)
        result["belief_aux_loss"] = F.cross_entropy(aux_scores, correct_choice_idx)
        result["belief_aux_acc"] = (aux_scores.argmax(dim=1) == correct_choice_idx).float().mean()
    return result


def _estimate_train_steps(loader, args):
    try:
        return max(1, len(loader) * max(1, args.epochs))
    except TypeError:
        dataset = getattr(loader, "dataset", None)
        records = getattr(dataset, "records", None)
        if records is None:
            return max(1, args.epochs)
        val_ratio = max(0.0, min(0.5, float(args.val_ratio)))
        if val_ratio > 0.0:
            approx_records = int(round(len(records) * (1.0 - val_ratio)))
        else:
            approx_records = len(records)
        batch_size = max(1, int(args.batch_size))
        approx_batches = max(1, (approx_records + batch_size - 1) // batch_size)
        return max(1, approx_batches * max(1, args.epochs))



def run_epoch(model, loader, optimizer, scheduler, accelerator, args, train, global_step):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_examples = 0
    step = 0
    total_lm_loss = 0.0
    total_mc_loss = 0.0
    total_belief_aux_loss = 0.0
    total_mc_acc = 0.0

    for batch in loader:
        step += 1
        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                metrics = {}
                batch_size = 0
                loss = None

                if args.train_objective == "generative":
                    labels = batch["labels"].to(accelerator.device)
                    inputs = {
                        key: value.to(accelerator.device) if torch.is_tensor(value) else value
                        for key, value in batch["inputs"].items()
                    }
                    outputs = model(inputs, labels=labels)
                    loss = outputs["loss"]
                    batch_size = labels.size(0)
                    metrics["lm_loss"] = loss.detach()
                    if args.debug_generate and args.debug_generate_every > 0 and step % args.debug_generate_every == 0:
                        debug_ids = accelerator.unwrap_model(model).generate(inputs, max_new_tokens=args.eval_max_new_tokens)
                        prompt_len = int(inputs["input_ids"].shape[1])
                        debug_new_tokens = debug_ids[:, prompt_len:]
                        debug_text = accelerator.unwrap_model(model).backbone.tokenizer.batch_decode(
                            debug_new_tokens, skip_special_tokens=True
                        )
                        accelerator.print(f"debug_text step={step}: {debug_text}")
                else:
                    gold_labels = batch["gold_labels"].to(accelerator.device)
                    gold_inputs = {
                        key: value.to(accelerator.device) if torch.is_tensor(value) else value
                        for key, value in batch["gold_inputs"].items()
                    }
                    choice_labels = batch["choice_labels"].to(accelerator.device)
                    choice_inputs = {
                        key: value.to(accelerator.device) if torch.is_tensor(value) else value
                        for key, value in batch["choice_inputs"].items()
                    }
                    choice_group_sizes = batch["choice_group_sizes"].to(accelerator.device)
                    correct_choice_idx = batch["correct_choice_idx"].to(accelerator.device)
                    batch_size = correct_choice_idx.size(0)
                    loss = gold_labels.new_zeros((), dtype=torch.float32)

                    if args.train_objective in {"generative", "hybrid"}:
                        gold_outputs = model(gold_inputs, labels=gold_labels)
                        lm_loss = gold_outputs["loss"]
                        metrics["lm_loss"] = lm_loss.detach()
                        loss = loss + args.lm_loss_weight * lm_loss

                    choice_metrics = _compute_choice_losses(
                        model=model,
                        choice_inputs=choice_inputs,
                        choice_labels=choice_labels,
                        choice_group_sizes=choice_group_sizes,
                        correct_choice_idx=correct_choice_idx,
                    )
                    metrics.update({k: v.detach() if torch.is_tensor(v) else v for k, v in choice_metrics.items() if k != "choice_scores"})
                    loss = loss + args.mc_loss_weight * choice_metrics["mc_loss"]
                    if "belief_aux_loss" in choice_metrics:
                        loss = loss + args.belief_aux_loss_weight * choice_metrics["belief_aux_loss"]

                if train:
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size
        if "lm_loss" in metrics:
            total_lm_loss += float(metrics["lm_loss"].item()) * batch_size
        if "mc_loss" in metrics:
            total_mc_loss += float(metrics["mc_loss"].item()) * batch_size
        if "belief_aux_loss" in metrics:
            total_belief_aux_loss += float(metrics["belief_aux_loss"].item()) * batch_size
        if "mc_acc" in metrics:
            total_mc_acc += float(metrics["mc_acc"].item()) * batch_size

        if args.log_every > 0 and step % args.log_every == 0:
            avg_loss = total_loss / max(total_examples, 1)
            phase = "train" if train else "val"
            message = f"{phase} step={step} loss={avg_loss:.4f}"
            if total_lm_loss > 0:
                message += f" lm_loss={total_lm_loss / max(total_examples, 1):.4f}"
            if total_mc_loss > 0:
                message += f" mc_loss={total_mc_loss / max(total_examples, 1):.4f}"
                message += f" mc_acc={total_mc_acc / max(total_examples, 1):.4f}"
            if total_belief_aux_loss > 0:
                message += f" belief_aux_loss={total_belief_aux_loss / max(total_examples, 1):.4f}"
            accelerator.print(message)
            if args.wandb:
                wandb_metrics = {f"{phase}/loss": avg_loss}
                if total_lm_loss > 0:
                    wandb_metrics[f"{phase}/lm_loss"] = total_lm_loss / max(total_examples, 1)
                if total_mc_loss > 0:
                    wandb_metrics[f"{phase}/mc_loss"] = total_mc_loss / max(total_examples, 1)
                    wandb_metrics[f"{phase}/mc_acc"] = total_mc_acc / max(total_examples, 1)
                if total_belief_aux_loss > 0:
                    wandb_metrics[f"{phase}/belief_aux_loss"] = total_belief_aux_loss / max(total_examples, 1)
                if train:
                    wandb_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                    accelerator.log(wandb_metrics, step=global_step + step)
                else:
                    accelerator.log(wandb_metrics, step=global_step)

    denom = max(total_examples, 1)
    return {
        "loss": total_loss / denom,
        "lm_loss": total_lm_loss / denom if total_lm_loss > 0 else 0.0,
        "mc_loss": total_mc_loss / denom if total_mc_loss > 0 else 0.0,
        "belief_aux_loss": total_belief_aux_loss / denom if total_belief_aux_loss > 0 else 0.0,
        "mc_acc": total_mc_acc / denom if total_mc_acc > 0 else 0.0,
        "global_step": (global_step + step if train else global_step),
    }


def main():
    args = parse_args()
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_future_predictor:
        if not args.future_predictor_checkpoint:
            raise RuntimeError("--use_future_predictor requires --future_predictor_checkpoint.")
        if args.future_frames <= 0:
            raise RuntimeError("--use_future_predictor requires --future_frames > 0.")
    if args.use_future_predictor and args.use_belief_model:
        raise RuntimeError("Use either --use_future_predictor or --use_belief_model, not both.")
    if args.use_belief_model and args.video_frames < 1:
        raise RuntimeError("--use_belief_model requires --video_frames >= 1.")
    if args.train_objective in {"mc", "hybrid"} and not args.annotation_path:
        raise RuntimeError("Multiple-choice training requires annotation records with choices.")

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    if args.peft == "qlora" and args.fsdp:
        raise RuntimeError("FSDP + QLoRA is not supported.")
    if args.use_future_predictor and not args.ddp_find_unused_parameters:
        args.ddp_find_unused_parameters = True
        print("Enabling DDP find_unused_parameters for future-conditioned training.")
    if args.use_belief_model and not args.ddp_find_unused_parameters:
        args.ddp_find_unused_parameters = True
        print("Enabling DDP find_unused_parameters for belief-conditioned training.")

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
        init_kwargs = {"wandb": {}}
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

    resume_ckpt = None
    if args.resume_checkpoint:
        resume_ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        _restore_bundled_future_predictor_args(args, resume_ckpt)

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
    total_training_steps = _estimate_train_steps(train_loader, args)
    if args.lr_scheduler == "cosine":
        warmup_steps = int(total_training_steps * max(0.0, float(args.warmup_ratio)))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        scheduler = None

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf") if args.save_best_metric == "val_mc_acc" else float("inf")
    if args.resume_checkpoint:
        ckpt = resume_ckpt
        _load_checkpoint_state(model, ckpt["model"], accelerator)
        if not args.load_model_only:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))
            best_metric = float(ckpt.get("best_metric", best_metric))
            accelerator.print(f"resumed checkpoint={args.resume_checkpoint} start_epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            args=args,
            train=True,
            global_step=global_step,
        )
        global_step = int(train_metrics["global_step"])
        accelerator.print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_lm_loss={train_metrics['lm_loss']:.4f} "
            f"train_mc_loss={train_metrics['mc_loss']:.4f} "
            f"train_mc_acc={train_metrics['mc_acc']:.4f}"
        )
        if args.wandb:
            accelerator.log(
                {
                    "train/epoch_loss": train_metrics["loss"],
                    "train/epoch_lm_loss": train_metrics["lm_loss"],
                    "train/epoch_mc_loss": train_metrics["mc_loss"],
                    "train/epoch_mc_acc": train_metrics["mc_acc"],
                },
                step=global_step,
            )

        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=optimizer,
                    scheduler=None,
                    accelerator=accelerator,
                    args=args,
                    train=False,
                    global_step=global_step,
                )
            accelerator.print(
                f"epoch={epoch} val_loss={val_metrics['loss']:.4f} "
                f"val_lm_loss={val_metrics['lm_loss']:.4f} "
                f"val_mc_loss={val_metrics['mc_loss']:.4f} "
                f"val_mc_acc={val_metrics['mc_acc']:.4f}"
            )
            if args.wandb:
                accelerator.log(
                    {
                        "val/epoch_loss": val_metrics["loss"],
                        "val/epoch_lm_loss": val_metrics["lm_loss"],
                        "val/epoch_mc_loss": val_metrics["mc_loss"],
                        "val/epoch_mc_acc": val_metrics["mc_acc"],
                    },
                    step=global_step,
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt")
            current_metric = None
            if val_metrics is not None:
                current_metric = val_metrics["mc_acc"] if args.save_best_metric == "val_mc_acc" else val_metrics["loss"]
                is_better = current_metric > best_metric if args.save_best_metric == "val_mc_acc" else current_metric < best_metric
                if is_better:
                    best_metric = current_metric
                    best_path = os.path.join(args.save_dir, "best_model.pt")
                    torch.save(
                        {
                            "model": unwrapped.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "args": vars(args),
                            "future_predictor": unwrapped.export_future_predictor_bundle(),
                            "best_metric": best_metric,
                        },
                        best_path,
                    )
                    accelerator.print(f"saved {best_path}")
            torch.save(
                {
                    "model": unwrapped.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": vars(args),
                    "future_predictor": unwrapped.export_future_predictor_bundle(),
                    "best_metric": best_metric,
                },
                ckpt_path,
            )
            accelerator.print(f"saved {ckpt_path}")

    if args.wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
