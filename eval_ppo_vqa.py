import argparse
import json
from collections import defaultdict
from types import SimpleNamespace

import torch

from data_loading import build_rl_vqa_loader
from train import _apply_peft, _configure_memory_optimizations, _resolve_vl_model_preset, build_model
from train_ppo_vqa import PPOAnswerPolicy, _build_quant_config, _masked_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO VQA policy checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--metadata_root", type=str, default="")
    parser.add_argument("--video_id_column", type=str, default="video_id")
    parser.add_argument("--video_path_column", type=str, default="video_path")
    parser.add_argument("--participant_column", type=str, default="participant_id")
    parser.add_argument("--question_column", type=str, default="question")
    parser.add_argument("--answer_column", type=str, default="answer")
    parser.add_argument("--options_column", type=str, default="options")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--video_frames", type=int, default=None)
    parser.add_argument("--vl_max_text_len", type=int, default=None)
    parser.add_argument("--vl_model_preset", type=str, default=None)
    parser.add_argument("--vl_model_name", type=str, default=None)
    parser.add_argument("--vl_backend", type=str, default=None)
    parser.add_argument("--vl_dtype", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--peft", type=str, default=None)
    parser.add_argument("--freeze_vl", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--state_pooling", type=str, default=None)
    parser.add_argument("--max_choice_options", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--print_samples", type=int, default=10)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--save_predictions", type=str, default="")
    return parser.parse_args()


def _merge_args(cli_args, ckpt_args):
    merged = dict(ckpt_args)
    merged.update(
        {
            "annotation_path": cli_args.annotation_path,
            "video_root": cli_args.video_root,
            "video_extension": cli_args.video_extension,
            "metadata_root": cli_args.metadata_root,
            "video_id_column": cli_args.video_id_column,
            "video_path_column": cli_args.video_path_column,
            "participant_column": cli_args.participant_column,
            "question_column": cli_args.question_column,
            "answer_column": cli_args.answer_column,
            "options_column": cli_args.options_column,
            "id_column": cli_args.id_column,
            "batch_size": cli_args.batch_size,
            "num_workers": cli_args.num_workers,
            "max_samples_per_split": cli_args.max_samples_per_split,
        }
    )

    optional_overrides = (
        "video_frames",
        "vl_max_text_len",
        "vl_model_preset",
        "vl_model_name",
        "vl_backend",
        "vl_dtype",
        "mixed_precision",
        "peft",
        "state_pooling",
        "max_choice_options",
    )
    for key in optional_overrides:
        value = getattr(cli_args, key)
        if value is not None:
            merged[key] = value

    if cli_args.freeze_vl:
        merged["freeze_vl"] = True
    if cli_args.gradient_checkpointing:
        merged["gradient_checkpointing"] = True
    if cli_args.disable_vl_cache:
        merged["disable_vl_cache"] = True
    if cli_args.allow_tf32:
        merged["allow_tf32"] = True

    merged.setdefault("dataset_type", "hd_epic_local")
    merged.setdefault("dataset_name", "hd_epic_local")
    merged.setdefault("train_split", "train")
    merged.setdefault("val_split", "validation")
    merged.setdefault("val_ratio", 0.0)
    merged.setdefault("vl_backend", "internvl")
    merged.setdefault("vl_model_preset", "internvl3_5_1b")
    merged.setdefault("vl_model_name", "OpenGVLab/InternVL3_5-1B-HF")
    merged.setdefault("vl_dtype", "bfloat16")
    merged.setdefault("video_frames", 8)
    merged.setdefault("vl_max_text_len", 256)
    merged.setdefault("freeze_vl", False)
    merged.setdefault("peft", "none")
    merged.setdefault("lora_r", 16)
    merged.setdefault("lora_alpha", 32)
    merged.setdefault("lora_dropout", 0.05)
    merged.setdefault("lora_target_modules", "")
    merged.setdefault("lora_bias", "none")
    merged.setdefault("gradient_checkpointing", False)
    merged.setdefault("disable_vl_cache", False)
    merged.setdefault("allow_tf32", False)
    merged.setdefault("state_pooling", "last")
    merged.setdefault("policy_dropout", 0.1)
    merged.setdefault("max_choice_options", 8)
    merged.setdefault("train_vlm_with_rl", False)
    return SimpleNamespace(**merged)


def _load_model_and_policy(checkpoint_path, args, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args.quantization_config = _build_quant_config(args)
    _resolve_vl_model_preset(args)

    model = build_model(args, device=device)
    model = _apply_peft(model, args)
    _configure_memory_optimizations(model, args)
    model.load_state_dict(ckpt["model"], strict=False)

    policy_state = ckpt["policy"]
    policy = PPOAnswerPolicy(
        hidden_dim=int(policy_state["policy_head.weight"].shape[1]),
        action_dim=int(policy_state["policy_head.weight"].shape[0]),
        dropout=float(getattr(args, "policy_dropout", 0.1)),
    )
    policy.load_state_dict(policy_state)
    policy.to(device)

    model.eval()
    policy.eval()
    return model, policy, ckpt


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    merged_args = _merge_args(args, ckpt_args)
    model, policy, _ = _load_model_and_policy(args.checkpoint, merged_args, device=device)

    loader = build_rl_vqa_loader(
        merged_args,
        split=merged_args.val_split,
        batch_size=merged_args.batch_size,
        num_workers=merged_args.num_workers,
        is_train=False,
    )

    total = 0
    correct = 0
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []

    for batch in loader:
        if args.max_samples > 0 and total >= args.max_samples:
            break

        num_choices = batch["num_choices"].to(device)
        correct_idx = batch["correct_idx"].to(device)
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch["inputs"].items()
        }

        with torch.no_grad():
            encoded = model(inputs, return_hidden_states=True, pooling=merged_args.state_pooling)
            state = encoded["pooled_state"].float()
            logits, values = policy(state)
            masked_logits = _masked_logits(logits, num_choices)
            probs = torch.softmax(masked_logits, dim=-1)
            pred_idx = torch.argmax(masked_logits, dim=-1)

        batch_size = int(correct_idx.numel())
        for row in range(batch_size):
            if args.max_samples > 0 and total >= args.max_samples:
                break
            is_correct = int(pred_idx[row].item()) == int(correct_idx[row].item())
            task_name = batch["task_names"][row]
            sample_id = batch["ids"][row]
            choices = batch["choices"][row]
            num_valid = int(num_choices[row].item())
            pred_choice_idx = int(pred_idx[row].item())
            correct_choice_idx = int(correct_idx[row].item())
            valid_probs = probs[row, :num_valid].detach().cpu().tolist()
            prediction = {
                "id": sample_id,
                "task_name": task_name,
                "pred_idx": pred_choice_idx,
                "correct_idx": correct_choice_idx,
                "pred_choice": choices[pred_choice_idx],
                "correct_choice": choices[correct_choice_idx],
                "choice_probs": [float(x) for x in valid_probs],
                "value": float(values[row].item()),
                "is_correct": bool(is_correct),
            }
            predictions.append(prediction)

            total += 1
            correct += int(is_correct)
            by_task[task_name]["correct"] += int(is_correct)
            by_task[task_name]["total"] += 1

            if total <= args.print_samples:
                print(
                    f"[{total}] task={task_name} id={sample_id} pred={pred_choice_idx} gt={correct_choice_idx} "
                    f"correct={bool(is_correct)}\n"
                    f"prompt={batch['prompts'][row]}\n"
                    f"pred_choice={choices[pred_choice_idx]}\n"
                    f"gt_choice={choices[correct_choice_idx]}\n"
                )

            if args.progress_every > 0 and total % args.progress_every == 0:
                print(f"progress {total} accuracy={correct / max(total, 1):.4f}")

    accuracy = correct / max(total, 1)
    print(f"final accuracy={accuracy:.4f} ({correct}/{total})")
    print("per-task accuracy:")
    for task_name in sorted(by_task.keys()):
        stats = by_task[task_name]
        task_acc = stats["correct"] / max(stats["total"], 1)
        print(f"  {task_name}: {task_acc:.4f} ({stats['correct']}/{stats['total']})")

    if args.save_predictions:
        with open(args.save_predictions, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "per_task": by_task,
                    "predictions": predictions,
                },
                handle,
                indent=2,
            )
        print(f"saved predictions to {args.save_predictions}")


if __name__ == "__main__":
    evaluate(parse_args())
