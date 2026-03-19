import argparse
import json
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from data_loading import (
    _get_first,
    _load_records,
    _normalize_text,
    _resolve_hd_epic_clip_window,
    _resolve_hd_epic_video_path,
    build_sft_example,
    collate_sft_batch,
    decode_mp4_frames,
)
from train import _apply_peft, _configure_memory_optimizations, _resolve_vl_model_preset, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HD-EPIC VQA multiple-choice accuracy.")
    parser.add_argument("--checkpoint", type=str, default="")
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
    parser.add_argument("--use_future_predictor", action="store_true")
    parser.add_argument("--future_predictor_checkpoint", type=str, default="")
    parser.add_argument("--future_frames", type=int, default=None)
    parser.add_argument("--use_belief_network", action="store_true")
    parser.add_argument("--belief_network_checkpoint", type=str, default="")
    parser.add_argument("--freeze_vl", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--max_samples", type=int, default=100)
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
        "future_frames",
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
    if cli_args.use_future_predictor:
        merged["use_future_predictor"] = True
        merged["future_predictor_checkpoint"] = cli_args.future_predictor_checkpoint
    if cli_args.use_belief_network:
        merged["use_belief_network"] = True
        merged["belief_network_checkpoint"] = cli_args.belief_network_checkpoint

    merged.setdefault("vl_backend", "internvl")
    merged.setdefault("vl_model_preset", "internvl3_5_1b")
    merged.setdefault("vl_model_name", "OpenGVLab/InternVL3_5-1B-HF")
    merged.setdefault("vl_dtype", "bfloat16")
    merged.setdefault("video_frames", 8)
    merged.setdefault("vl_max_text_len", 256)
    merged.setdefault("freeze_vl", False)
    merged.setdefault("peft", "none")
    merged.setdefault("use_future_predictor", False)
    merged.setdefault("future_predictor_checkpoint", "")
    merged.setdefault("future_frames", 0)
    merged.setdefault("use_belief_network", False)
    merged.setdefault("belief_network_checkpoint", "")
    merged.setdefault("lora_r", 16)
    merged.setdefault("lora_alpha", 32)
    merged.setdefault("lora_dropout", 0.05)
    merged.setdefault("lora_target_modules", "")
    merged.setdefault("lora_bias", "none")
    merged.setdefault("gradient_checkpointing", False)
    merged.setdefault("disable_vl_cache", False)
    merged.setdefault("allow_tf32", False)
    merged["future_predictor_bundle"] = getattr(cli_args, "future_predictor_bundle", None)
    merged["belief_network_bundle"] = getattr(cli_args, "belief_network_bundle", None)
    return SimpleNamespace(**merged)


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


def _restore_bundled_belief_network_args(args, ckpt):
    if not isinstance(ckpt, dict):
        return
    bundle = ckpt.get("belief_network")
    if bundle is None:
        return
    args.use_belief_network = True
    if not getattr(args, "belief_network_checkpoint", ""):
        args.belief_network_bundle = bundle


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


def _load_model(checkpoint_path, args, device):
    args.quantization_config = _build_quant_config(args)
    _resolve_vl_model_preset(args)
    model = build_model(args, device=device)
    model = _apply_peft(model, args)
    _configure_memory_optimizations(model, args)
    ckpt = None
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            incompatible = model.load_state_dict(state_dict, strict=False)
            print(
                "Non-strict checkpoint load complete: "
                f"missing_keys={len(getattr(incompatible, 'missing_keys', []))} "
                f"unexpected_keys={len(getattr(incompatible, 'unexpected_keys', []))}"
            )
    model.eval()
    return model, ckpt


def _build_eval_prompt(record, args):
    prompt = _get_first(record, [args.question_column, "question", "prompt", "instruction", "query"])
    prompt = _normalize_text(prompt)
    choices = _get_first(record, [args.options_column, "options", "choices", "answer_options"])
    if not isinstance(choices, (list, tuple)):
        raise RuntimeError(
            "HD-EPIC VQA evaluation expects multiple-choice records with a `choices` field."
        )
    choices = [_normalize_text(choice) for choice in choices]
    prompt_with_options = prompt
    options_text = _normalize_text(choices)
    if options_text:
        prompt_with_options = f"{prompt_with_options}\nOptions:\n{options_text}"
    return prompt_with_options, choices


def _resolve_correct_choice_index(record, num_choices):
    correct_idx = _get_first(record, ["correct_idx", "answer_idx", "label_idx"])
    if correct_idx in (None, ""):
        raise RuntimeError("Record is missing `correct_idx` for multiple-choice evaluation.")
    idx = int(correct_idx)
    if 0 <= idx < num_choices:
        return idx
    if 1 <= idx <= num_choices:
        return idx - 1
    raise RuntimeError(f"Choice index {idx} is out of range for {num_choices} choices.")


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


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else {}
    _restore_bundled_future_predictor_args(args, ckpt)
    _restore_bundled_belief_network_args(args, ckpt)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    merged_args = _merge_args(args, ckpt_args)
    if merged_args.use_future_predictor and merged_args.use_belief_network:
        raise RuntimeError("Use only one of --use_future_predictor or --use_belief_network.")
    if merged_args.use_future_predictor:
        has_bundled_predictor = getattr(merged_args, "future_predictor_bundle", None) is not None
        if not merged_args.future_predictor_checkpoint and not has_bundled_predictor:
            raise RuntimeError("--use_future_predictor requires --future_predictor_checkpoint.")
        if int(merged_args.future_frames) <= 0:
            raise RuntimeError("--use_future_predictor requires --future_frames > 0.")
    if merged_args.use_belief_network:
        has_bundled_belief = getattr(merged_args, "belief_network_bundle", None) is not None
        if not merged_args.belief_network_checkpoint and not has_bundled_belief:
            raise RuntimeError("--use_belief_network requires --belief_network_checkpoint.")
    model, _ = _load_model(args.checkpoint, merged_args, device=device)
    processor = model.backbone.processor
    records = _load_records(merged_args)

    total = 0
    correct = 0
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []

    for record in records:
        if args.max_samples > 0 and total >= args.max_samples:
            break

        prompt, choices = _build_eval_prompt(record, merged_args)
        correct_idx = _resolve_correct_choice_index(record, len(choices))
        video_path = _resolve_hd_epic_video_path(merged_args, record)
        start_time_sec, end_time_sec = _resolve_hd_epic_clip_window(record)
        frames = decode_mp4_frames(
            video_path,
            merged_args.video_frames,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
        )

        choice_examples = []
        for choice_text in choices:
            packed = build_sft_example(
                processor=processor,
                frames=frames,
                prompt=prompt,
                answer=choice_text,
                vl_backend=merged_args.vl_backend,
                max_text_len=merged_args.vl_max_text_len,
            )
            choice_examples.append(
                {
                    "inputs": {k: v for k, v in packed.items() if k not in {"labels", "prompt_text", "answer_text"}},
                    "labels": packed["labels"],
                }
            )

        batch = collate_sft_batch(choice_examples)
        labels = batch["labels"].to(device)
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch["inputs"].items()
        }

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            choice_losses = _sequence_nll(outputs["logits"], labels)

        pred_idx = int(torch.argmin(choice_losses).item())
        is_correct = pred_idx == correct_idx
        task_name = record.get("task_name", "unknown")
        sample_id = str(record.get("id", total))

        total += 1
        correct += int(is_correct)
        by_task[task_name]["correct"] += int(is_correct)
        by_task[task_name]["total"] += 1

        prediction = {
            "id": sample_id,
            "task_name": task_name,
            "video_id": _get_first(record, [merged_args.video_id_column, "video_id"]) or "",
            "pred_idx": pred_idx,
            "correct_idx": correct_idx,
            "pred_choice": choices[pred_idx],
            "correct_choice": choices[correct_idx],
            "choice_losses": [float(x) for x in choice_losses.tolist()],
            "is_correct": is_correct,
        }
        predictions.append(prediction)

        if total <= args.print_samples:
            print(
                f"[{total}] task={task_name} id={sample_id} pred={pred_idx} gt={correct_idx} "
                f"correct={is_correct}\n"
                f"question={prompt}\n"
                f"pred_choice={choices[pred_idx]}\n"
                f"gt_choice={choices[correct_idx]}\n"
            )
        if args.progress_every > 0 and total % args.progress_every == 0:
            running_accuracy = correct / max(total, 1)
            print(
                f"progress samples={total} accuracy={running_accuracy:.4f} "
                f"({correct}/{total}) = {running_accuracy * 100.0:.2f}%"
            )

    accuracy = correct / max(total, 1)
    print(f"overall_accuracy={accuracy:.4f} ({correct}/{total}) = {accuracy * 100.0:.2f}%")
    for task_name in sorted(by_task.keys()):
        task_correct = by_task[task_name]["correct"]
        task_total = by_task[task_name]["total"]
        task_accuracy = task_correct / max(task_total, 1)
        print(
            f"task={task_name} accuracy={task_accuracy:.4f} "
            f"({task_correct}/{task_total}) = {task_accuracy * 100.0:.2f}%"
        )

    if args.save_predictions:
        with open(args.save_predictions, "w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
        print(f"saved_predictions={args.save_predictions}")


if __name__ == "__main__":
    evaluate(parse_args())
