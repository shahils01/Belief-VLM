import argparse
from collections import defaultdict
from types import SimpleNamespace

import torch

from belief_db import BeliefVectorDB
from data_loading import _load_records, build_rl_vqa_loader
from train import _apply_peft, _configure_memory_optimizations, _resolve_vl_model_preset, build_model
from train_db_prior import (
    PPOAnswerPolicy,
    PriorSelectorPolicy,
    _build_quant_config,
    _combine_inputs_with_prompts,
    _compose_prior_prompt,
    _extract_state,
    _prior_selector_forward,
    _score_answers_with_vlm,
    _answer_forward,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval-augmented prior models.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--metadata_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--print_samples", type=int, default=10)
    return parser.parse_args()


def _merge_args(cli_args, ckpt_args):
    merged = dict(ckpt_args)
    merged.update(
        {
            "annotation_path": cli_args.annotation_path,
            "video_root": cli_args.video_root,
            "video_extension": cli_args.video_extension,
            "metadata_root": cli_args.metadata_root,
            "batch_size": cli_args.batch_size,
            "num_workers": cli_args.num_workers,
            "max_samples_per_split": cli_args.max_samples_per_split,
        }
    )
    if cli_args.val_ratio is not None:
        merged["val_ratio"] = cli_args.val_ratio
    merged.setdefault("dataset_type", "hd_epic_local")
    merged.setdefault("train_split", "train")
    merged.setdefault("val_split", "validation")
    merged.setdefault("val_ratio", 0.1)
    merged.setdefault("use_rl_prior_selector", False)
    merged.setdefault("use_rl_answer_head", False)
    merged.setdefault("prior_top_k", 4)
    merged.setdefault("prior_prompt_prefix", "Belief prior:")
    merged.setdefault("retrieval_embedder_model", "")
    merged.setdefault("retrieval_hash_dim", 512)
    merged.setdefault("vl_backend", "internvl")
    merged.setdefault("vl_model_preset", "internvl3_5_1b")
    merged.setdefault("vl_model_name", "OpenGVLab/InternVL3_5-1B-HF")
    merged.setdefault("vl_dtype", "bfloat16")
    merged.setdefault("vl_max_text_len", 256)
    merged.setdefault("peft", "none")
    merged.setdefault("freeze_vl", False)
    merged.setdefault("gradient_checkpointing", False)
    merged.setdefault("disable_vl_cache", False)
    merged.setdefault("allow_tf32", False)
    merged.setdefault("state_pooling", "last")
    merged.setdefault("max_choice_options", 8)
    merged.setdefault("policy_dropout", 0.2)
    merged.setdefault("seed", 42)
    return SimpleNamespace(**merged)


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    merged_args = _merge_args(args, ckpt.get("args", {}))
    merged_args.quantization_config = _build_quant_config(merged_args)
    _resolve_vl_model_preset(merged_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = _load_records(merged_args)
    belief_db = BeliefVectorDB.from_records(records, merged_args, split_name=merged_args.train_split)
    loader = build_rl_vqa_loader(merged_args, merged_args.val_split, merged_args.batch_size, merged_args.num_workers, is_train=False)

    model = build_model(merged_args, device=device)
    model = _apply_peft(model, merged_args)
    _configure_memory_optimizations(model, merged_args)
    model.load_state_dict(ckpt["model"], strict=False)
    processor = model.backbone.processor
    model.eval()

    selector_policy = None
    if merged_args.use_rl_prior_selector and "selector_policy" in ckpt:
        probe_batch = next(iter(build_rl_vqa_loader(merged_args, merged_args.train_split, batch_size=1, num_workers=0, is_train=True)))
        with torch.no_grad():
            hidden_dim = int(model(probe_batch["inputs"], return_hidden_states=True, pooling=merged_args.state_pooling)["pooled_state"].shape[-1])
        selector_policy = PriorSelectorPolicy(hidden_dim, belief_db.embedder.dim, merged_args.policy_dropout)
        selector_policy.load_state_dict(ckpt["selector_policy"])
        selector_policy.to(device).eval()

    answer_policy = None
    if merged_args.use_rl_answer_head and "answer_policy" in ckpt:
        hidden_dim = int(ckpt["answer_policy"]["policy_head.weight"].shape[1])
        action_dim = int(ckpt["answer_policy"]["policy_head.weight"].shape[0])
        answer_policy = PPOAnswerPolicy(hidden_dim, action_dim, merged_args.policy_dropout)
        answer_policy.load_state_dict(ckpt["answer_policy"])
        answer_policy.to(device).eval()

    total = 0
    correct = 0
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})

    for batch in loader:
        query_state = _extract_state(model, batch["inputs"], merged_args)
        retrieval = belief_db.retrieve(batch["task_names"], batch["prompts"], merged_args.prior_top_k, exclude_ids=batch["ids"])
        candidate_embeddings = retrieval.candidate_embeddings.to(device)
        candidate_counts = retrieval.candidate_counts.to(device)

        if selector_policy is not None:
            with torch.no_grad():
                selector_dist, _ = _prior_selector_forward(selector_policy, query_state, candidate_embeddings, candidate_counts)
                selector_actions = torch.argmax(selector_dist.logits, dim=-1)
        else:
            selector_actions = torch.zeros(query_state.shape[0], dtype=torch.long, device=device)

        chosen_priors = []
        for row, action in enumerate(selector_actions.tolist()):
            valid_idx = min(int(action), len(retrieval.candidate_texts[row]) - 1)
            chosen_priors.append(retrieval.candidate_texts[row][valid_idx])
        prior_prompts = [
            _compose_prior_prompt(prompt, prior, merged_args.prior_prompt_prefix)
            for prompt, prior in zip(batch["prompts"], chosen_priors)
        ]
        prior_inputs = _combine_inputs_with_prompts(batch["inputs"], prior_prompts, processor, merged_args)
        if answer_policy is not None:
            with torch.no_grad():
                prior_state = _extract_state(model, prior_inputs, merged_args)
                answer_dist, _ = _answer_forward(answer_policy, prior_state, batch["num_choices"].to(device))
                pred_idx = torch.argmax(answer_dist.logits, dim=-1)
        else:
            pred_idx = _score_answers_with_vlm(model, processor, prior_inputs, prior_prompts, batch["choices"], merged_args)

        gt = batch["correct_idx"].to(device)
        for row in range(int(gt.numel())):
            is_correct = int(pred_idx[row].item()) == int(gt[row].item())
            total += 1
            correct += int(is_correct)
            task_name = batch["task_names"][row]
            by_task[task_name]["correct"] += int(is_correct)
            by_task[task_name]["total"] += 1
            if total <= args.print_samples:
                print(
                    f"[{total}] task={task_name} pred={int(pred_idx[row].item())} gt={int(gt[row].item())} "
                    f"correct={bool(is_correct)} prior={chosen_priors[row]}"
                )

    print(f"accuracy={correct / max(total, 1):.4f} total={total}")
    for task_name in sorted(by_task):
        stats = by_task[task_name]
        print(f"{task_name}: {stats['correct'] / max(stats['total'], 1):.4f} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
