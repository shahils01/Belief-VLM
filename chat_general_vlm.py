import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

from general_data_utils import build_multimodal_prompt_only_example, decode_media
from train import _apply_peft, _configure_memory_optimizations, _resolve_vl_model_preset, build_model
from vector_memory import OnlineVectorMemory


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a generalized Belief-VLM checkpoint using an image or video.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--task_name", type=str, default="chat")
    parser.add_argument("--sample_id", type=str, default="interactive")
    parser.add_argument("--video_frames", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
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
    parser.add_argument("--use_memory_retrieval", action="store_true")
    parser.add_argument("--memory_top_k", type=int, default=None)
    parser.add_argument("--memory_index_backend", type=str, default=None, choices=["auto", "faiss", "numpy"])
    parser.add_argument("--memory_same_task_first", action="store_true")
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


def _load_model_and_memory(checkpoint_path, args, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    merged_args = _merge_args(args, ckpt_args)
    merged_args.quantization_config = _build_quant_config(merged_args)
    _resolve_vl_model_preset(merged_args)
    model = build_model(merged_args, device=device)
    model = _apply_peft(model, merged_args)
    _configure_memory_optimizations(model, merged_args)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    memory = None
    memory_fusion = None
    if merged_args.use_memory_retrieval and isinstance(ckpt, dict) and ckpt.get("vector_memory") is not None:
        memory = OnlineVectorMemory.from_state_dict(ckpt["vector_memory"], merged_args)
    if merged_args.use_memory_retrieval and isinstance(ckpt, dict) and ckpt.get("memory_fusion") is not None:
        dim = None
        if memory is not None:
            dim = memory.dim
        else:
            try:
                dim = int(ckpt["memory_fusion"]["context_proj.weight"].shape[0])
            except Exception:
                pass
        if dim is None:
            raise RuntimeError("Could not determine memory fusion hidden size from checkpoint.")
        memory_fusion = GatedMemoryFusion(dim).to(device)
        memory_fusion.load_state_dict(ckpt["memory_fusion"], strict=False)
        memory_fusion.eval()
    return model, memory, memory_fusion, merged_args


def _decode_generated_text(model, generated_ids, prompt_input_ids):
    tokenizer = model.backbone.tokenizer
    prompt_len = int(prompt_input_ids.shape[-1])
    gen_tokens = generated_ids[0, prompt_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    if bool(args.image_path) == bool(args.video_path):
        raise RuntimeError("Provide exactly one of --image_path or --video_path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, memory, memory_fusion, merged_args = _load_model_and_memory(args.checkpoint, args, device)
    media_type = "image" if args.image_path else "video"
    media_path = args.image_path or args.video_path
    media = decode_media(media_type, media_path, video_frames=merged_args.video_frames, metadata={})

    packed = build_multimodal_prompt_only_example(
        processor=model.backbone.processor,
        media_type=media_type,
        media=media,
        prompt=args.question,
        vl_backend=merged_args.vl_backend,
        max_text_len=merged_args.vl_max_text_len,
    )
    inputs = {
        k: v.unsqueeze(0).to(device) if torch.is_tensor(v) and v.dim() > 0 else v
        for k, v in packed.items()
        if torch.is_tensor(v)
    }

    hook_handle = None
    if memory is not None and memory_fusion is not None and len(memory) > 0:
        with torch.no_grad():
            query_state = model.encode_inputs(
                inputs,
                pooling="last",
                layer_idx=merged_args.memory_layer_idx,
            )["pooled_state"].float()
        retrieved = memory.retrieve_aggregates(
            query_state,
            [args.sample_id],
            [args.task_name],
            top_k=merged_args.memory_top_k,
        )
        memory_context = torch.from_numpy(retrieved["context"]).to(device, dtype=query_state.dtype)
        memory_answer = torch.from_numpy(retrieved["answer"]).to(device, dtype=query_state.dtype)
        memory_reward = torch.from_numpy(retrieved["reward"]).to(device, dtype=query_state.dtype)
        fused_memory = memory_fusion(query_state, memory_context, memory_answer, memory_reward)
        inject_layer_idx = max(int(merged_args.memory_layer_idx) + int(merged_args.memory_inject_offset), 0)
        hook_handle = model.inject_pooled_memory_context(fused_memory, inject_layer_idx)

    try:
        with torch.no_grad():
            generated = model.generate(inputs, max_new_tokens=args.max_new_tokens)
        answer = _decode_generated_text(model, generated, inputs["input_ids"])
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    print(answer)


if __name__ == "__main__":
    main()
