import hashlib
import os
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


DEFAULT_PROMPT = "Describe what is happening in this egocentric video."


def _normalize_media_size(image_size):
    if isinstance(image_size, (tuple, list)):
        if len(image_size) >= 2:
            return {"height": int(image_size[0]), "width": int(image_size[1])}
        if len(image_size) == 1:
            size = int(image_size[0])
            return {"height": size, "width": size}
    if image_size is None:
        return None
    size = int(image_size)
    return {"height": size, "width": size}


def build_vlm_processor(args):
    from transformers import AutoConfig, AutoProcessor

    trust_remote_code = args.vl_backend == "internvl"
    processor = AutoProcessor.from_pretrained(args.vl_model_name, trust_remote_code=trust_remote_code)
    cfg_hf = AutoConfig.from_pretrained(args.vl_model_name, trust_remote_code=trust_remote_code)
    vision_cfg = getattr(cfg_hf, "vision_config", None)
    media_size = _normalize_media_size(getattr(vision_cfg, "image_size", None) if vision_cfg is not None else None)
    if media_size is not None:
        for proc_name in ("image_processor", "video_processor"):
            proc = getattr(processor, proc_name, None)
            if proc is None:
                continue
            if hasattr(proc, "size"):
                proc.size = dict(media_size)
            if hasattr(proc, "crop_size"):
                proc.crop_size = dict(media_size)
    return processor


def _frame_to_pil(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)

    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if np.issubdtype(frame.dtype, np.floating):
        scale = 255.0 if frame.max() <= 1.0 else 1.0
        frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    return Image.fromarray(frame).convert("RGB")


def _sample_frame_indices(num_frames: int, num_samples: int):
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_frames <= num_samples:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=num_samples, dtype=np.int64).tolist()


def _load_npy_video(video_path: str):
    array = np.load(video_path, allow_pickle=False)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim not in (3, 4):
        raise ValueError(f"Unsupported video array shape {array.shape} from {video_path}")
    return array


def _decode_npy_frames(video_array, num_frames: int):
    if video_array.ndim == 3:
        video_array = video_array[None, ...]
    total_frames = int(video_array.shape[0])
    frame_indices = _sample_frame_indices(total_frames, num_frames)
    frames = [_frame_to_pil(video_array[idx]) for idx in frame_indices]
    if len(frames) < num_frames and frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


def resolve_video_path(args, video_ref: str) -> str:
    if os.path.isfile(video_ref):
        return video_ref
    if args.video_root:
        candidate = os.path.join(args.video_root, video_ref)
        if os.path.isfile(candidate):
            return candidate
    try:
        from huggingface_hub import hf_hub_download
        try:
            from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError
        except Exception:
            EntryNotFoundError = RemoteEntryNotFoundError = tuple()
    except Exception as e:
        raise RuntimeError(
            "Could not resolve the Ego4D video file. Install `huggingface_hub` or pass --video_root pointing to the downloaded .npy files."
        ) from e
    try:
        return hf_hub_download(
            repo_id=args.dataset_name,
            filename=video_ref,
            repo_type="dataset",
            revision=args.dataset_revision,
        )
    except Exception as e:
        msg = str(e)
        if (
            "404" in msg
            or "Not Found" in msg
            or (EntryNotFoundError and isinstance(e, EntryNotFoundError))
            or (RemoteEntryNotFoundError and isinstance(e, RemoteEntryNotFoundError))
        ):
            raise RuntimeError(
                f"Video file `{video_ref}` was referenced by the dataset row but does not exist as a standalone file "
                f"in the HF dataset repo `{args.dataset_name}`. This repo appears to store the videos inside large "
                "archive parts (for example `ego4d_video.z01`, `ego4d_video.z02`, ...), so HF-only per-sample "
                "download will not work. Extract the archive locally and rerun with `--video_root /path/to/extracted_npy`."
            ) from e
        raise


def _normalize_turn_role(role: str) -> str:
    role = str(role).strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    return role


def parse_conversations(conversations) -> tuple[str, str]:
    if not conversations:
        return DEFAULT_PROMPT, ""
    user_turns = []
    assistant_turns = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = _normalize_turn_role(turn.get("from", turn.get("role", "")))
        value = str(turn.get("value", turn.get("content", ""))).strip()
        if role == "user" and value:
            user_turns.append(value)
        elif role == "assistant" and value:
            assistant_turns.append(value)
    prompt = "\n".join(user_turns).strip() or DEFAULT_PROMPT
    answer = "\n".join(assistant_turns).strip()
    return prompt, answer


def build_sft_example(processor, frames, prompt, answer, vl_backend, max_text_len):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The selected processor does not expose a tokenizer.")

    prompt_text = f"User: {prompt}\nAssistant:"
    full_text = f"{prompt_text} {answer}".strip()

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

    prompt_with_media = _add_media_token(prompt_text)
    full_with_media = _add_media_token(full_text)

    try:
        inputs = processor(
            text=full_with_media,
            videos=frames,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_len,
        )
    except TypeError:
        inputs = processor(
            text=full_with_media,
            images=frames,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_len,
        )

    prompt_ids = tokenizer(
        prompt_with_media,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_text_len,
        add_special_tokens=True,
    )["input_ids"][0]

    packed = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            value = value.squeeze(0)
        packed[key] = value

    labels = packed["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    prompt_len = min(int(prompt_ids.numel()), int(labels.numel()))
    labels[:prompt_len] = -100
    packed["labels"] = labels
    packed["prompt_text"] = prompt_text
    packed["answer_text"] = answer
    return packed


def _stable_fold(value: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class Ego4DConversationDataset(IterableDataset):
    def __init__(self, dataset, processor, args, split_name: str, is_train: bool):
        self.dataset = dataset
        self.processor = processor
        self.args = args
        self.split_name = split_name
        self.is_train = is_train

    def _iter_dataset(self) -> Iterable:
        dataset = self.dataset
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        if hasattr(dataset, "shard") and num_workers > 1:
            dataset = dataset.shard(num_shards=num_workers, index=worker_id, contiguous=False)

        is_map_style = hasattr(dataset, "__len__")
        if self.is_train and hasattr(dataset, "shuffle"):
            if is_map_style:
                dataset = dataset.shuffle(seed=self.args.seed + worker_id)
            else:
                dataset = dataset.shuffle(seed=self.args.seed + worker_id, buffer_size=self.args.shuffle_buffer)

        if hasattr(dataset, "__iter__") and not is_map_style:
            return iter(dataset)
        indices = range(worker_id, len(dataset), num_workers)
        return (dataset[idx] for idx in indices)

    def _keep_sample(self, sample_id: str) -> bool:
        val_ratio = max(0.0, min(0.5, float(self.args.val_ratio)))
        if val_ratio == 0.0:
            return self.is_train or self.split_name != self.args.val_split
        fold = _stable_fold(sample_id, self.args.seed)
        in_val = fold < val_ratio
        return (not in_val) if self.is_train else in_val

    def __iter__(self):
        sample_count = 0
        for sample in self._iter_dataset():
            sample_id = str(sample.get(self.args.id_column, sample.get("id", sample_count)))
            if not self._keep_sample(sample_id):
                continue
            if self.args.max_samples_per_split > 0 and sample_count >= self.args.max_samples_per_split:
                break

            prompt, answer = parse_conversations(sample.get(self.args.conversations_column))
            video_ref = str(sample[self.args.video_column])
            video_path = resolve_video_path(self.args, video_ref)
            frames = _decode_npy_frames(_load_npy_video(video_path), self.args.video_frames)
            packed = build_sft_example(
                processor=self.processor,
                frames=frames,
                prompt=prompt,
                answer=answer,
                vl_backend=self.args.vl_backend,
                max_text_len=self.args.vl_max_text_len,
            )
            sample_count += 1
            yield {
                "inputs": {k: v for k, v in packed.items() if k not in {"labels", "prompt_text", "answer_text"}},
                "labels": packed["labels"],
                "prompt_text": packed["prompt_text"],
                "answer_text": packed["answer_text"],
                "sample_id": sample_id,
                "video_ref": video_ref,
            }


def _stack_inputs(items):
    output = {}
    for key in items[0].keys():
        values = [item[key] for item in items]
        if torch.is_tensor(values[0]):
            if key == "pixel_values" and values[0].dim() == 4:
                output[key] = torch.cat(values, dim=0)
            else:
                output[key] = torch.stack(values, dim=0)
        else:
            output[key] = values
    return output


def collate_ego4d_batch(batch):
    return {
        "inputs": _stack_inputs([item["inputs"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "prompt_text": [item["prompt_text"] for item in batch],
        "answer_text": [item["answer_text"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "video_ref": [item["video_ref"] for item in batch],
    }


def _load_split(args, split: str):
    from datasets import load_dataset

    load_kwargs = {
        "name": args.dataset_config or None,
        "split": args.dataset_split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code_dataset:
        load_kwargs["trust_remote_code"] = True
    return load_dataset(args.dataset_name, **load_kwargs)


def ego4d_video_loader(args, split: str, batch_size: int, num_workers: int, is_train: bool):
    dataset = _load_split(args, split)
    processor = build_vlm_processor(args)
    wrapped = Ego4DConversationDataset(dataset=dataset, processor=processor, args=args, split_name=split, is_train=is_train)
    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_ego4d_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return loader
