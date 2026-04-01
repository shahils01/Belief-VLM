import csv
import io
import json
import os
from glob import glob

import numpy as np
import torch
from PIL import Image

from data_loading import _frame_to_pil, _parse_timecode_seconds, decode_mp4_frames


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(normalize_text(item) for item in value if item is not None).strip()
    return str(value).strip()


def get_first(record, keys, default=None):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def expand_annotation_paths(path_spec: str):
    if not path_spec:
        return []
    raw_parts = [part.strip() for part in path_spec.split(",") if part.strip()]
    if not raw_parts:
        raw_parts = [path_spec]
    expanded = []
    for part in raw_parts:
        if os.path.isdir(part):
            expanded.extend(sorted(glob(os.path.join(part, "*.json"))))
            expanded.extend(sorted(glob(os.path.join(part, "*.jsonl"))))
            expanded.extend(sorted(glob(os.path.join(part, "*.csv"))))
        else:
            expanded.append(part)
    deduped = []
    seen = set()
    for path in expanded:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def load_local_records(path_spec: str):
    records = []
    for path in expand_annotation_paths(path_spec):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                obj = json.load(handle)
            if isinstance(obj, list):
                records.extend(obj)
            elif isinstance(obj, dict):
                if obj and all(isinstance(v, dict) for v in obj.values()):
                    for key, value in obj.items():
                        item = dict(value)
                        item.setdefault("id", key)
                        records.append(item)
                else:
                    value = next((obj[k] for k in ("data", "samples", "items", "annotations") if isinstance(obj.get(k), list)), None)
                    if value is None:
                        raise RuntimeError(f"Unsupported JSON manifest format: {path}")
                    records.extend(value)
            else:
                raise RuntimeError(f"Unsupported JSON manifest format: {path}")
        elif path.endswith(".csv"):
            with open(path, "r", encoding="utf-8", newline="") as handle:
                records.extend(list(csv.DictReader(handle)))
        else:
            raise RuntimeError(f"Unsupported local manifest format: {path}")
    return records


def sample_indices(num_frames: int, num_samples: int):
    if num_frames <= 0:
        return []
    if num_frames <= num_samples:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=num_samples, dtype=np.int64).tolist()


def _decode_image(media):
    if isinstance(media, Image.Image):
        return media.convert("RGB")
    if isinstance(media, dict):
        if media.get("bytes") is not None:
            return Image.open(io.BytesIO(media["bytes"])).convert("RGB")
        if media.get("path"):
            return Image.open(media["path"]).convert("RGB")
        if media.get("image") is not None:
            return _decode_image(media["image"])
    if isinstance(media, str) and os.path.isfile(media):
        return Image.open(media).convert("RGB")
    return _frame_to_pil(media)


def _decode_video_from_sequence(media, num_frames: int):
    if isinstance(media, dict) and "frames" in media:
        media = media["frames"]
    frames = [_frame_to_pil(frame) for frame in media]
    if not frames:
        raise RuntimeError("Video media sequence is empty.")
    idx = sample_indices(len(frames), num_frames)
    sampled = [frames[i] for i in idx]
    if len(sampled) < num_frames:
        sampled.extend([sampled[-1]] * (num_frames - len(sampled)))
    return sampled


def decode_media(media_type: str, media, video_frames: int, metadata=None):
    metadata = metadata or {}
    if media_type == "image":
        if isinstance(media, (list, tuple)):
            decoded = [_decode_image(item) for item in media if item is not None]
            if not decoded:
                raise RuntimeError("Image media sequence is empty.")
            return decoded
        return [_decode_image(media)]
    if media_type != "video":
        raise RuntimeError(f"Unsupported media_type={media_type}")

    if isinstance(media, (list, tuple)):
        return _decode_video_from_sequence(media, video_frames)
    if isinstance(media, dict):
        if "frames" in media or "images" in media:
            return _decode_video_from_sequence(media.get("frames") or media.get("images"), video_frames)
        path = media.get("path")
        if path:
            start = _parse_timecode_seconds(metadata.get("start_time_sec", metadata.get("start_time")))
            end = _parse_timecode_seconds(metadata.get("end_time_sec", metadata.get("end_time")))
            return decode_mp4_frames(path, video_frames, start_time_sec=start, end_time_sec=end)
    if isinstance(media, str):
        start = _parse_timecode_seconds(metadata.get("start_time_sec", metadata.get("start_time")))
        end = _parse_timecode_seconds(metadata.get("end_time_sec", metadata.get("end_time")))
        return decode_mp4_frames(media, video_frames, start_time_sec=start, end_time_sec=end)
    raise RuntimeError("Unsupported video media representation.")


def _apply_media_template(processor, prompt, answer, media_type, add_answer, vl_backend):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The selected processor does not expose a tokenizer.")

    content_type = "video" if media_type == "video" else "image"
    prompt_text = f"User: {prompt}\nAssistant:"
    full_text = f"{prompt_text} {answer}".strip()

    def _add_media_token(text: str) -> str:
        vocab = tokenizer.get_vocab()
        if any(token in text for token in ("<video>", "<image>", "<img>")):
            return text
        if vl_backend == "internvl":
            preferred = "<video>" if media_type == "video" else "<image>"
            if preferred in vocab:
                return f"{preferred}\n{text}"
            for token in ("<video>", "<image>", "<img>"):
                if token in vocab:
                    return f"{token}\n{text}"
        return text

    if vl_backend == "internvl" and hasattr(processor, "apply_chat_template"):
        user_message = {
            "role": "user",
            "content": [
                {"type": content_type},
                {"type": "text", "text": prompt},
            ],
        }
        prompt_with_media = processor.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        if add_answer:
            full_with_media = processor.apply_chat_template(
                [user_message, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            full_with_media = prompt_with_media
    else:
        prompt_with_media = _add_media_token(prompt_text)
        full_with_media = _add_media_token(full_text if add_answer else prompt_text)
    return prompt_text, prompt_with_media, full_with_media


def build_multimodal_sft_example(processor, media_type, media, prompt, answer, vl_backend, max_text_len):
    tokenizer = getattr(processor, "tokenizer", None)
    prompt_text, prompt_with_media, full_with_media = _apply_media_template(
        processor=processor,
        prompt=prompt,
        answer=answer,
        media_type=media_type,
        add_answer=True,
        vl_backend=vl_backend,
    )

    kwargs = {
        "text": [full_with_media],
        "return_tensors": "pt",
        "padding": "longest",
        "truncation": False,
    }
    if media_type == "video":
        kwargs["videos"] = [media]
    else:
        kwargs["images"] = media
    inputs = processor(**kwargs)

    prompt_ids = tokenizer(
        prompt_with_media,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"][0]
    packed = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            value = value.squeeze(0)
        packed[key] = value
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels = packed["input_ids"].clone()
    labels[labels == pad_token_id] = -100
    prompt_len = min(int(prompt_ids.numel()), int(labels.numel()))
    labels[:prompt_len] = -100
    packed["labels"] = labels
    packed["prompt_text"] = prompt_text
    packed["answer_text"] = answer
    return packed


def build_multimodal_prompt_only_example(processor, media_type, media, prompt, vl_backend, max_text_len):
    prompt_text, _, prompt_with_media = _apply_media_template(
        processor=processor,
        prompt=prompt,
        answer="",
        media_type=media_type,
        add_answer=False,
        vl_backend=vl_backend,
    )
    kwargs = {
        "text": [prompt_with_media],
        "return_tensors": "pt",
        "padding": "longest",
        "truncation": False,
    }
    if media_type == "video":
        kwargs["videos"] = [media]
    else:
        kwargs["images"] = media
    inputs = processor(**kwargs)
    packed = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            value = value.squeeze(0)
        packed[key] = value
    packed["prompt_text"] = prompt_text
    return packed


def stack_inputs(items):
    output = {}
    for key in items[0].keys():
        values = [item[key] for item in items]
        if torch.is_tensor(values[0]):
            if key == "pixel_values" and values[0].dim() == 4:
                output[key] = torch.cat(values, dim=0)
            elif key in {"input_ids", "attention_mask"}:
                max_len = max(int(v.shape[0]) for v in values)
                padded = [torch.nn.functional.pad(v, (0, max_len - int(v.shape[0])), value=0) for v in values]
                output[key] = torch.stack(padded, dim=0)
            else:
                output[key] = torch.stack(values, dim=0)
        else:
            output[key] = values
    return output


def collate_sft_batch(batch):
    max_len = max(int(item["labels"].shape[0]) for item in batch)
    labels = [
        torch.nn.functional.pad(item["labels"], (0, max_len - int(item["labels"].shape[0])), value=-100)
        for item in batch
    ]
    return {
        "ids": [item["id"] for item in batch],
        "task_names": [item["task_name"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "choices": [item.get("choices") for item in batch],
        "correct_idx": [item.get("correct_idx") for item in batch],
        "media_type": [item["media_type"] for item in batch],
        "inputs": stack_inputs([item["inputs"] for item in batch]),
        "labels": torch.stack(labels, dim=0),
    }
