import csv
import hashlib
import json
import os
from typing import Iterable

import cv2
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


def _sample_frame_indices(num_frames: int, num_samples: int):
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_frames <= num_samples:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=num_samples, dtype=np.int64).tolist()


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


def decode_mp4_frames(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    wanted = set(_sample_frame_indices(frame_count, num_frames))
    frames = []
    cur = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cur += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to decode any frames from {video_path}")
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


def build_sft_example(processor, frames, prompt, answer, vl_backend, max_text_len):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The selected processor does not expose a tokenizer.")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

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

    prompt_with_media = _add_media_token(prompt_text)
    full_with_media = _add_media_token(full_text)

    prompt_ids = tokenizer(
        prompt_with_media,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_text_len,
        add_special_tokens=True,
    )["input_ids"][0]

    text_inputs = tokenizer(
        [full_with_media],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        add_special_tokens=True,
    )

    try:
        processor_kwargs = {
            "videos": [frames],
            "return_tensors": "pt",
        }
        processor_kwargs["text"] = [""]
        processor_kwargs["padding"] = "longest"
        processor_kwargs["truncation"] = False
        inputs = processor(**processor_kwargs)
    except TypeError:
        processor_kwargs = {
            "images": [frames],
            "return_tensors": "pt",
        }
        processor_kwargs["text"] = [""]
        processor_kwargs["padding"] = "longest"
        processor_kwargs["truncation"] = False
        inputs = processor(**processor_kwargs)

    packed = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            value = value.squeeze(0)
        packed[key] = value

    packed["input_ids"] = text_inputs["input_ids"].squeeze(0)
    packed["attention_mask"] = text_inputs["attention_mask"].squeeze(0)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    labels = packed["input_ids"].clone()
    labels[labels == pad_token_id] = -100
    prompt_len = min(int(prompt_ids.numel()), int(labels.numel()))
    labels[:prompt_len] = -100
    packed["labels"] = labels
    packed["prompt_text"] = prompt_text
    packed["answer_text"] = answer
    return packed


def _stable_fold(value: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _load_records(args):
    path = args.annotation_path
    if not path:
        raise RuntimeError(
            "HD-EPIC training requires --annotation_path pointing to a local json/jsonl/csv supervision file."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")

    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("data", "samples", "annotations", "items"):
                value = obj.get(key)
                if isinstance(value, list):
                    return value
        raise RuntimeError("JSON annotation file must be a list or contain one of: data/samples/annotations/items.")

    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise RuntimeError("Unsupported annotation file format. Use .json, .jsonl, or .csv")


def _discover_hd_epic_records(args):
    if not args.video_root or not args.metadata_root:
        raise RuntimeError(
            "HD-EPIC auto-discovery requires both --video_root and --metadata_root when --annotation_path is not set."
        )

    records = []
    for root, _, files in os.walk(args.video_root):
        for filename in files:
            if not filename.lower().endswith(f".{args.video_extension.lower()}"):
                continue
            video_path = os.path.join(root, filename)
            video_id = os.path.splitext(filename)[0]
            participant = video_id.split("-", 1)[0]
            metadata_path = os.path.join(args.metadata_root, participant, video_id, "framewise_info.jsonl")
            if not os.path.isfile(metadata_path):
                continue
            records.append(
                {
                    "id": video_id,
                    "video_id": video_id,
                    "participant_id": participant,
                    "video_path": video_path,
                    "metadata_path": metadata_path,
                }
            )

    if not records:
        raise RuntimeError("No HD-EPIC video/metadata pairs were found. Check --video_root and --metadata_root.")
    return sorted(records, key=lambda item: item["video_id"])


def _normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_normalize_text(item) for item in value if item is not None).strip()
    return str(value).strip()


def _get_first(record, keys):
    for key in keys:
        if key and key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _resolve_hd_epic_video_path(args, record):
    direct = _get_first(record, [args.video_path_column, "video_path", "clip_path", "path"])
    if direct:
        direct = str(direct)
        if os.path.isfile(direct):
            return direct
        if args.video_root:
            candidate = os.path.join(args.video_root, direct)
            if os.path.isfile(candidate):
                return candidate

    video_id = _get_first(record, [args.video_id_column, "video_id", "clip_id", "uid", "video_uid"])
    if not video_id:
        raise RuntimeError(
            f"Could not resolve a video path for record. Set --video_id_column/--video_path_column. Keys: {sorted(record.keys())}"
        )
    video_id = str(video_id)
    participant = _get_first(record, [args.participant_column, "participant_id", "participant", "user_id"])
    if not participant:
        participant = video_id.split("-", 1)[0]

    candidate = os.path.join(args.video_root, str(participant), f"{video_id}.{args.video_extension}")
    if os.path.isfile(candidate):
        return candidate

    alt = os.path.join(args.video_root, f"{video_id}.{args.video_extension}")
    if os.path.isfile(alt):
        return alt

    raise FileNotFoundError(
        f"Could not find video for video_id={video_id}. Tried {candidate} and {alt}. "
        "Check --video_root, --video_extension, and the annotation columns."
    )


def _build_prompt_answer(args, record):
    prompt = _get_first(record, [args.question_column, "question", "prompt", "instruction", "query"])
    answer = _get_first(record, [args.answer_column, "answer", "response", "label", "caption", "narration"])

    prompt = _normalize_text(prompt) or DEFAULT_PROMPT
    answer = _normalize_text(answer)
    options = _get_first(record, [args.options_column, "options", "choices", "answer_options"])
    if options:
        options_text = _normalize_text(options)
        if options_text:
            prompt = f"{prompt}\nOptions:\n{options_text}"

    if not answer:
        raise RuntimeError(
            "The selected annotation record does not contain a target answer. "
            "Set --answer_column to a valid field in your HD-EPIC annotation file."
        )
    return prompt, answer


def _round_list(values, ndigits=3):
    return [round(float(v), ndigits) for v in values]


def _extract_translation(transform):
    if not isinstance(transform, list) or len(transform) < 3:
        return None
    try:
        return [float(transform[0][3]), float(transform[1][3]), float(transform[2][3])]
    except Exception:
        return None


def _build_prompt_answer_from_metadata(record):
    metadata_path = record.get("metadata_path")
    if not metadata_path or not os.path.isfile(metadata_path):
        raise RuntimeError(f"Missing metadata file for record: {record.get('video_id', 'unknown')}")

    entries = []
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        raise RuntimeError(f"Metadata file has no entries: {metadata_path}")

    sample_idxs = _sample_frame_indices(len(entries), min(3, len(entries)))
    sampled = [entries[idx] for idx in sample_idxs]

    def _frame_summary(entry, tag):
        parts = [f"{tag}_frame={entry.get('frame_index', -1)}"]
        gaze = entry.get("gaze_centre_in_pixels")
        parts.append(f"{tag}_gaze={_round_list(gaze, 1)}" if gaze is not None else f"{tag}_gaze=none")
        translation = _extract_translation(entry.get("T_world_device"))
        parts.append(
            f"{tag}_position={_round_list(translation, 3)}" if translation is not None else f"{tag}_position=none"
        )
        gaze_dir = entry.get("gaze_direction_in_world")
        parts.append(
            f"{tag}_gaze_dir={_round_list(gaze_dir, 3)}" if gaze_dir is not None else f"{tag}_gaze_dir=none"
        )
        return "; ".join(parts)

    prompt = "Summarize the wearer motion and gaze at the start, middle, and end of this egocentric video."
    answer = "; ".join(
        [
            f"id={record['video_id']}",
            f"frames={len(entries)}",
            _frame_summary(sampled[0], "start"),
            _frame_summary(sampled[len(sampled) // 2], "middle"),
            _frame_summary(sampled[-1], "end"),
        ]
    )
    return prompt, answer


class LocalHD_EPICDataset(IterableDataset):
    def __init__(self, records, processor, args, split_name: str, is_train: bool):
        self.records = records
        self.processor = processor
        self.args = args
        self.split_name = split_name
        self.is_train = is_train

    def _iter_records(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        indices = list(range(worker_id, len(self.records), num_workers))
        if self.is_train and indices:
            generator = np.random.default_rng(self.args.seed + worker_id)
            generator.shuffle(indices)
        for idx in indices:
            yield self.records[idx]

    def _keep_sample(self, sample_id: str) -> bool:
        val_ratio = max(0.0, min(0.5, float(self.args.val_ratio)))
        if val_ratio == 0.0:
            return self.is_train or self.split_name != self.args.val_split
        fold = _stable_fold(sample_id, self.args.seed)
        in_val = fold < val_ratio
        return (not in_val) if self.is_train else in_val

    def __iter__(self):
        sample_count = 0
        for record in self._iter_records():
            sample_id = str(
                _get_first(record, [self.args.id_column, "id", "sample_id", "uid", "video_id"]) or sample_count
            )
            if not self._keep_sample(sample_id):
                continue
            if self.args.max_samples_per_split > 0 and sample_count >= self.args.max_samples_per_split:
                break

            video_path = _resolve_hd_epic_video_path(self.args, record)
            if self.args.annotation_path:
                prompt, answer = _build_prompt_answer(self.args, record)
            else:
                prompt, answer = _build_prompt_answer_from_metadata(record)
            frames = decode_mp4_frames(video_path, self.args.video_frames)
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
                "video_ref": video_path,
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


def collate_sft_batch(batch):
    return {
        "inputs": _stack_inputs([item["inputs"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "prompt_text": [item["prompt_text"] for item in batch],
        "answer_text": [item["answer_text"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "video_ref": [item["video_ref"] for item in batch],
    }


def build_train_loader(args, split: str, batch_size: int, num_workers: int, is_train: bool):
    processor = build_vlm_processor(args)

    if args.dataset_type == "hd_epic_local":
        records = _load_records(args) if args.annotation_path else _discover_hd_epic_records(args)
        dataset = LocalHD_EPICDataset(
            records=records, processor=processor, args=args, split_name=split, is_train=is_train
        )
    else:
        raise RuntimeError(
            f"Unsupported dataset_type={args.dataset_type}. "
            "HD-EPIC integration uses --dataset_type hd_epic_local."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_sft_batch,
        pin_memory=torch.cuda.is_available(),
    )
