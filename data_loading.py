from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


DEFAULT_TEXT_COLUMNS = ("text", "template", "sentence", "caption")
DEFAULT_LABEL_COLUMNS = ("label", "labels")


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


def preprocess_vlm_video_inputs(
    vlm_processor,
    frames,
    text,
    vl_backend="llava_video",
    vlm_max_text_len=256,
    squeeze_batch_dim=True,
):
    tokenizer = getattr(vlm_processor, "tokenizer", None)
    prompt = text if isinstance(text, str) else ""
    if tokenizer is not None:
        vocab = tokenizer.get_vocab()
        if not any(token in prompt for token in ("<video>", "<image>", "<img>")):
            if vl_backend == "internvl":
                for token in ("<video>", "<image>", "<img>"):
                    if token in vocab:
                        prompt = f"{token}\n{prompt}"
                        break
            elif "<video>" in vocab:
                prompt = f"<video>\n{prompt}"
            elif "<image>" in vocab:
                prompt = f"<image>\n{prompt}"

    try:
        inputs = vlm_processor(
            text=prompt,
            videos=frames,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            max_length=vlm_max_text_len,
        )
    except TypeError:
        inputs = vlm_processor(
            text=prompt,
            images=frames,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            max_length=vlm_max_text_len,
        )

    packed = {}
    for key, value in dict(inputs).items():
        if squeeze_batch_dim and torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            value = value.squeeze(0)
        packed[key] = value
    return packed


def _frame_to_pil(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    elif hasattr(frame, "asnumpy"):
        frame = frame.asnumpy()
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


def _video_length(video) -> int:
    for attr in ("num_frames", "frames"):
        value = getattr(video, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    metadata = getattr(video, "metadata", None)
    for attr in ("num_frames", "frames"):
        value = getattr(metadata, attr, None) if metadata is not None else None
        if isinstance(value, int) and value > 0:
            return value
    try:
        return len(video)
    except Exception as e:
        raise RuntimeError(
            "Could not determine the decoded video length. Install the video decoding extras used by datasets, "
            "for example `pip install datasets[video] torchcodec`."
        ) from e


def _extract_frames(video, frame_indices):
    if hasattr(video, "get_batch"):
        batch = video.get_batch(frame_indices)
        if hasattr(batch, "asnumpy"):
            batch = batch.asnumpy()
        return [_frame_to_pil(frame) for frame in batch]

    if hasattr(video, "get_frames"):
        batch = video.get_frames(frame_indices)
        if hasattr(batch, "asnumpy"):
            batch = batch.asnumpy()
        return [_frame_to_pil(frame) for frame in batch]

    if hasattr(video, "get_frame_at"):
        frames = []
        for idx in frame_indices:
            item = video.get_frame_at(int(idx))
            data = item[0] if isinstance(item, tuple) else getattr(item, "data", item)
            frames.append(_frame_to_pil(data))
        return frames

    return [_frame_to_pil(video[int(idx)]) for idx in frame_indices]


def decode_video_frames(video, num_frames: int):
    total_frames = _video_length(video)
    frame_indices = _sample_frame_indices(total_frames, num_frames)
    frames = _extract_frames(video, frame_indices)
    if len(frames) < num_frames and frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


def format_sample_text(template: str, sample_text: str) -> str:
    sample_text = "" if sample_text is None else str(sample_text).strip()
    if not template:
        return sample_text
    try:
        return template.format(text=sample_text)
    except Exception:
        if sample_text:
            return f"{template}\n{sample_text}"
        return template


def _infer_label_names(dataset, label_column: str):
    features = getattr(dataset, "features", None)
    if not features or label_column not in features:
        return None
    label_feature = features[label_column]
    names = getattr(label_feature, "names", None)
    if names:
        return list(names)
    return None


def _resolve_text_column(sample, requested_column: Optional[str]):
    if requested_column and requested_column in sample:
        return requested_column
    for candidate in DEFAULT_TEXT_COLUMNS:
        if candidate in sample:
            return candidate
    return None


def _resolve_label_value(sample, requested_column: Optional[str]):
    if requested_column and requested_column in sample:
        return int(sample[requested_column])
    for candidate in DEFAULT_LABEL_COLUMNS:
        if candidate in sample:
            return int(sample[candidate])
    raise KeyError(f"Could not find a label column in sample. Checked: {DEFAULT_LABEL_COLUMNS}")


class HFVideoClassificationDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        processor,
        args,
        split: str,
        is_train: bool,
    ):
        self.dataset = dataset
        self.processor = processor
        self.args = args
        self.split = split
        self.is_train = is_train

    def _iter_dataset(self) -> Iterable:
        dataset = self.dataset
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        if hasattr(dataset, "shard") and num_workers > 1:
            dataset = dataset.shard(num_shards=num_workers, index=worker_id, contiguous=False)

        if self.is_train and hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=self.args.seed + worker_id, buffer_size=self.args.shuffle_buffer)

        if hasattr(dataset, "__iter__") and not hasattr(dataset, "__len__"):
            return iter(dataset)

        indices = range(worker_id, len(dataset), num_workers)
        return (dataset[idx] for idx in indices)

    def __iter__(self):
        sample_count = 0
        iterator = self._iter_dataset()
        text_column = self.args.text_column
        for sample in iterator:
            if self.args.max_samples_per_split > 0 and sample_count >= self.args.max_samples_per_split:
                break

            text_column = _resolve_text_column(sample, text_column)
            sample_text = sample.get(text_column, "") if text_column is not None else ""
            prompt = format_sample_text(self.args.text_prompt_template, sample_text)
            video = sample[self.args.video_column]
            frames = decode_video_frames(video, self.args.video_frames)
            inputs = preprocess_vlm_video_inputs(
                vlm_processor=self.processor,
                frames=frames,
                text=prompt,
                vl_backend=self.args.vl_backend,
                vlm_max_text_len=self.args.vl_max_text_len,
            )
            label = _resolve_label_value(sample, self.args.label_column)

            sample_count += 1
            yield {
                "inputs": inputs,
                "label": torch.tensor(label, dtype=torch.long),
                "text": sample_text,
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


def collate_hf_batch(batch):
    return {
        "inputs": _stack_inputs([item["inputs"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "texts": [item["text"] for item in batch],
    }


def _load_split(args, split: str):
    from datasets import load_dataset

    load_kwargs = {
        "name": args.dataset_config or None,
        "split": split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code_dataset:
        load_kwargs["trust_remote_code"] = True

    try:
        dataset = load_dataset(args.dataset_name, **load_kwargs)
    except RuntimeError as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg:
            raise RuntimeError(
                "This dataset repo is script-backed, but your installed `datasets` version no longer supports Hub "
                "dataset scripts. Use one of these options: (1) downgrade to `datasets<4.0` if you want to load "
                f"`{args.dataset_name}` directly, (2) convert/download the dataset to a standard format and load it "
                "locally, or (3) switch to a parquet/arrow-backed Hub dataset. The current training code is fine; "
                "the failure is in dataset loading, not DDP."
            ) from exc
        raise
    return dataset


def hf_video_loader(args, split: str, batch_size: int, num_workers: int, is_train: bool):
    dataset = _load_split(args, split)
    label_names = _infer_label_names(dataset, args.label_column)
    processor = build_vlm_processor(args)
    wrapped = HFVideoClassificationDataset(dataset=dataset, processor=processor, args=args, split=split, is_train=is_train)
    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_hf_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, label_names
