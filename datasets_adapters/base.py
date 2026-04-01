import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from general_data_utils import get_first, load_local_records, normalize_text


def _lazy_load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The generalized benchmark pipeline requires the `datasets` package. "
            "Install the repo environment with `bash setup_palmetto_env.sh ...` "
            "or `pip install datasets`."
        ) from exc
    return load_dataset(*args, **kwargs)


def normalize_choices(raw_choices):
    if raw_choices is None:
        return None
    if isinstance(raw_choices, dict):
        return [normalize_text(raw_choices[key]) for key in sorted(raw_choices)]
    if isinstance(raw_choices, str):
        return [line.strip() for line in raw_choices.split("\n") if line.strip()]
    return [normalize_text(choice) for choice in raw_choices]


def letter_to_index(value: str) -> Optional[int]:
    if value is None:
        return None
    text = normalize_text(value)
    if not text:
        return None
    if text.isdigit():
        idx = int(text)
        return idx - 1 if idx > 0 else idx
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    first = text[0].upper()
    if first in alpha:
        return alpha.index(first)
    return None


def verbalize_choices(choices):
    if not choices:
        return ""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for idx, choice in enumerate(choices):
        prefix = letters[idx] if idx < len(letters) else str(idx + 1)
        lines.append(f"{prefix}. {normalize_text(choice)}")
    return "\n".join(lines)


def build_mc_prompt(question: str, choices) -> str:
    question = normalize_text(question)
    options = verbalize_choices(choices)
    if options:
        return f"{question}\nOptions:\n{options}\nAnswer with the best option text."
    return question


def ensure_local_media_path(media, media_root: str = ""):
    if isinstance(media, dict):
        candidate = media.get("path") or media.get("video") or media.get("image")
        if candidate is None and media.get("bytes") is not None:
            return media
        if candidate is None:
            return media
        media = candidate
    if isinstance(media, str) and media and not os.path.isabs(media) and media_root:
        return os.path.join(media_root, media)
    return media


class NormalizedMapDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class NormalizedIterableDataset(IterableDataset):
    def __init__(self, source: Iterable[Dict[str, Any]], formatter):
        self.source = source
        self.formatter = formatter

    def __iter__(self):
        for item in self.source:
            sample = self.formatter(item)
            if sample is not None:
                yield sample


class DatasetAdapter(ABC):
    name = "base"
    supports_streaming = True
    media_type = "image"
    default_hf_repo = ""
    default_hf_config = None
    default_train_split = "train"
    default_eval_split = "validation"
    default_evaluator = "generic_generation"

    def dataset_repo(self, args) -> str:
        return args.dataset_repo or self.default_hf_repo

    def dataset_config(self, args) -> Optional[str]:
        return args.dataset_config or self.default_hf_config

    def train_split(self, args) -> str:
        return args.train_split or self.default_train_split

    def eval_split(self, args) -> str:
        return args.eval_split or self.default_eval_split

    def _limit_items(self, items, limit: int):
        if limit <= 0:
            return items
        return items[:limit]

    def _load_hf_records(self, args, split: str):
        repo = self.dataset_repo(args)
        config = self.dataset_config(args)
        kwargs = {"split": split, "streaming": bool(args.streaming)}
        if config:
            kwargs["name"] = config
        if args.dataset_revision:
            kwargs["revision"] = args.dataset_revision
        if args.trust_remote_code_dataset:
            kwargs["trust_remote_code"] = True
        return _lazy_load_dataset(repo, **kwargs)

    def _load_local_records(self, args):
        if not args.annotation_path:
            raise RuntimeError(f"{self.name} requires either --annotation_path or a Hugging Face dataset.")
        return load_local_records(args.annotation_path)

    def build_train_dataset(self, args):
        return self._build_dataset(args, self.train_split(args), args.max_train_samples)

    def build_eval_dataset(self, args, split=None):
        target_split = split or self.eval_split(args)
        return self._build_dataset(args, target_split, args.max_eval_samples)

    def _build_dataset(self, args, split: str, limit: int):
        if args.annotation_path:
            items = [
                sample
                for sample in (self.format_sample(record, split, args) for record in self._load_local_records(args))
                if sample is not None
            ]
            return NormalizedMapDataset(self._limit_items(items, limit))

        records = self._load_hf_records(args, split)
        if args.streaming:
            return NormalizedIterableDataset(records, lambda raw: self.format_sample(raw, split, args))

        items = [
            sample
            for sample in (self.format_sample(record, split, args) for record in records)
            if sample is not None
        ]
        return NormalizedMapDataset(self._limit_items(items, limit))

    @abstractmethod
    def format_sample(self, raw_item: Dict[str, Any], split: str, args) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

