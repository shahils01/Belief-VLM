import csv
import json
import os
from dataclasses import dataclass
from glob import glob
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from data_loading import (
    _get_first,
    _normalize_text,
    _parse_timecode_seconds,
    _stable_fold,
    decode_mp4_frames,
)


@dataclass
class IntentSample:
    sample_id: str
    action: str
    question: str
    choices: List[str]
    label: int
    answer_text: str
    frames: List


def _load_csv(path: str):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "samples", "annotations", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
        if obj and all(isinstance(v, dict) for v in obj.values()):
            rows = []
            for sample_id, value in obj.items():
                item = dict(value)
                item.setdefault("id", sample_id)
                rows.append(item)
            return rows
    raise RuntimeError(f"Unsupported JSON structure for {path}.")


def _load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_records_from_path(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")
    if path.endswith(".csv"):
        return _load_csv(path)
    if path.endswith(".json"):
        return _load_json(path)
    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    raise RuntimeError(f"Unsupported annotation format: {path}")


def _resolve_split_annotation_path(annotation_path: str, split_name: str):
    if os.path.isfile(annotation_path):
        return annotation_path
    if not os.path.isdir(annotation_path):
        raise FileNotFoundError(f"annotation_path is neither file nor directory: {annotation_path}")

    aliases = [split_name]
    if split_name == "validation":
        aliases.extend(["val", "dev"])
    if split_name == "train":
        aliases.extend(["training"])
    if split_name == "test":
        aliases.extend(["testing"])

    candidates = []
    for alias in aliases:
        candidates.extend(
            [
                os.path.join(annotation_path, f"{alias}.csv"),
                os.path.join(annotation_path, f"{alias}.json"),
                os.path.join(annotation_path, f"{alias}.jsonl"),
            ]
        )
    for c in candidates:
        if os.path.isfile(c):
            return c

    # fallback: pick the first matching file by pattern
    all_files = []
    all_files.extend(sorted(glob(os.path.join(annotation_path, "*.csv"))))
    all_files.extend(sorted(glob(os.path.join(annotation_path, "*.json"))))
    all_files.extend(sorted(glob(os.path.join(annotation_path, "*.jsonl"))))
    if len(all_files) == 1:
        return all_files[0]

    raise FileNotFoundError(
        f"Could not resolve split file for split={split_name} inside {annotation_path}. "
        "Expected train/val/test naming."
    )


def _parse_choices_from_record(record, options_column: str):
    explicit = _get_first(record, [options_column, "options", "choices", "answer_options"])
    if isinstance(explicit, (list, tuple)):
        vals = [_normalize_text(v) for v in explicit if _normalize_text(v)]
        if vals:
            return vals
    if isinstance(explicit, str):
        text = explicit.strip()
        if text:
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        vals = [_normalize_text(v) for v in parsed if _normalize_text(v)]
                        if vals:
                            return vals
                except Exception:
                    pass
            if "|" in text:
                vals = [_normalize_text(v) for v in text.split("|") if _normalize_text(v)]
                if vals:
                    return vals
            if "\n" in text:
                vals = [_normalize_text(v) for v in text.split("\n") if _normalize_text(v)]
                if vals:
                    return vals

    # IntentQA / NExT-QA style: a0 ... a4
    vals = []
    i = 0
    while True:
        key = f"a{i}"
        if key not in record:
            break
        one = _normalize_text(record.get(key))
        if one:
            vals.append(one)
        i += 1
    if vals:
        return vals
    return []


def _resolve_label_and_answer(record, answer_column: str, choices: List[str]):
    idx = _get_first(record, ["correct_idx", "answer_idx", "label_idx", answer_column, "answer"])
    if idx not in (None, ""):
        try:
            idx_int = int(idx)
            if 0 <= idx_int < len(choices):
                return idx_int, choices[idx_int]
            if 1 <= idx_int <= len(choices):
                return idx_int - 1, choices[idx_int - 1]
        except Exception:
            pass

    # fallback by text match
    answer_txt = _normalize_text(_get_first(record, [answer_column, "answer", "label"]))
    if answer_txt:
        for i, c in enumerate(choices):
            if c.strip().lower() == answer_txt.strip().lower():
                return i, c
    return None, ""


def _get_or_build_video_lookup(args):
    lookup = getattr(args, "_nextvqa_video_lookup", None)
    if lookup is not None:
        return lookup

    ext = (args.video_extension if args.video_extension else "mp4").lstrip(".")
    pattern = os.path.join(args.video_root, "**", f"*.{ext}")
    files = sorted(glob(pattern, recursive=True))

    lookup = {}
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        base_with_ext = os.path.basename(path)
        rel = os.path.relpath(path, args.video_root)
        rel_norm = rel.replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_norm)[0]
        for key in (base, base_with_ext, rel_norm, rel_no_ext):
            if key and key not in lookup:
                lookup[key] = path

    setattr(args, "_nextvqa_video_lookup", lookup)
    return lookup


def _resolve_video_path(args, record):
    direct = _get_first(record, [args.video_path_column, "video_path", "path"])
    if direct:
        direct = str(direct)
        if os.path.isfile(direct):
            return direct
        candidate = os.path.join(args.video_root, direct)
        if os.path.isfile(candidate):
            return candidate

    vid = _get_first(record, [args.video_id_column, "video_id", "vid", "video", "gif_name"])
    if vid in (None, ""):
        raise RuntimeError("Could not resolve video id/path from record.")
    vid = str(vid).strip()
    ext = (args.video_extension if args.video_extension else "mp4").lstrip(".")

    candidates = [
        os.path.join(args.video_root, f"{vid}.{ext}"),
        os.path.join(args.video_root, vid, f"{vid}.{ext}"),
        os.path.join(args.video_root, vid),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Recursive fallback for layouts like VIDEO_ROOT/0000/*.mp4
    lookup = _get_or_build_video_lookup(args)
    keys = [
        vid,
        f"{vid}.{ext}",
        vid.replace("\\", "/"),
        vid.replace("\\", "/").lstrip("./"),
    ]
    for key in keys:
        if key in lookup:
            return lookup[key]

    raise FileNotFoundError(f"Could not find video for id={vid}. Tried direct candidates and recursive lookup under {args.video_root}")


def _resolve_clip_window(record):
    start = _get_first(record, ["start_time", "clip_start_time", "video_start_time", "start"])
    end = _get_first(record, ["end_time", "clip_end_time", "video_end_time", "end"])
    return _parse_timecode_seconds(start), _parse_timecode_seconds(end)


class NextVQAIntentDataset(Dataset):
    def __init__(self, records, args, split_name: str, is_train: bool):
        self.args = args
        self.split_name = split_name
        selected = []
        # If annotation comes from a split-specific file, do not hash-split again.
        split_from_dir = os.path.isdir(args.annotation_path)
        if split_from_dir:
            selected = list(records[: args.max_samples_per_split or None])
        else:
            for idx, record in enumerate(records):
                sample_id = str(_get_first(record, [args.id_column, "id", "qid", "sample_id", "uid"]) or idx)
                val_ratio = max(0.0, min(0.5, float(args.val_ratio)))
                if val_ratio > 0.0:
                    fold = _stable_fold(sample_id, args.seed)
                    in_val = fold < val_ratio
                    if is_train and in_val:
                        continue
                    if (not is_train) and (not in_val):
                        continue
                selected.append(record)
                if args.max_samples_per_split > 0 and len(selected) >= args.max_samples_per_split:
                    break

        dropped_no_choices = 0
        dropped_no_label = 0
        dropped_no_video = 0
        missing_video_ids = []
        validated = []
        for idx, record in enumerate(selected):
            sample_id = str(_get_first(record, [args.id_column, "id", "qid", "sample_id", "uid"]) or idx)
            choices = _parse_choices_from_record(record, args.options_column)
            if not choices:
                dropped_no_choices += 1
                continue
            label, _ = _resolve_label_and_answer(record, args.answer_column, choices)
            if label is None:
                dropped_no_label += 1
                continue
            try:
                _resolve_video_path(args, record)
            except FileNotFoundError:
                dropped_no_video += 1
                if len(missing_video_ids) < 5:
                    vid = _get_first(record, [args.video_id_column, "video_id", "vid", "video", "gif_name"])
                    missing_video_ids.append(str(vid) if vid is not None else sample_id)
                continue
            validated.append(record)

        self.records = validated
        if dropped_no_choices or dropped_no_label or dropped_no_video:
            print(
                f"[NextVQAIntentDataset:{split_name}] dropped {dropped_no_choices} samples with no choices, "
                f"{dropped_no_label} with no valid label, and {dropped_no_video} with missing videos."
            )
            if missing_video_ids:
                print(f"[NextVQAIntentDataset:{split_name}] example missing video ids: {missing_video_ids}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sample_id = str(_get_first(record, [self.args.id_column, "id", "qid", "sample_id", "uid"]) or index)
        action = _normalize_text(
            _get_first(
                record,
                [self.args.action_column, "action", "action_id", "verb", "lemmatized_verb"],
            )
        )
        question = _normalize_text(
            _get_first(record, [self.args.question_column, "question", "prompt", "instruction", "query"])
        )
        choices = _parse_choices_from_record(record, self.args.options_column)
        if not choices:
            raise RuntimeError(f"Sample {sample_id} has no answer choices.")
        label, answer_text = _resolve_label_and_answer(record, self.args.answer_column, choices)
        if label is None:
            raise RuntimeError(f"Sample {sample_id} has no valid answer index/text.")

        video_path = _resolve_video_path(self.args, record)
        start_time_sec, end_time_sec = _resolve_clip_window(record)
        frames = decode_mp4_frames(
            video_path,
            self.args.video_frames,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
        )

        return IntentSample(
            sample_id=sample_id,
            action=action,
            question=question,
            choices=choices,
            label=label,
            answer_text=answer_text,
            frames=frames,
        )


def collate_intent_batch(batch: List[IntentSample]):
    return {
        "ids": [x.sample_id for x in batch],
        "actions": [x.action for x in batch],
        "questions": [x.question for x in batch],
        "choices": [x.choices for x in batch],
        "labels": torch.tensor([x.label for x in batch], dtype=torch.long),
        "answer_texts": [x.answer_text for x in batch],
        "frames": [x.frames for x in batch],
    }


def build_nextvqa_intent_loader(args, split_name: str, batch_size: int, num_workers: int, is_train: bool):
    split_path = _resolve_split_annotation_path(args.annotation_path, split_name)
    records = _load_records_from_path(split_path)
    dataset = NextVQAIntentDataset(records=records, args=args, split_name=split_name, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        collate_fn=collate_intent_batch,
        pin_memory=torch.cuda.is_available(),
    )
