import warnings

import numpy as np
import torch

from belief_db import default_belief_text
from data_loading import (
    _build_vqa_prompt_choices,
    _get_first,
    _resolve_hd_epic_clip_window,
    _resolve_hd_epic_video_path,
    _stack_inputs,
    build_prompt_only_example,
    decode_mp4_frames,
)


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return array / norms


class _FaissIndex:
    def __init__(self, embeddings: np.ndarray):
        import faiss

        self._faiss = faiss
        self.index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        self.index.add(embeddings.astype(np.float32))

    def search(self, queries: np.ndarray, top_k: int):
        scores, indices = self.index.search(queries.astype(np.float32), int(top_k))
        return scores, indices


class _NumpyIndex:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)

    def search(self, queries: np.ndarray, top_k: int):
        scores = queries @ self.embeddings.T
        order = np.argsort(-scores, axis=1)[:, :top_k]
        gathered_scores = np.take_along_axis(scores, order, axis=1)
        return gathered_scores, order


def _extract_frames_from_batch(batch_inputs, idx: int, batch_size: int, video_frames: int):
    frame_tensor = None
    for key in ("pixel_values", "pixel_values_videos", "video_values", "video", "videos"):
        value = batch_inputs.get(key)
        if torch.is_tensor(value):
            frame_tensor = value
            break
    if frame_tensor is None:
        raise RuntimeError("Could not locate video tensor in batch inputs.")

    if frame_tensor.dim() == 4:
        start = idx * video_frames
        end = start + video_frames
        clip = frame_tensor[start:end]
    elif frame_tensor.dim() == 5:
        clip = frame_tensor[idx]
    else:
        raise RuntimeError(f"Unsupported video tensor shape: {tuple(frame_tensor.shape)}")
    return clip


def _build_prompt_only_inputs_from_frames(processor, frames, prompt, args):
    packed = build_prompt_only_example(
        processor=processor,
        frames=[frame for frame in frames],
        prompt=prompt,
        vl_backend=args.vl_backend,
        max_text_len=args.vl_max_text_len,
    )
    return {k: v for k, v in packed.items() if k != "prompt_text"}


def _compose_prior_prompt(base_prompt: str, prior_texts, prefix: str):
    texts = [str(text).strip() for text in prior_texts if str(text).strip()]
    if not texts:
        return base_prompt
    prior_block = "\n".join(f"- {text}" for text in texts)
    return f"{prefix}\n{prior_block}\n\n{base_prompt}"


class MultimodalBeliefDB:
    def __init__(self, entries, embeddings, prior_prefix: str, backend: str):
        self.entries = entries
        self.embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
        self.prior_prefix = str(prior_prefix)
        self.backend = backend
        if backend == "faiss":
            self.index = _FaissIndex(self.embeddings)
        else:
            self.index = _NumpyIndex(self.embeddings)

    @classmethod
    def from_records(cls, records, model, processor, args, device):
        entries = []
        embeddings = []
        batch_size = max(1, int(getattr(args, "db_build_batch_size", 4)))
        pending_inputs = []
        pending_meta = []

        def flush():
            if not pending_inputs:
                return
            batch_inputs = _stack_inputs(pending_inputs)
            batch_inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch_inputs.items()
            }
            with torch.no_grad():
                encoded = model(batch_inputs, return_hidden_states=True, pooling=args.state_pooling)
                pooled = encoded["pooled_state"].detach().float().cpu().numpy()
            embeddings.extend(pooled)
            entries.extend(pending_meta)
            pending_inputs.clear()
            pending_meta.clear()

        for idx, record in enumerate(records):
            try:
                prompt, _, _ = _build_vqa_prompt_choices(args, record)
                video_path = _resolve_hd_epic_video_path(args, record)
                start_time_sec, end_time_sec = _resolve_hd_epic_clip_window(record)
                frames = decode_mp4_frames(
                    video_path,
                    args.video_frames,
                    start_time_sec=start_time_sec,
                    end_time_sec=end_time_sec,
                )
                sample_id = str(_get_first(record, [args.id_column, "id", "sample_id", "uid", "video_id"]) or idx)
                pending_inputs.append(
                    _build_prompt_only_inputs_from_frames(
                        processor=processor,
                        frames=frames,
                        prompt=prompt,
                        args=args,
                    )
                )
                pending_meta.append(
                    {
                        "id": sample_id,
                        "belief_text": default_belief_text(record),
                    }
                )
                if len(pending_inputs) >= batch_size:
                    flush()
            except Exception as exc:
                warnings.warn(f"Skipping multimodal DB record at idx={idx}: {exc}", RuntimeWarning)
        flush()

        if not entries:
            warnings.warn(
                "Multimodal belief DB was built with zero entries; retrieval will be disabled.",
                RuntimeWarning,
            )
            return cls([], np.zeros((0, 1), dtype=np.float32), getattr(args, "db_prior_prefix", "Belief prior:"), "numpy")

        requested_backend = getattr(args, "db_index_backend", "auto")
        backend = "numpy"
        if requested_backend in ("auto", "faiss"):
            try:
                import faiss  # noqa: F401

                backend = "faiss"
            except Exception:
                warnings.warn(
                    f"Requested multimodal DB backend '{requested_backend}' but faiss is not available. "
                    "Falling back to NumPy similarity search.",
                    RuntimeWarning,
                )
        elif requested_backend == "numpy":
            warnings.warn(
                "Using NumPy similarity search for multimodal belief DB. Install faiss for a faster indexed backend.",
                RuntimeWarning,
            )
        return cls(entries, np.asarray(embeddings, dtype=np.float32), getattr(args, "db_prior_prefix", "Belief prior:"), backend)

    def retrieve(self, query_embeddings: torch.Tensor, sample_ids, top_k: int):
        if not self.entries or int(top_k) <= 0:
            return [[] for _ in sample_ids]
        queries = query_embeddings.detach().float().cpu().numpy()
        queries = _normalize_rows(queries)
        scores, indices = self.index.search(queries, min(int(top_k) + 1, len(self.entries)))
        all_texts = []
        for row, sample_id in enumerate(sample_ids):
            texts = []
            for idx in indices[row].tolist():
                if idx < 0:
                    continue
                entry = self.entries[idx]
                if entry["id"] == str(sample_id):
                    continue
                if entry["belief_text"]:
                    texts.append(entry["belief_text"])
                if len(texts) >= int(top_k):
                    break
            all_texts.append(texts)
        return all_texts

    def augment_prompts(self, prompts, retrieved_texts):
        return [
            _compose_prior_prompt(prompt, texts, self.prior_prefix)
            for prompt, texts in zip(prompts, retrieved_texts)
        ]


def build_prior_inputs_from_batch(processor, batch_inputs, prompts, args):
    batch_size = len(prompts)
    items = []
    for row in range(batch_size):
        frames = _extract_frames_from_batch(batch_inputs, row, batch_size, args.video_frames)
        items.append(
            _build_prompt_only_inputs_from_frames(
                processor=processor,
                frames=frames,
                prompt=prompts[row],
                args=args,
            )
        )
    return _stack_inputs(items)
