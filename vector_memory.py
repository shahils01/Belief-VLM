import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from data_loading import _stack_inputs, build_prompt_only_example, build_sft_example


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return array.astype(np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (array / norms).astype(np.float32)


def _short_belief_text(answer_text: str, max_words: int = 12) -> str:
    text = str(answer_text or "").replace("\n", " ").strip()
    if not text:
        return ""
    tokens = text.split()
    if len(tokens) <= max_words:
        return f"Likely goal: {' '.join(tokens).strip(' ,.;:')}."
    return f"Likely goal: {' '.join(tokens[:max_words]).strip(' ,.;:')}..."


def _compose_prior_prompt(base_prompt: str, prior_texts, prefix: str):
    texts = [str(text).strip() for text in prior_texts if str(text).strip()]
    if not texts:
        return base_prompt
    prior_block = "\n".join(f"- {text}" for text in texts)
    return f"{prefix}\n{prior_block}\n\n{base_prompt}"


def extract_frames_from_batch(batch_inputs, idx: int, video_frames: int):
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


def build_prompt_only_inputs_from_batch(processor, batch_inputs, prompts, args):
    items = []
    for row, prompt in enumerate(prompts):
        frames = extract_frames_from_batch(batch_inputs, row, args.video_frames)
        packed = build_prompt_only_example(
            processor=processor,
            frames=[frame for frame in frames],
            prompt=prompt,
            vl_backend=args.vl_backend,
            max_text_len=args.vl_max_text_len,
        )
        items.append({k: v for k, v in packed.items() if k != "prompt_text"})
    return _stack_inputs(items)


def build_sft_batch_from_batch(processor, batch_inputs, prompts, answers, args):
    items = []
    for row, (prompt, answer) in enumerate(zip(prompts, answers)):
        frames = extract_frames_from_batch(batch_inputs, row, args.video_frames)
        packed = build_sft_example(
            processor=processor,
            frames=[frame for frame in frames],
            prompt=prompt,
            answer=answer,
            vl_backend=args.vl_backend,
            max_text_len=args.vl_max_text_len,
        )
        items.append(
            {
                "inputs": {k: v for k, v in packed.items() if k not in {"labels", "prompt_text", "answer_text"}},
                "labels": packed["labels"],
            }
        )
    max_len = max(int(item["labels"].shape[0]) for item in items)
    labels = [
        torch.nn.functional.pad(item["labels"], (0, max_len - int(item["labels"].shape[0])), value=-100)
        for item in items
    ]
    return {
        "inputs": _stack_inputs([item["inputs"] for item in items]),
        "labels": torch.stack(labels, dim=0),
    }


class _FaissIndex:
    def __init__(self, dim: int):
        import faiss

        self.index = faiss.IndexFlatIP(int(dim))

    def add(self, embeddings: np.ndarray):
        if embeddings.size:
            self.index.add(embeddings.astype(np.float32))

    def search(self, queries: np.ndarray, top_k: int):
        return self.index.search(queries.astype(np.float32), int(top_k))


class _NumpyIndex:
    def __init__(self):
        self.embeddings = np.zeros((0, 1), dtype=np.float32)

    def add(self, embeddings: np.ndarray):
        embeddings = embeddings.astype(np.float32)
        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.concatenate([self.embeddings, embeddings], axis=0)

    def search(self, queries: np.ndarray, top_k: int):
        if self.embeddings.size == 0:
            n = queries.shape[0]
            return np.zeros((n, 0), dtype=np.float32), np.zeros((n, 0), dtype=np.int64)
        scores = queries @ self.embeddings.T
        order = np.argsort(-scores, axis=1)[:, :top_k]
        gathered_scores = np.take_along_axis(scores, order, axis=1)
        return gathered_scores, order


@dataclass
class MemoryEntry:
    id: str
    task_name: str
    belief_text: str
    reward: float


class OnlineVectorMemory:
    def __init__(self, dim: int, prior_prefix: str, backend: str = "auto", same_task_first: bool = True):
        self.dim = int(dim)
        self.prior_prefix = str(prior_prefix)
        self.same_task_first = bool(same_task_first)
        self.entries: list[MemoryEntry] = []
        self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
        self.answer_embeddings = np.zeros((0, self.dim), dtype=np.float32)
        self.rewards = np.zeros((0,), dtype=np.float32)
        requested = str(backend or "auto")
        self.backend = "numpy"
        self._index = None
        if requested in ("auto", "faiss"):
            try:
                self._index = _FaissIndex(self.dim)
                self.backend = "faiss"
            except Exception:
                warnings.warn(
                    f"Requested vector-memory backend '{requested}' but faiss is not available. "
                    "Falling back to NumPy similarity search.",
                    RuntimeWarning,
                )
        if self._index is None:
            if requested == "numpy":
                warnings.warn(
                    "Using NumPy similarity search for vector memory. Install faiss for faster retrieval.",
                    RuntimeWarning,
                )
            self._index = _NumpyIndex()

    @classmethod
    def from_state_dict(cls, state, args):
        memory = cls(
            dim=int(state["dim"]),
            prior_prefix=state.get("prior_prefix", getattr(args, "db_prior_prefix", "Belief prior:")),
            backend=getattr(args, "db_index_backend", "auto"),
            same_task_first=bool(state.get("same_task_first", getattr(args, "db_same_task_first", True))),
        )
        embeddings = np.asarray(state.get("embeddings", np.zeros((0, memory.dim), dtype=np.float32)), dtype=np.float32)
        answer_embeddings = np.asarray(
            state.get("answer_embeddings", np.zeros((0, memory.dim), dtype=np.float32)),
            dtype=np.float32,
        )
        rewards = np.asarray(state.get("rewards", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        entries = [
            MemoryEntry(
                id=str(item["id"]),
                task_name=str(item.get("task_name", "")),
                belief_text=str(item.get("belief_text", "")),
                reward=float(item.get("reward", 0.0)),
            )
            for item in state.get("entries", [])
        ]
        memory.entries = entries
        memory.embeddings = _normalize_rows(embeddings)
        memory.answer_embeddings = _normalize_rows(answer_embeddings)
        memory.rewards = rewards.astype(np.float32)
        if memory.embeddings.size:
            memory._index.add(memory.embeddings)
        return memory

    def state_dict(self):
        return {
            "dim": self.dim,
            "prior_prefix": self.prior_prefix,
            "same_task_first": self.same_task_first,
            "entries": [
                {"id": e.id, "task_name": e.task_name, "belief_text": e.belief_text, "reward": e.reward}
                for e in self.entries
            ],
            "embeddings": self.embeddings,
            "answer_embeddings": self.answer_embeddings,
            "rewards": self.rewards,
        }

    def __len__(self):
        return len(self.entries)

    def add(
        self,
        embeddings: torch.Tensor | np.ndarray,
        sample_ids,
        task_names,
        answer_texts,
        max_words: int = 12,
        answer_embeddings: torch.Tensor | np.ndarray | None = None,
        rewards: torch.Tensor | np.ndarray | None = None,
    ):
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().float().cpu().numpy()
        embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
        if answer_embeddings is None:
            answer_embeddings = np.zeros_like(embeddings, dtype=np.float32)
        elif torch.is_tensor(answer_embeddings):
            answer_embeddings = answer_embeddings.detach().float().cpu().numpy()
        answer_embeddings = _normalize_rows(np.asarray(answer_embeddings, dtype=np.float32))
        if rewards is None:
            rewards = np.ones((embeddings.shape[0],), dtype=np.float32)
        elif torch.is_tensor(rewards):
            rewards = rewards.detach().float().cpu().numpy()
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        rows = []
        new_entries = []
        for row, (sample_id, task_name, answer_text) in enumerate(zip(sample_ids, task_names, answer_texts)):
            belief_text = _short_belief_text(answer_text, max_words=max_words)
            if not belief_text:
                continue
            rows.append(row)
            new_entries.append(
                MemoryEntry(
                    id=str(sample_id),
                    task_name=str(task_name),
                    belief_text=belief_text,
                    reward=float(rewards[row]),
                )
            )
        if not rows:
            return
        row_index = np.asarray(rows, dtype=np.int64)
        new_embeddings = embeddings[row_index]
        new_answer_embeddings = answer_embeddings[row_index]
        new_rewards = rewards[row_index]
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.concatenate([self.embeddings, new_embeddings], axis=0)
        if self.answer_embeddings.size == 0:
            self.answer_embeddings = new_answer_embeddings
        else:
            self.answer_embeddings = np.concatenate([self.answer_embeddings, new_answer_embeddings], axis=0)
        if self.rewards.size == 0:
            self.rewards = new_rewards
        else:
            self.rewards = np.concatenate([self.rewards, new_rewards], axis=0)
        self.entries.extend(new_entries)
        self._index.add(new_embeddings)

    def retrieve(self, query_embeddings: torch.Tensor | np.ndarray, sample_ids, task_names, top_k: int):
        if len(self.entries) == 0 or int(top_k) <= 0:
            return [[] for _ in sample_ids]
        if torch.is_tensor(query_embeddings):
            query_embeddings = query_embeddings.detach().float().cpu().numpy()
        queries = _normalize_rows(np.asarray(query_embeddings, dtype=np.float32))
        overfetch = min(len(self.entries), max(int(top_k) * 16, int(top_k) + 8))
        _, indices = self._index.search(queries, max(overfetch, int(top_k)))
        retrieved = []
        for row, (sample_id, task_name) in enumerate(zip(sample_ids, task_names)):
            same_task = []
            fallback = []
            for idx in indices[row].tolist():
                if idx < 0 or idx >= len(self.entries):
                    continue
                entry = self.entries[idx]
                if entry.id == str(sample_id):
                    continue
                if not entry.belief_text:
                    continue
                if entry.task_name == str(task_name):
                    same_task.append(entry.belief_text)
                else:
                    fallback.append(entry.belief_text)
            if self.same_task_first:
                texts = same_task[: int(top_k)]
                if len(texts) < int(top_k):
                    texts.extend(fallback[: int(top_k) - len(texts)])
            else:
                texts = (same_task + fallback)[: int(top_k)]
            retrieved.append(texts)
        return retrieved

    def retrieve_aggregates(self, query_embeddings: torch.Tensor | np.ndarray, sample_ids, task_names, top_k: int):
        if torch.is_tensor(query_embeddings):
            query_embeddings = query_embeddings.detach().float().cpu().numpy()
        queries = _normalize_rows(np.asarray(query_embeddings, dtype=np.float32))
        batch_size = int(queries.shape[0])
        if len(self.entries) == 0 or int(top_k) <= 0:
            zeros = np.zeros((batch_size, self.dim), dtype=np.float32)
            zero_scalar = np.zeros((batch_size,), dtype=np.float32)
            return {
                "context": zeros,
                "answer": zeros,
                "reward": zero_scalar,
                "similarity": zero_scalar,
                "count": zero_scalar.astype(np.int64),
            }

        overfetch = min(len(self.entries), max(int(top_k) * 16, int(top_k) + 8))
        scores, indices = self._index.search(queries, max(overfetch, int(top_k)))
        aggregated_context = np.zeros((batch_size, self.dim), dtype=np.float32)
        aggregated_answer = np.zeros((batch_size, self.dim), dtype=np.float32)
        aggregated_reward = np.zeros((batch_size,), dtype=np.float32)
        aggregated_similarity = np.zeros((batch_size,), dtype=np.float32)
        counts = np.zeros((batch_size,), dtype=np.int64)

        for row, (sample_id, task_name) in enumerate(zip(sample_ids, task_names)):
            same_task = []
            fallback = []
            for col, idx in enumerate(indices[row].tolist()):
                if idx < 0 or idx >= len(self.entries):
                    continue
                entry = self.entries[idx]
                if entry.id == str(sample_id):
                    continue
                item = (idx, float(scores[row][col]))
                if entry.task_name == str(task_name):
                    same_task.append(item)
                else:
                    fallback.append(item)
            selected = same_task[: int(top_k)] if self.same_task_first else []
            if len(selected) < int(top_k):
                pool = fallback if self.same_task_first else same_task + fallback
                selected.extend(pool[: int(top_k) - len(selected)])
            if not selected and not self.same_task_first:
                selected = (same_task + fallback)[: int(top_k)]
            if not selected:
                continue

            selected_indices = np.asarray([idx for idx, _ in selected], dtype=np.int64)
            selected_scores = np.asarray([score for _, score in selected], dtype=np.float32)
            selected_scores = selected_scores - selected_scores.max()
            weights = np.exp(selected_scores)
            weights = weights / np.maximum(weights.sum(), 1e-12)

            aggregated_context[row] = (self.embeddings[selected_indices] * weights[:, None]).sum(axis=0)
            aggregated_answer[row] = (self.answer_embeddings[selected_indices] * weights[:, None]).sum(axis=0)
            aggregated_reward[row] = float((self.rewards[selected_indices] * weights).sum())
            aggregated_similarity[row] = float((np.asarray([score for _, score in selected], dtype=np.float32) * weights).sum())
            counts[row] = len(selected_indices)

        return {
            "context": aggregated_context,
            "answer": aggregated_answer,
            "reward": aggregated_reward,
            "similarity": aggregated_similarity,
            "count": counts,
        }


def build_answer_only_inputs(processor, answers, max_text_len):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The selected processor does not expose a tokenizer.")
    texts = [f"Assistant: {str(answer).strip()}" for answer in answers]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_text_len,
        add_special_tokens=True,
    )
    packed = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value):
            packed[key] = value
    return packed

    def augment_prompts(self, prompts, retrieved_texts):
        return [
            _compose_prior_prompt(prompt, texts, self.prior_prefix)
            for prompt, texts in zip(prompts, retrieved_texts)
        ]
