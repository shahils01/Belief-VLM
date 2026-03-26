import hashlib
import math
import re
import warnings
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from data_loading import _get_first, _normalize_text, _stable_fold


def _tokenize_text(text: str):
    return re.findall(r"[A-Za-z0-9_<>:-]+", (text or "").lower())


class _HashingTextEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _tokenize_text(text)
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest[:8], 16) % self.dim
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                matrix[row, idx] += sign
            matrix[row] /= max(math.sqrt(len(tokens)), 1.0)
        encoded = torch.from_numpy(matrix)
        return F.normalize(encoded, dim=-1)


class TextEmbedder:
    def __init__(self, model_name: str = "", hashing_dim: int = 512):
        self.model_name = model_name or ""
        self.hashing_dim = int(hashing_dim)
        self._st_model = None
        if self.model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(self.model_name)
            except Exception as exc:
                self._st_model = None
                warnings.warn(
                    f"Could not load sentence-transformers model '{self.model_name}'. "
                    f"Falling back to hashing-based text embeddings. Original error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "No retrieval_embedder_model was provided. Falling back to hashing-based text embeddings.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._fallback = _HashingTextEmbedder(dim=self.hashing_dim)

    @property
    def dim(self) -> int:
        if self._st_model is not None:
            return int(self._st_model.get_sentence_embedding_dimension())
        return self.hashing_dim

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if self._st_model is not None:
            encoded = self._st_model.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
            return encoded.detach().cpu().float()
        return self._fallback.encode(texts)


def default_belief_text(record: dict) -> str:
    task_name = str(record.get("task_name", "unknown")).replace("_", " ")
    question = _normalize_text(_get_first(record, ["question", "prompt", "instruction", "query"]))
    answer = _normalize_text(_get_first(record, ["answer", "response", "label"]))
    choices = _get_first(record, ["choices", "options", "answer_options"])
    correct_idx = _get_first(record, ["correct_idx", "answer_idx", "label_idx"])
    if not answer and isinstance(choices, (list, tuple)) and correct_idx not in (None, ""):
        idx = int(correct_idx)
        if 1 <= idx <= len(choices):
            idx -= 1
        if 0 <= idx < len(choices):
            answer = _normalize_text(choices[idx])
    belief = _get_first(record, ["belief", "belief_text", "prior", "rationale", "explanation"])
    belief = _normalize_text(belief)
    if belief:
        return belief
    if answer:
        return f"Task: {task_name}. In a similar example, the best answer was: {answer}"
    return f"Task: {task_name}. Similar question: {question}"


def build_query_text(task_name: str, prompt: str) -> str:
    task_part = f"Task: {str(task_name).replace('_', ' ')}."
    return f"{task_part} Question: {_normalize_text(prompt)}"


@dataclass
class RetrievedBeliefBatch:
    candidate_ids: List[List[str]]
    candidate_texts: List[List[str]]
    candidate_embeddings: torch.Tensor
    candidate_counts: torch.Tensor


class BeliefVectorDB:
    def __init__(self, items: list, embedder: TextEmbedder):
        self.items = items
        self.embedder = embedder
        self.item_texts = [item["belief_text"] for item in self.items]
        self.item_queries = [item["query_text"] for item in self.items]
        self.item_embeddings = self.embedder.encode(self.item_queries)

    @classmethod
    def from_records(cls, records: Sequence[dict], args, split_name: str = "train"):
        val_ratio = max(0.0, min(0.5, float(args.val_ratio)))
        items = []
        for idx, record in enumerate(records):
            sample_id = str(_get_first(record, [args.id_column, "id", "sample_id", "uid", "video_id"]) or idx)
            in_val = _stable_fold(sample_id, args.seed) < val_ratio if val_ratio > 0 else False
            keep = (not in_val) if split_name == getattr(args, "train_split", "train") else in_val
            if not keep:
                continue
            question = _normalize_text(_get_first(record, [args.question_column, "question", "prompt", "instruction", "query"]))
            items.append(
                {
                    "id": sample_id,
                    "task_name": str(record.get("task_name", "unknown")),
                    "belief_text": default_belief_text(record),
                    "query_text": build_query_text(record.get("task_name", "unknown"), question),
                }
            )
        embedder = TextEmbedder(
            model_name=getattr(args, "retrieval_embedder_model", ""),
            hashing_dim=int(getattr(args, "retrieval_hash_dim", 512)),
        )
        return cls(items=items, embedder=embedder)

    def retrieve(self, task_names: Sequence[str], prompts: Sequence[str], top_k: int, exclude_ids: Sequence[str] | None = None):
        top_k = max(int(top_k), 1)
        query_texts = [build_query_text(task_name, prompt) for task_name, prompt in zip(task_names, prompts)]
        query_embeddings = self.embedder.encode(query_texts)
        sim = torch.matmul(query_embeddings, self.item_embeddings.T)

        candidate_ids = []
        candidate_texts = []
        candidate_embs = []
        candidate_counts = []
        exclude_ids = list(exclude_ids) if exclude_ids is not None else [None] * len(query_texts)

        for row_idx, exclude_id in enumerate(exclude_ids):
            scores = sim[row_idx].clone()
            if exclude_id is not None:
                for item_idx, item in enumerate(self.items):
                    if item["id"] == exclude_id:
                        scores[item_idx] = -1e9
            k = min(top_k, scores.numel())
            top_scores, top_indices = torch.topk(scores, k=k, dim=0)
            valid_mask = top_scores > -1e8
            row_ids = []
            row_texts = []
            row_embs = []
            for item_idx, is_valid in zip(top_indices.tolist(), valid_mask.tolist()):
                if not is_valid:
                    continue
                row_ids.append(self.items[item_idx]["id"])
                row_texts.append(self.items[item_idx]["belief_text"])
                row_embs.append(self.item_embeddings[item_idx])
            if not row_texts:
                row_ids = [self.items[0]["id"]]
                row_texts = [self.items[0]["belief_text"]]
                row_embs = [self.item_embeddings[0]]
            candidate_ids.append(row_ids)
            candidate_texts.append(row_texts)
            candidate_counts.append(len(row_texts))
            while len(row_embs) < top_k:
                row_embs.append(row_embs[-1])
            candidate_embs.append(torch.stack(row_embs[:top_k], dim=0))

        return RetrievedBeliefBatch(
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            candidate_embeddings=torch.stack(candidate_embs, dim=0),
            candidate_counts=torch.tensor(candidate_counts, dtype=torch.long),
        )
