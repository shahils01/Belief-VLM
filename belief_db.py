import warnings
from dataclasses import dataclass

import numpy as np


def _normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        return "\n".join(_normalize_text(item) for item in text if _normalize_text(item))
    return str(text).strip()


class _HashingTextEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def encode(self, texts):
        features = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [token for token in _normalize_text(text).lower().split() if token]
            if not tokens:
                continue
            for token in tokens:
                bucket = hash(token) % self.dim
                features[row, bucket] += 1.0
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return features / norms


class TextEmbedder:
    def __init__(self, model_name: str = "", hash_dim: int = 512):
        self.model_name = str(model_name or "").strip()
        self.dim = int(hash_dim)
        self._model = None
        self._fallback = _HashingTextEmbedder(dim=hash_dim)
        if self.model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                probe = self._model.encode(["probe"], normalize_embeddings=True)
                self.dim = int(probe.shape[-1])
            except Exception as exc:
                warnings.warn(
                    f"Failed to load sentence-transformers model '{self.model_name}'. "
                    f"Falling back to hashing-based text embeddings. Error: {exc}",
                    RuntimeWarning,
                )
                self._model = None
        else:
            warnings.warn(
                "No retrieval_embedder_model provided. Falling back to hashing-based text embeddings.",
                RuntimeWarning,
            )

    def encode(self, texts):
        clean = [_normalize_text(text) for text in texts]
        if self._model is not None:
            return np.asarray(self._model.encode(clean, normalize_embeddings=True), dtype=np.float32)
        return self._fallback.encode(clean)


def _get_first(record, keys):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def build_query_text(record, prompt: str):
    task_name = _normalize_text(record.get("task_name", ""))
    question = _normalize_text(
        _get_first(record, ["question", "prompt", "instruction", "query"]) or prompt
    )
    parts = []
    if task_name:
        parts.append(f"Task: {task_name}")
    if question:
        parts.append(f"Question: {question}")
    return "\n".join(parts)


def default_belief_text(record):
    answer = _normalize_text(_get_first(record, ["answer", "response", "label", "caption", "narration"]))
    if not answer:
        choices = _get_first(record, ["options", "choices", "answer_options"])
        correct_idx = _get_first(record, ["correct_idx", "answer_idx", "label_idx"])
        if isinstance(choices, (list, tuple)) and correct_idx not in (None, ""):
            try:
                idx = int(correct_idx)
                if 1 <= idx <= len(choices):
                    idx -= 1
                if 0 <= idx < len(choices):
                    answer = _normalize_text(choices[idx])
            except Exception:
                pass

    if not answer:
        return ""

    # Keep retrieved priors short so they act as a hint rather than a
    # full exemplar that can distract the VLM prompt.
    tokens = answer.replace("\n", " ").split()
    short_answer = " ".join(tokens[:12]).strip(" ,.;:")
    if not short_answer:
        return ""
    if len(tokens) > 12:
        short_answer = f"{short_answer}..."
    return f"Likely goal: {short_answer}."


@dataclass
class RetrievalResult:
    texts: list[str]
    scores: list[float]


class BeliefVectorDB:
    def __init__(self, entries, embeddings, embedder, prior_prefix: str = "Belief prior:"):
        self.entries = entries
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embedder = embedder
        self.prior_prefix = str(prior_prefix)

    @classmethod
    def from_records(cls, records, args):
        embedder = TextEmbedder(
            model_name=getattr(args, "retrieval_embedder_model", ""),
            hash_dim=getattr(args, "retrieval_hash_dim", 512),
        )
        entries = []
        queries = []
        for idx, record in enumerate(records):
            sample_id = _normalize_text(
                _get_first(record, [getattr(args, "id_column", "id"), "id", "sample_id", "uid", "video_id"]) or idx
            )
            belief_text = default_belief_text(record)
            if not belief_text:
                continue
            task_name = _normalize_text(record.get("task_name", ""))
            prompt = _normalize_text(_get_first(record, [getattr(args, "question_column", "question"), "question", "prompt", "instruction", "query"]))
            entries.append(
                {
                    "id": sample_id,
                    "task_name": task_name,
                    "belief_text": belief_text,
                    "query_text": build_query_text(record, prompt),
                }
            )
            queries.append(entries[-1]["query_text"])
        if not entries:
            return cls([], np.zeros((0, embedder.dim), dtype=np.float32), embedder, getattr(args, "db_prior_prefix", "Belief prior:"))
        embeddings = embedder.encode(queries)
        return cls(entries, embeddings, embedder, getattr(args, "db_prior_prefix", "Belief prior:"))

    def retrieve(self, record, prompt: str, sample_id: str, top_k: int = 1):
        if not self.entries or top_k <= 0:
            return RetrievalResult(texts=[], scores=[])
        query = self.embedder.encode([build_query_text(record, prompt)])[0]
        scores = self.embeddings @ query
        order = np.argsort(-scores)
        texts = []
        vals = []
        for idx in order.tolist():
            entry = self.entries[idx]
            if entry["id"] == str(sample_id):
                continue
            texts.append(entry["belief_text"])
            vals.append(float(scores[idx]))
            if len(texts) >= int(top_k):
                break
        return RetrievalResult(texts=texts, scores=vals)

    def augment_prompt(self, record, prompt: str, sample_id: str, top_k: int = 1):
        result = self.retrieve(record, prompt, sample_id, top_k=top_k)
        if not result.texts:
            return prompt
        prior_block = "\n".join(f"- {text}" for text in result.texts)
        return f"{self.prior_prefix}\n{prior_block}\n\n{prompt}"
