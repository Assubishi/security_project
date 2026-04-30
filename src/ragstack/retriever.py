from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import RetrieverConfig
from .types import Passage, RetrievalHit


class BaseRetriever(ABC):
    @abstractmethod
    def build(self, passages: list[Passage]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, question: str, top_k: int) -> list[RetrievalHit]:
        raise NotImplementedError

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.passages: list[Passage] = []
        self.matrix = None

    def build(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.matrix = self.vectorizer.fit_transform([p.text for p in passages])

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalHit]:
        if self.matrix is None:
            raise RuntimeError("Retriever not built.")
        k = top_k or self.config.top_k
        q = self.vectorizer.transform([question])
        scores = cosine_similarity(q, self.matrix)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        hits = []
        for rank, idx in enumerate(top_idx, start=1):
            hits.append(RetrievalHit(self.passages[idx], float(scores[idx]), rank))
        return hits

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.matrix is None:
            temp = self.vectorizer.fit_transform(texts)
            return temp.toarray()
        return self.vectorizer.transform(texts).toarray()


class ContrieverRetriever(BaseRetriever):
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.passages: list[Passage] = []
        self.index = None
        self.embeddings = None
        self.tokenizer = None
        self.model = None

    def _lazy_init(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for ContrieverRetriever. Install requirements.txt first."
            ) from exc
        self.torch = torch
        self.device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
        self.model.eval()

    def _mean_pool(self, outputs, mask):
        token_embeddings = outputs.last_hidden_state
        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked = token_embeddings * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self._lazy_init()
        all_vecs = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with self.torch.no_grad():
                outputs = self.model(**toks)
                pooled = self._mean_pool(outputs, toks["attention_mask"])
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vecs.append(pooled.detach().cpu().numpy())
        return np.vstack(all_vecs)

    def build(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.embeddings = self.encode_texts([p.text for p in passages])

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalHit]:
        if self.embeddings is None:
            raise RuntimeError("Retriever not built.")
        k = top_k or self.config.top_k
        q = self.encode_texts([question])[0]
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:k]
        hits = []
        for rank, idx in enumerate(top_idx, start=1):
            hits.append(RetrievalHit(self.passages[idx], float(scores[idx]), rank))
        return hits


@dataclass
class RetrieverFactory:
    config: RetrieverConfig

    def create(self) -> BaseRetriever:
        if self.config.kind == "tfidf":
            return TfidfRetriever(self.config)
        if self.config.kind == "contriever":
            return ContrieverRetriever(self.config)
        raise ValueError(f"Unsupported retriever kind: {self.config.kind}")
