from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..retriever import BaseRetriever
from ..types import CandidateConfig, DefenseResult, RetrievalHit
from ..utils import normalize_text


class RAGDefender:
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def apply(self, question: str, hits: list[RetrievalHit], cfg: CandidateConfig) -> DefenseResult:
        if len(hits) <= 1:
            return DefenseResult(safe_hits=hits, adversarial_hits=[], n_adv_estimate=0)

        texts = [h.passage.text for h in hits]
        embeddings = self.retriever.encode_texts(texts)
        sim = cosine_similarity(embeddings)

        if cfg.grouping == "clustering":
            n_adv = self._estimate_nadv_clustering(texts, embeddings, cfg.tfidf_m)
            scores = self._rank_clustering_scores(texts, sim, cfg.tfidf_m)
        elif cfg.grouping == "concentration":
            n_adv = self._estimate_nadv_concentration(sim)
            scores = self._rank_concentration_scores(sim)
        else:
            raise ValueError(f"Unknown grouping mode: {cfg.grouping}")

        n_adv = max(0, min(n_adv, len(hits) - 1))
        top_idx = set(np.argsort(scores)[::-1][:n_adv].tolist()) if n_adv > 0 else set()

        safe_hits, adv_hits = [], []
        for idx, hit in enumerate(hits):
            if idx in top_idx:
                adv_hits.append(hit)
            else:
                safe_hits.append(hit)

        return DefenseResult(
            safe_hits=safe_hits,
            adversarial_hits=adv_hits,
            n_adv_estimate=n_adv,
            metadata={
                "scores": [float(x) for x in scores],
                "grouping": cfg.grouping,
            },
        )

    def _estimate_nadv_clustering(self, texts: list[str], embeddings: np.ndarray, tfidf_m: int) -> int:
        clustering = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
        labels = clustering.fit_predict(embeddings)
        label_counts = Counter(labels.tolist())
        nmin = min(label_counts.values())

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        tfidf_scores = np.asarray(matrix.sum(axis=0)).reshape(-1)
        vocab = np.array(vectorizer.get_feature_names_out())
        top_terms = set(vocab[np.argsort(tfidf_scores)[::-1][: max(1, tfidf_m)]])
        n_tfidf = 0
        for text in texts:
            toks = set(normalize_text(text).split())
            if sum(1 for term in top_terms if term in toks) > max(1, len(top_terms) // 2):
                n_tfidf += 1
        if n_tfidf <= len(texts) / 2:
            return nmin
        return len(texts) - nmin

    def _rank_clustering_scores(self, texts: list[str], sim: np.ndarray, tfidf_m: int) -> np.ndarray:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        tfidf_scores = np.asarray(matrix.sum(axis=0)).reshape(-1)
        vocab = np.array(vectorizer.get_feature_names_out())
        top_terms = set(vocab[np.argsort(tfidf_scores)[::-1][: max(1, tfidf_m)]])
        scores = []
        for row_id, text in enumerate(texts):
            toks = set(normalize_text(text).split())
            overlap = sum(1 for term in top_terms if term in toks) / max(1, len(top_terms))
            avg_sim = float((sim[row_id].sum() - 1.0) / max(1, len(texts) - 1))
            scores.append(0.6 * avg_sim + 0.4 * overlap)
        return np.array(scores)

    def _estimate_nadv_concentration(self, sim: np.ndarray) -> int:
        means = []
        medians = []
        for i in range(sim.shape[0]):
            others = np.delete(sim[i], i)
            means.append(float(np.mean(others)))
            medians.append(float(np.median(others)))
        global_mean = float(np.mean(means))
        global_median = float(np.median(medians))
        return int(sum(1 for m, d in zip(means, medians) if m > global_mean and d > global_median))

    def _rank_concentration_scores(self, sim: np.ndarray) -> np.ndarray:
        scores = []
        for i in range(sim.shape[0]):
            others = np.delete(sim[i], i)
            scores.append(float(np.mean(others) + np.median(others)))
        return np.array(scores)
