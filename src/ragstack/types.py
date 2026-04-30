from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Passage:
    passage_id: str
    text: str
    source: str = "kb"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAExample:
    example_id: str
    question: str
    gold_answer: str | None = None
    target_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalHit:
    passage: Passage
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passage"] = self.passage.to_dict()
        return data


@dataclass
class CandidateConfig:
    mode: str = "none"
    grouping: str = "clustering"
    tfidf_m: int = 6
    similarity_threshold: float = 0.15
    aggregation: str = "keyword"
    alpha: float = 0.5
    beta: int = 3
    trigger: str = "Nadv>=tau"
    tau: int = 2
    s: int = 3

    def fingerprint(self) -> str:
        return (
            f"mode={self.mode}|grouping={self.grouping}|tfidf_m={self.tfidf_m}|"
            f"similarity_threshold={self.similarity_threshold:.3f}|alpha={self.alpha:.3f}|"
            f"beta={self.beta}|trigger={self.trigger}|tau={self.tau}|s={self.s}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefenseResult:
    safe_hits: list[RetrievalHit]
    adversarial_hits: list[RetrievalHit]
    n_adv_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe_hits": [h.to_dict() for h in self.safe_hits],
            "adversarial_hits": [h.to_dict() for h in self.adversarial_hits],
            "n_adv_estimate": self.n_adv_estimate,
            "metadata": self.metadata,
        }


@dataclass
class EvalRecord:
    example_id: str
    question: str
    split: str
    mode: str
    response: str
    gold_answer: str | None
    target_answer: str | None
    retrieved_hits: list[dict[str, Any]]
    safe_hits: list[dict[str, Any]]
    adversarial_hits: list[dict[str, Any]]
    gate_fired: bool
    llm_calls: int
    latency_sec: float
    poison_in_context: bool
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchRoundSummary:
    round_id: int
    candidate: CandidateConfig
    metrics: dict[str, Any]
    artifact_dir: str
