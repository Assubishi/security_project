from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import CandidateConfig


@dataclass
class DataConfig:
    passages_path: str
    targets_path: str
    clean_path: str


@dataclass
class RetrieverConfig:
    kind: str = "tfidf"
    model_name: str = "facebook/contriever-msmarco"
    top_k: int = 5
    batch_size: int = 16
    use_gpu: bool = False


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 256
    reasoning_effort: str | None = None


@dataclass
class AttackConfig:
    enabled: bool = True
    n_injected: int = 5
    use_llm_generation: bool = True


@dataclass
class SearchConfig:
    rounds: int = 2
    top_k_keep: int = 4
    min_clean_accuracy: float = 0.30
    max_avg_llm_calls: float = 8.0
    improvement_epsilon: float = 0.01
    stagnation_rounds: int = 2
    initial_groupings: list[str] = field(default_factory=lambda: ["clustering", "concentration"])
    initial_alphas: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    initial_betas: list[int] = field(default_factory=lambda: [2, 3, 4])
    initial_taus: list[int] = field(default_factory=lambda: [1, 2, 3])
    initial_s_values: list[int] = field(default_factory=lambda: [2, 3, 4])
    proposer_num_candidates: int = 10


@dataclass
class EvalConfig:
    experiment_name: str = "demo"
    seed: int = 7
    output_dir: str = "results"
    scoring: str = "substring"
    data: DataConfig | None = None
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    default_candidate: CandidateConfig = field(default_factory=CandidateConfig)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> EvalConfig:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text())
    if overrides:
        data = _deep_merge(data, overrides)

    data_cfg = DataConfig(**data["data"])
    retriever_cfg = RetrieverConfig(**data.get("retriever", {}))
    llm_cfg = LLMConfig(**data.get("llm", {}))
    attack_cfg = AttackConfig(**data.get("attack", {}))
    search_cfg = SearchConfig(**data.get("search", {}))
    candidate_cfg = CandidateConfig(**data.get("default_candidate", {}))
    cfg = EvalConfig(
        experiment_name=data.get("experiment_name", cfg_path.stem),
        seed=data.get("seed", 7),
        output_dir=data.get("output_dir", "results"),
        scoring=data.get("scoring", "substring"),
        data=data_cfg,
        retriever=retriever_cfg,
        llm=llm_cfg,
        attack=attack_cfg,
        search=search_cfg,
        default_candidate=candidate_cfg,
    )
    return cfg
