from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

from .config import LLMConfig
from .utils import normalize_text


class BaseLLM(ABC):
    def __init__(self) -> None:
        self.call_count = 0

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, *, expect_json: bool = False) -> str:
        raise NotImplementedError

    def reset_call_count(self) -> None:
        self.call_count = 0


class MockLLM(BaseLLM):
    def generate(self, system_prompt: str, user_prompt: str, *, expect_json: bool = False) -> str:
        self.call_count += 1
        prompt = (system_prompt + "\n" + user_prompt).lower()
        if expect_json:
            if "extract a candidate answer" in prompt:
                text = _extract_last_context_block(user_prompt)
                answer = _best_sentence(text)
                keywords = _keywords(answer)
                return json.dumps({"candidate_answer": answer, "keywords": keywords})
            if "propose 10 new cfgs" in prompt:
                return json.dumps(
                    [
                        {
                            "mode": "stacked_gated",
                            "grouping": "clustering",
                            "tfidf_m": 6,
                            "alpha": 0.5,
                            "beta": 3,
                            "trigger": "Nadv>=tau",
                            "tau": 2,
                            "s": 3,
                        },
                        {
                            "mode": "stacked_gated",
                            "grouping": "concentration",
                            "tfidf_m": 6,
                            "alpha": 0.7,
                            "beta": 2,
                            "trigger": "|Rsafe|<=s",
                            "tau": 2,
                            "s": 2,
                        },
                    ]
                )
        if "extract a candidate answer" in prompt:
            text = _extract_last_context_block(user_prompt)
            answer = _best_sentence(text)
            return json.dumps({"candidate_answer": answer, "keywords": _keywords(answer)})
        if "answer the question" in prompt or "final answer" in prompt:
            text = _extract_last_context_block(user_prompt)
            return _best_sentence(text)
        return "I do not know."


class OpenAIResponsesLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai package to use the OpenAI LLM backend.") from exc
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing API key in environment variable {config.api_key_env}.")
        self.client = OpenAI(api_key=api_key)
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str, *, expect_json: bool = False) -> str:
        self.call_count += 1
        extra: dict[str, Any] = {}
        if self.config.reasoning_effort:
            extra["reasoning"] = {"effort": self.config.reasoning_effort}
        response = self.client.responses.create(
            model=self.config.model,
            max_output_tokens=self.config.max_output_tokens,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            **extra,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text
        parts = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    parts.append(content.text)
        return "\n".join(parts)


class LLMFactory:
    def __init__(self, config: LLMConfig):
        self.config = config

    def create(self) -> BaseLLM:
        if self.config.provider == "mock":
            return MockLLM()
        if self.config.provider == "openai":
            return OpenAIResponsesLLM(self.config)
        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")


def _extract_last_context_block(prompt: str) -> str:
    if "Context:" not in prompt:
        return prompt
    return prompt.rsplit("Context:", 1)[-1].strip()


def _best_sentence(text: str) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return text.strip()[:200]
    non_query = [s for s in sentences if not s.lower().startswith("question")]
    if non_query:
        return max(non_query, key=lambda s: len(s.split()))
    return sentences[0]


def _keywords(text: str) -> list[str]:
    tokens = [t for t in normalize_text(text).split() if len(t) > 3]
    counts = Counter(tokens)
    return [tok for tok, _ in counts.most_common(5)]
