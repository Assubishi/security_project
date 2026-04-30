from __future__ import annotations

import json
from collections import Counter

from ..llm import BaseLLM
from ..types import CandidateConfig, RetrievalHit
from ..utils import normalize_text


ISOLATE_SYSTEM_PROMPT = """You are a secure answer extractor.
Read one passage in isolation.
Return JSON with keys candidate_answer and keywords.
The candidate answer should be short.
Keywords should be a short list of supporting content words.
If the passage is irrelevant, use candidate_answer = \"irrelevant\"."""

FINAL_SYSTEM_PROMPT = """You are a robust RAG answerer.
Use only the consolidated evidence.
If the evidence is conflicting, prefer the answer supported by more independent passages."""


class RobustRAG:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def answer(self, question: str, hits: list[RetrievalHit], cfg: CandidateConfig) -> tuple[str, dict]:
        isolated = []
        keyword_counter: Counter[str] = Counter()
        for hit in hits:
            user_prompt = (
                f"Question: {question}\n"
                f"Passage:\n{hit.passage.text}\n\n"
                f"Extract a candidate answer and a small keyword list. Return JSON only."
            )
            raw = self.llm.generate(ISOLATE_SYSTEM_PROMPT, user_prompt, expect_json=True)
            parsed = _parse_json_loose(raw)
            candidate = str(parsed.get("candidate_answer", "irrelevant"))
            keywords = [str(k) for k in parsed.get("keywords", [])]
            isolated.append({
                "passage_id": hit.passage.passage_id,
                "candidate_answer": candidate,
                "keywords": keywords,
                "text": hit.passage.text,
            })
            keyword_counter.update(normalize_text(" ".join(keywords)).split())

        n = len(hits)
        mu = min(max(1, int(round(cfg.alpha * n))), max(1, cfg.beta))
        stable_keywords = {k for k, v in keyword_counter.items() if v >= mu}

        consolidated_chunks = []
        for item in isolated:
            item_keywords = set(normalize_text(" ".join(item["keywords"])).split())
            if not stable_keywords or item_keywords & stable_keywords:
                consolidated_chunks.append(
                    f"[{item['passage_id']}] candidate_answer={item['candidate_answer']} | passage={item['text']}"
                )

        if not consolidated_chunks:
            consolidated_chunks = [f"[{h.passage.passage_id}] {h.passage.text}" for h in hits]

        user_prompt = (
            f"Question: {question}\n"
            f"Consolidated evidence:\n" + "\n".join(consolidated_chunks) + "\n\n"
            "Return the final answer in one short sentence."
        )
        response = self.llm.generate(FINAL_SYSTEM_PROMPT, user_prompt)
        metadata = {
            "isolated": isolated,
            "stable_keywords": sorted(stable_keywords),
            "mu": mu,
            "used_chunks": consolidated_chunks,
        }
        return response, metadata


def _parse_json_loose(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        return {"candidate_answer": raw, "keywords": []}
