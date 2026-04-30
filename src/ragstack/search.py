from __future__ import annotations

import itertools
import json
from pathlib import Path

from .config import EvalConfig
from .evaluator import Evaluator
from .types import CandidateConfig
from .utils import ensure_dir, write_json


SEARCH_SYSTEM_PROMPT = """You are tuning a defense configuration for a RAG poisoning benchmark.
You must propose diverse new configurations as JSON.
Return a JSON array of objects matching this schema:
{mode, grouping, tfidf_m, alpha, beta, trigger, tau, s}
Use mode=stacked_gated unless there is a strong reason otherwise."""


class CandidateSearch:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.evaluator = Evaluator(cfg)
        self.llm = self.evaluator.llm

    def run(self, out_dir: str | Path) -> dict:
        out_dir = ensure_dir(out_dir)
        pool = self._initial_pool()
        history = []
        best_asr = float("inf")
        stale_rounds = 0

        for round_id in range(1, self.cfg.search.rounds + 1):
            round_results = []
            for idx, candidate in enumerate(pool):
                artifact_dir = ensure_dir(Path(out_dir) / f"round_{round_id:02d}" / f"cand_{idx:02d}")
                result = self.evaluator.evaluate(candidate, artifact_dir)
                round_results.append(
                    {
                        "round_id": round_id,
                        "candidate": candidate.to_dict(),
                        "fingerprint": candidate.fingerprint(),
                        "metrics": result["metrics"],
                        "artifact_dir": str(artifact_dir),
                    }
                )
            history.extend(round_results)

            survivors = self._select_survivors(round_results)
            current_best_asr = survivors[0]["metrics"]["attack_success_rate"] if survivors else float("inf")
            if best_asr - current_best_asr < self.cfg.search.improvement_epsilon:
                stale_rounds += 1
            else:
                stale_rounds = 0
                best_asr = current_best_asr
            if stale_rounds >= self.cfg.search.stagnation_rounds:
                break
            pool = self._propose_next_round(survivors)
            if not pool:
                break

        summary = {
            "history": history,
            "best": sorted(history, key=lambda x: (x["metrics"]["attack_success_rate"], -x["metrics"]["clean_accuracy"]))[:5],
        }
        write_json(Path(out_dir) / "search_summary.json", summary)
        return summary

    def _initial_pool(self) -> list[CandidateConfig]:
        candidates = [CandidateConfig(mode="none"), CandidateConfig(mode="ragdefender"), CandidateConfig(mode="robustrag")]
        for grouping, alpha, beta, tau, s in itertools.product(
            self.cfg.search.initial_groupings,
            self.cfg.search.initial_alphas,
            self.cfg.search.initial_betas,
            self.cfg.search.initial_taus,
            self.cfg.search.initial_s_values,
        ):
            candidates.append(
                CandidateConfig(
                    mode="stacked_gated",
                    grouping=grouping,
                    tfidf_m=6,
                    alpha=alpha,
                    beta=beta,
                    trigger="Nadv>=tau",
                    tau=tau,
                    s=s,
                )
            )
        uniq = {}
        for cand in candidates:
            uniq[cand.fingerprint()] = cand
        return list(uniq.values())

    def _select_survivors(self, results: list[dict]) -> list[dict]:
        eligible = [
            r for r in results
            if r["metrics"]["clean_accuracy"] >= self.cfg.search.min_clean_accuracy
            and r["metrics"]["avg_llm_calls"] <= self.cfg.search.max_avg_llm_calls
        ]
        if not eligible:
            eligible = results
        eligible = sorted(
            eligible,
            key=lambda x: (x["metrics"]["attack_success_rate"], -x["metrics"]["clean_accuracy"], x["metrics"]["avg_llm_calls"]),
        )
        survivors = []
        seen_groupings = set()
        for item in eligible:
            grouping = item["candidate"].get("grouping", "none")
            if grouping not in seen_groupings or len(survivors) < self.cfg.search.top_k_keep:
                survivors.append(item)
                seen_groupings.add(grouping)
            if len(survivors) >= self.cfg.search.top_k_keep:
                break
        return survivors

    def _propose_next_round(self, survivors: list[dict]) -> list[CandidateConfig]:
        if not survivors:
            return []
        summary_table = [
            {
                "candidate": s["candidate"],
                "metrics": s["metrics"],
            }
            for s in survivors
        ]
        user_prompt = (
            "Given these metrics and traces, propose 10 new cfgs by adjusting gating and keyword thresholds; avoid duplicates.\n"
            "Return JSON only.\n\n"
            + json.dumps(summary_table, indent=2)
        )
        raw = self.llm.generate(SEARCH_SYSTEM_PROMPT, user_prompt, expect_json=True)
        proposed = self._parse_candidate_array(raw)
        if not proposed:
            proposed = self._mutate_survivors(survivors)
        uniq = {}
        for cand in proposed:
            uniq[cand.fingerprint()] = cand
        return list(uniq.values())[: self.cfg.search.proposer_num_candidates]

    def _mutate_survivors(self, survivors: list[dict]) -> list[CandidateConfig]:
        out = []
        for item in survivors:
            cfg = CandidateConfig(**item["candidate"])
            out.extend(
                [
                    CandidateConfig(**{**cfg.to_dict(), "tau": max(1, cfg.tau - 1)}),
                    CandidateConfig(**{**cfg.to_dict(), "tau": min(5, cfg.tau + 1)}),
                    CandidateConfig(**{**cfg.to_dict(), "beta": max(1, cfg.beta - 1)}),
                    CandidateConfig(**{**cfg.to_dict(), "beta": min(5, cfg.beta + 1)}),
                    CandidateConfig(**{**cfg.to_dict(), "grouping": "concentration" if cfg.grouping == "clustering" else "clustering"}),
                ]
            )
        return out

    def _parse_candidate_array(self, raw: str) -> list[CandidateConfig]:
        raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]")
            if start < 0 or end <= start:
                return []
            payload = json.loads(raw[start : end + 1])
        candidates = []
        for item in payload:
            try:
                candidates.append(CandidateConfig(**item))
            except TypeError:
                continue
        return candidates
