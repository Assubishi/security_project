from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .evaluator import Evaluator
from .search import CandidateSearch
from .types import CandidateConfig
from .utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG poisoning defense project")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("evaluate", help="Run one evaluation")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--mode", default=None, choices=["none", "ragdefender", "robustrag", "stacked_gated"])
    eval_parser.add_argument("--grouping", default=None, choices=["clustering", "concentration"])
    eval_parser.add_argument("--tau", type=int, default=None)
    eval_parser.add_argument("--s", type=int, default=None)
    eval_parser.add_argument("--alpha", type=float, default=None)
    eval_parser.add_argument("--beta", type=int, default=None)
    eval_parser.add_argument("--out", default=None)

    search_parser = sub.add_parser("search", help="Run iterative candidate search")
    search_parser.add_argument("--config", required=True)
    search_parser.add_argument("--out", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "evaluate":
        candidate = cfg.default_candidate
        overrides = {}
        for key in ["mode", "grouping", "tau", "s", "alpha", "beta"]:
            value = getattr(args, key)
            if value is not None:
                overrides[key] = value
        candidate = CandidateConfig(**{**candidate.to_dict(), **overrides})
        out_dir = args.out or str(Path(cfg.output_dir) / cfg.experiment_name / candidate.fingerprint().replace("|", "__"))
        evaluator = Evaluator(cfg)
        result = evaluator.evaluate(candidate, ensure_dir(out_dir))
        print(result["metrics"])
        return

    if args.command == "search":
        out_dir = args.out or str(Path(cfg.output_dir) / cfg.experiment_name / "search")
        summary = CandidateSearch(cfg).run(ensure_dir(out_dir))
        print({"best": summary["best"][:3], "history_len": len(summary["history"])})
        return


if __name__ == "__main__":
    main()
