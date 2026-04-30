from pathlib import Path
import argparse
import csv
import gzip
import json
import random
import re

DEFAULT_SEED = 7
DEFAULT_N_CLEAN = 12
DEFAULT_N_TARGET = 6
DEFAULT_MAX_MATCHED_PASSAGES = 180
DEFAULT_MAX_QUERY_RELEVANT = 120
DEFAULT_MAX_RANDOM_NEGS = 120

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
    "is", "was", "are", "were", "who", "what", "when", "where",
    "which", "how", "why", "did", "does", "do", "with", "by",
    "from", "that", "this", "these", "those", "it", "as", "at",
    "be", "been", "being", "into", "about", "after", "before",
    "than", "then", "also", "can", "could", "would", "should"
}


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reduced NQ + DPR dataset for poisoned RAG experiments.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-clean", type=int, default=DEFAULT_N_CLEAN)
    parser.add_argument("--n-target", type=int, default=DEFAULT_N_TARGET)
    parser.add_argument("--max-matched-passages", type=int, default=DEFAULT_MAX_MATCHED_PASSAGES)
    parser.add_argument("--max-query-relevant", type=int, default=DEFAULT_MAX_QUERY_RELEVANT)
    parser.add_argument("--max-random-negs", type=int, default=DEFAULT_MAX_RANDOM_NEGS)
    parser.add_argument(
        "--nq-path",
        type=str,
        default="external/natural-questions/nq_open/NQ-open.dev.jsonl",
    )
    parser.add_argument(
        "--wiki-path",
        type=str,
        default="external/dpr_wiki/psgs_w100.tsv.gz",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    nq_path = Path(args.nq_path)
    wiki_path = Path(args.wiki_path)

    if not nq_path.exists():
        raise FileNotFoundError(f"Missing file: {nq_path}")
    if not wiki_path.exists():
        raise FileNotFoundError(f"Missing file: {wiki_path}")

    # -----------------------------
    # 1) Load NQ-open dev questions
    # -----------------------------
    rows: list[dict] = []
    with nq_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            answers = ex.get("answer", [])
            ans = answers[0] if isinstance(answers, list) and answers else None
            if ans:
                rows.append({
                    "question": ex["question"],
                    "gold_answer": ans
                })

    if len(rows) < args.n_clean + args.n_target:
        raise ValueError(
            f"Not enough examples in {nq_path}. "
            f"Needed {args.n_clean + args.n_target}, found {len(rows)}"
        )

    rng.shuffle(rows)
    rows = rows[: args.n_clean + args.n_target]

    clean_rows = rows[:args.n_clean]
    target_rows = rows[args.n_clean:args.n_clean + args.n_target]

    # --------------------------------------------
    # 2) Build clean set and target attack set
    # --------------------------------------------
    target_gold_answers = [x["gold_answer"] for x in target_rows]
    rotated = target_gold_answers[1:] + target_gold_answers[:1]

    clean_out: list[dict] = []
    for i, ex in enumerate(clean_rows):
        clean_out.append({
            "example_id": f"clean-{i:05d}",
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "target_answer": None,
            "metadata": {"source": "nq_open_dev"}
        })

    target_out: list[dict] = []
    for i, ex in enumerate(target_rows):
        wrong = rotated[i]
        if wrong.strip().lower() == ex["gold_answer"].strip().lower():
            wrong = target_gold_answers[(i + 2) % len(target_gold_answers)]
        target_out.append({
            "example_id": f"target-{i:05d}",
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "target_answer": wrong,
            "metadata": {"source": "nq_open_dev"}
        })

    all_examples = clean_out + target_out

    # --------------------------------------------
    # 3) Build answer strings and query tokens
    # --------------------------------------------
    answer_strings = {
        x["gold_answer"].strip().lower()
        for x in all_examples
        if x["gold_answer"] and len(x["gold_answer"].strip()) >= 3
    }

    query_token_sets = [tokenize(x["question"]) for x in all_examples]

    # --------------------------------------------
    # 4) Build passage pool:
    #    - answer-match passages
    #    - query-relevant distractors
    #    - random negatives
    # --------------------------------------------
    matched: list[dict] = []
    query_relevant: list[dict] = []
    negatives: list[dict] = []
    seen_ids: set[str] = set()

    with gzip.open(wiki_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            pid = str(row["id"])
            if pid in seen_ids:
                continue

            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            if not text:
                continue

            full_text = f"{title}. {text}" if title else text
            low = full_text.lower()
            toks = tokenize(full_text)

            has_answer = any(ans in low for ans in answer_strings)
            max_query_overlap = max((len(toks & qtok) for qtok in query_token_sets), default=0)

            if has_answer:
                matched.append({
                    "passage_id": f"wiki-{pid}",
                    "text": full_text,
                    "source": "dpr_psgs_w100",
                    "metadata": {
                        "title": title,
                        "kind": "answer_match"
                    }
                })
                seen_ids.add(pid)

            elif max_query_overlap >= 2 and len(query_relevant) < args.max_query_relevant:
                query_relevant.append({
                    "passage_id": f"wiki-{pid}",
                    "text": full_text,
                    "source": "dpr_psgs_w100",
                    "metadata": {
                        "title": title,
                        "kind": "query_relevant",
                        "overlap": max_query_overlap
                    }
                })
                seen_ids.add(pid)

            else:
                if len(negatives) < args.max_random_negs and rng.random() < 0.0015:
                    negatives.append({
                        "passage_id": f"wiki-{pid}",
                        "text": full_text,
                        "source": "dpr_psgs_w100",
                        "metadata": {
                            "title": title,
                            "kind": "random_negative"
                        }
                    })
                    seen_ids.add(pid)

            if (
                len(matched) >= args.max_matched_passages and
                len(query_relevant) >= args.max_query_relevant and
                len(negatives) >= args.max_random_negs
            ):
                break

    passages_out = (
        matched[:args.max_matched_passages] +
        query_relevant[:args.max_query_relevant] +
        negatives[:args.max_random_negs]
    )
    rng.shuffle(passages_out)

    # --------------------------------------------
    # 5) Save outputs
    # --------------------------------------------
    write_jsonl(root / "nq_passages.jsonl", passages_out)
    write_jsonl(root / "nq_targets.jsonl", target_out)
    write_jsonl(root / "nq_clean.jsonl", clean_out)

    build_info = {
        "seed": args.seed,
        "dataset": "Natural Questions Open",
        "split": "dev",
        "knowledge_base": "DPR Wikipedia passages (psgs_w100.tsv.gz)",
        "retrieval_corpus_summary": {
            "answer_match_passages": len(matched[:args.max_matched_passages]),
            "query_relevant_passages": len(query_relevant[:args.max_query_relevant]),
            "random_negative_passages": len(negatives[:args.max_random_negs]),
            "total_passages": len(passages_out),
        },
        "question_summary": {
            "n_clean": len(clean_out),
            "n_target": len(target_out),
        },
        "paths": {
            "passages": str(root / "nq_passages.jsonl"),
            "targets": str(root / "nq_targets.jsonl"),
            "clean": str(root / "nq_clean.jsonl"),
        },
    }

    with (root / "nq_build_info.json").open("w", encoding="utf-8") as f:
        json.dump(build_info, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(passages_out)} passages")
    print(f"wrote {len(target_out)} target examples")
    print(f"wrote {len(clean_out)} clean examples")
    print(f"answer-match passages: {len(matched[:args.max_matched_passages])}")
    print(f"query-relevant passages: {len(query_relevant[:args.max_query_relevant])}")
    print(f"random negatives: {len(negatives[:args.max_random_negs])}")
    print(f"metadata file: {root / 'nq_build_info.json'}")


if __name__ == "__main__":
    main()