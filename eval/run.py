"""Measure retrieval and refusal against a golden set.

Run against an already-ingested corpus:

    python -m eval.run                    # retrieval metrics for every mode
    python -m eval.run --answers          # also ask the model (slow, costs calls)
    python -m eval.run --mode hybrid      # one mode only

Ground truth is a list of strings that must all appear in a chunk for it to
count as relevant, rather than a chunk id. Chunk ids change every time the
chunking strategy does, which would make the set need rewriting after exactly
the change it exists to evaluate.
"""

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.core import answering, database, retrieval
from app.providers import build_embedding_provider, build_llm_provider

DATASET = Path(__file__).parent / "dataset.json"
MODES = ("dense", "keyword", "hybrid")


@dataclass
class RetrievalMetrics:
    """Standard retrieval measures.

    hit@k is the practical one: whether the answer was in what the model got
    to read at all. MRR additionally rewards putting it near the top, which
    matters because early chunks get more of the model's attention.
    """

    cases: int = 0
    hits: int = 0
    reciprocal_ranks: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.cases if self.cases else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_ranks / self.cases if self.cases else 0.0


def is_relevant(text: str, required: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in required)


def rank_of_relevant(hits, required: list[str]) -> int | None:
    for position, hit in enumerate(hits, start=1):
        if is_relevant(hit.text, required):
            return position
    return None


def evaluate_retrieval(conn, provider, cases, mode: str, top_k: int, settings):
    metrics = RetrievalMetrics()
    misses = []

    for case in cases:
        if case.get("should_refuse"):
            continue
        hits, _ = retrieval.search(
            conn, provider, case["question"], top_k=top_k, mode=mode,
            candidates=settings.search_candidates,
            dense_weight=settings.search_dense_weight,
            keyword_weight=settings.search_keyword_weight,
        )
        rank = rank_of_relevant(hits, case["relevant_if_contains"])

        metrics.cases += 1
        if rank:
            metrics.hits += 1
            metrics.reciprocal_ranks += 1.0 / rank
        else:
            misses.append(case["question"])

    return metrics, misses


def evaluate_answers(conn, embedder, llm, cases, mode: str, top_k: int, settings):
    """Ask for real. Slow and costs API calls, so it is opt-in."""
    results = {"answered": 0, "answerable": 0, "refused_correctly": 0,
               "should_refuse": 0, "wrongly_refused": [], "wrongly_answered": []}

    for case in cases:
        hits, _ = retrieval.search(
            conn, embedder, case["question"], top_k=top_k, mode=mode,
            candidates=settings.search_candidates,
            dense_weight=settings.search_dense_weight,
            keyword_weight=settings.search_keyword_weight,
        )
        sources = answering.build_sources(hits)
        if not sources:
            answer = answering.Answer("", [], [], True, llm.model)
        else:
            raw = llm.generate(
                answering.SYSTEM_PROMPT,
                answering.build_prompt(case["question"], sources),
            )
            answer = answering.assemble(raw, sources, llm.model)

        if case.get("should_refuse"):
            results["should_refuse"] += 1
            if answer.refused:
                results["refused_correctly"] += 1
            else:
                results["wrongly_answered"].append(case["question"])
        else:
            results["answerable"] += 1
            if answer.refused:
                results["wrongly_refused"].append(case["question"])
            else:
                results["answered"] += 1

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, help="evaluate one mode only")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answers", action="store_true",
                        help="also generate answers and score refusals")
    args = parser.parse_args()

    settings = get_settings()
    dataset = json.loads(DATASET.read_text())
    cases = dataset["cases"]
    retrievable = [c for c in cases if not c.get("should_refuse")]

    conn = database.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    embedder = build_embedding_provider(settings)

    indexed = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if indexed == 0:
        print("Nothing indexed. Upload the corpus and POST /documents/{id}/ingest.")
        return 1

    print(f"corpus: {dataset['corpus']}  ({indexed} chunks indexed)")
    print(f"cases: {len(retrievable)} retrieval, "
          f"{len(cases) - len(retrievable)} expected refusals, top_k={args.top_k}\n")

    print(f"{'mode':<10} {'hit@k':>8} {'MRR':>8}")
    print("-" * 28)
    all_misses = {}
    for mode in ([args.mode] if args.mode else MODES):
        metrics, misses = evaluate_retrieval(
            conn, embedder, cases, mode, args.top_k, settings
        )
        all_misses[mode] = misses
        print(f"{mode:<10} {metrics.hit_rate:>8.3f} {metrics.mrr:>8.3f}")

    for mode, misses in all_misses.items():
        if misses:
            print(f"\n{mode} missed:")
            for question in misses:
                print(f"  - {question}")

    if args.answers:
        mode = args.mode or settings.search_mode
        print(f"\ngenerating answers (mode={mode})...")
        llm = build_llm_provider(settings)
        results = evaluate_answers(
            conn, embedder, llm, cases, mode, args.top_k, settings
        )
        print(f"\n{'answered when answerable':<30} "
              f"{results['answered']}/{results['answerable']}")
        print(f"{'refused when unanswerable':<30} "
              f"{results['refused_correctly']}/{results['should_refuse']}")
        for question in results["wrongly_refused"]:
            print(f"  refused but should not have: {question}")
        for question in results["wrongly_answered"]:
            print(f"  answered but should have refused: {question}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
