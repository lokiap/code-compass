"""Retrieval benchmark: Recall@k, MRR and nDCG over a hand-written question set.

Relevance is defined on *code*, not on chunk ids, so that every chunking
strategy is judged the same way: a retrieved chunk is relevant when it comes
from an expected file and its line range overlaps the expected symbol's
definition (resolved by parsing the file at evaluation time).
"""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from tree_sitter import Parser

from .chunking import PY_LANGUAGE, Chunk, _name, _unwrap_decorated

KS = (1, 3, 5, 10)


@dataclass
class Target:
    path: str
    symbol: str
    start: int
    end: int

    def matches(self, chunk: Chunk) -> bool:
        return chunk.path == self.path and chunk.start_line <= self.end and chunk.end_line >= self.start


@dataclass
class Question:
    id: str
    question: str
    category: str
    targets: list[Target] = field(default_factory=list)


def symbol_table(source: str) -> dict[str, tuple[int, int]]:
    """Map every qualified function/class name of a Python file to its 1-based line range."""
    tree = Parser(PY_LANGUAGE).parse(source.encode())
    table: dict[str, tuple[int, int]] = {}

    def walk(node, prefix: str):
        for child in node.children:
            target = _unwrap_decorated(child)
            if target.type in ("function_definition", "class_definition"):
                name = prefix + _name(target)
                table.setdefault(name, (child.start_point[0] + 1, child.end_point[0] + 1))
                body = target.child_by_field_name("body")
                if target.type == "class_definition" and body is not None:
                    walk(body, name + ".")

    walk(tree.root_node, "")
    return table


@lru_cache(maxsize=None)
def _table_for(path: Path) -> dict[str, tuple[int, int]]:
    return symbol_table(path.read_text(encoding="utf-8"))


def load_questions(path: Path, repo: Path) -> list[Question]:
    data = yaml.safe_load(Path(path).read_text())
    questions = []
    for q in data["questions"]:
        targets = []
        for ref in q["expected"]:
            file, symbol = ref.split("::")
            table = _table_for(Path(repo) / file)
            if symbol not in table:
                raise KeyError(f"{q['id']}: symbol {symbol!r} not found in {file}")
            targets.append(Target(file, symbol, *table[symbol]))
        questions.append(Question(q["id"], q["question"], q["category"], targets))
    return questions


def first_relevant_rank(question: Question, hits: list[Chunk]) -> int | None:
    for rank, chunk in enumerate(hits, start=1):
        if any(t.matches(chunk) for t in question.targets):
            return rank
    return None


def first_file_rank(question: Question, hits: list[Chunk]) -> int | None:
    """Looser view: rank of the first chunk coming from an expected *file*."""
    paths = {t.path for t in question.targets}
    return next((rank for rank, c in enumerate(hits, start=1) if c.path in paths), None)


def ndcg(question: Question, hits: list[Chunk], k: int) -> float:
    """Binary nDCG@k; several chunks can be relevant (a long function split in windows)."""
    gains = [1.0 if any(t.matches(c) for t in question.targets) else 0.0 for c in hits[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, max(int(sum(gains)), 1))))
    return dcg / ideal if ideal else 0.0


def evaluate(retriever, questions: list[Question], k_max: int = 10) -> dict:
    per_question, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        hits = [h.chunk for h in retriever.search(q.question, k_max)]
        latencies.append(time.perf_counter() - start)
        rank = first_relevant_rank(q, hits)
        per_question.append({"id": q.id, "category": q.category, "rank": rank, "file_rank": first_file_rank(q, hits),
                             "ndcg": ndcg(q, hits, k_max), "top": [f"{c.path}::{c.symbol}" for c in hits[:5]]})
    return {"summary": summarize(per_question, latencies), "per_question": per_question}


def summarize(rows: list[dict], latencies: list[float] | None = None) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["category"]].append(r)
    out = {}
    for name, items in groups.items():
        n = len(items)
        stats = {"n": n}
        for k in KS:
            stats[f"recall@{k}"] = sum(1 for r in items if r["rank"] and r["rank"] <= k) / n
        stats["file_recall@5"] = sum(1 for r in items if r.get("file_rank") and r["file_rank"] <= 5) / n
        stats["mrr@10"] = sum(1 / r["rank"] for r in items if r["rank"]) / n
        stats["ndcg@10"] = sum(r["ndcg"] for r in items) / n
        out[name] = stats
    if latencies:
        out["all"]["latency_ms_p50"] = 1000 * statistics.median(latencies)
    return out
