"""Retrievers: lexical (BM25), dense (ChromaDB embeddings) and hybrid (reciprocal rank fusion)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

from .chunking import Chunk

# --------------------------------------------------------------------------- tokenisation

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset(
    """a an and are as at be by does do for from how i in is it of on or that the this to what when where which
    who why with within can should would there their its into we you used use""".split()
)


def code_tokens(text: str) -> list[str]:
    """Tokenise code *and* questions the same way.

    ``KMeansPlusPlus`` and ``kmeans_plusplus`` both yield their full lower-cased
    identifier plus the sub-words, so a question saying "k-means plus plus"
    still matches the identifier.
    """
    out: list[str] = []
    for word in _WORD.findall(text):
        lower = word.lower()
        parts = [p.lower() for piece in word.split("_") if piece for p in _CAMEL.findall(piece)]
        if lower not in STOPWORDS:
            out.append(lower)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in STOPWORDS)
    return out


# --------------------------------------------------------------------------- retrievers


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


class BM25Retriever:
    """Okapi BM25 over code-aware tokens. Small, dependency-free, fast enough for ~10^5 chunks."""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], texts: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.chunks, self.k1, self.b = list(chunks), k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = []
        for i, text in enumerate(texts):
            tf = Counter(code_tokens(text))
            self.doc_len.append(sum(tf.values()))
            for term, count in tf.items():
                self.postings[term].append((i, count))
        n = len(self.doc_len)
        self.avgdl = sum(self.doc_len) / max(n, 1)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(code_tokens(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avgdl)
                scores[doc] += idf * tf * (self.k1 + 1) / (tf + norm)
        best = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(self.chunks[i], s) for i, s in best]


class DenseRetriever:
    """Nearest neighbours in a ChromaDB collection (cosine distance)."""

    name = "dense"

    def __init__(self, collection, chunks_by_id: dict[str, Chunk]):
        self.collection, self.chunks_by_id = collection, chunks_by_id

    def search(self, query: str, k: int = 10) -> list[Hit]:
        res = self.collection.query(query_texts=[query], n_results=k, include=["distances"])
        return [Hit(self.chunks_by_id[i], 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]


class HybridRetriever:
    """Reciprocal Rank Fusion: score(d) = sum over retrievers of 1 / (rrf_k + rank(d)).

    RRF only uses ranks, so it needs no score normalisation between BM25 and cosine similarity.
    """

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, depth: int = 50):
        self.retrievers, self.rrf_k, self.depth = list(retrievers), rrf_k, depth

    def search(self, query: str, k: int = 10) -> list[Hit]:
        fused: dict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, self.depth), start=1):
                fused[hit.chunk.id] += 1.0 / (self.rrf_k + rank)
                chunks[hit.chunk.id] = hit.chunk
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(chunks[i], s) for i, s in best]
