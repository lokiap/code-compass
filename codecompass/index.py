"""Build and load an index: chunks on disk (JSONL) + embeddings in a persistent ChromaDB collection."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import Chunk, chunk_repository, make_chunker
from .search import BM25Retriever, DenseRetriever, HybridRetriever

DEFAULT_EMBEDDING = "minilm"


@dataclass(frozen=True)
class IndexConfig:
    chunker: str = "ast"  # "ast" | "lines"
    header: bool = True  # prepend "path :: symbol" to every chunk before indexing
    embedding: str = DEFAULT_EMBEDDING  # "minilm" or any sentence-transformers model name

    @property
    def name(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.embedding.lower()).strip("-")
        return f"{self.chunker}-{'hdr' if self.header else 'raw'}-{slug}"[:60]


def embedding_function(name: str):
    from chromadb.utils import embedding_functions as ef

    if name == "minilm":
        # all-MiniLM-L6-v2 through ONNX Runtime: no PyTorch needed, runs anywhere.
        return ef.DefaultEmbeddingFunction()
    # Any Hugging Face sentence-transformers model, e.g. "jinaai/jina-embeddings-v2-base-code".
    return ef.SentenceTransformerEmbeddingFunction(model_name=name, trust_remote_code=True)


def _client(index_dir: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(index_dir / "chroma"))


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_index(repo: Path, index_dir: Path, config: IndexConfig, batch_size: int = 128, log=print) -> int:
    """Chunk ``repo`` and embed every chunk into the collection ``config.name``. Returns the chunk count."""
    repo, index_dir = Path(repo), Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_repository(repo, make_chunker(config.chunker))
    log(f"[{config.name}] {len(chunks)} chunks from {repo}")

    with open(index_dir / f"{config.name}.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    client = _client(index_dir)
    try:
        client.delete_collection(config.name)
    except Exception:  # collection did not exist yet
        pass
    collection = client.create_collection(
        config.name, embedding_function=embedding_function(config.embedding), metadata={"hnsw:space": "cosine"}
    )
    start = time.time()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.indexed_text(config.header) for c in batch],
            metadatas=[{"path": c.path, "symbol": c.symbol, "kind": c.kind} for c in batch],
        )
        log(f"  embedded {min(i + batch_size, len(chunks))}/{len(chunks)} ({time.time() - start:.0f}s)", end="\r")
    log("")

    manifest = {"repo": str(repo.resolve()), "commit": _git_commit(repo), "config": asdict(config), "chunks": len(chunks)}
    (index_dir / f"{config.name}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return len(chunks)


class Index:
    """A loaded index exposing the three retrievers."""

    def __init__(self, index_dir: Path, config: IndexConfig):
        index_dir = Path(index_dir)
        path = index_dir / f"{config.name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no index '{config.name}' in {index_dir}; run `code-compass index` first")
        self.config = config
        self.manifest = json.loads((index_dir / f"{config.name}.manifest.json").read_text())
        self.chunks = [Chunk(**json.loads(line)) for line in path.open()]
        self.by_id = {c.id: c for c in self.chunks}
        collection = _client(index_dir).get_collection(config.name, embedding_function=embedding_function(config.embedding))
        self.bm25 = BM25Retriever(self.chunks, [c.indexed_text(config.header) for c in self.chunks])
        self.dense = DenseRetriever(collection, self.by_id)
        self.hybrid = HybridRetriever([self.bm25, self.dense])

    def retriever(self, mode: str):
        try:
            return {"bm25": self.bm25, "dense": self.dense, "hybrid": self.hybrid}[mode]
        except KeyError:
            raise ValueError(f"unknown retrieval mode {mode!r} (bm25, dense or hybrid)") from None
