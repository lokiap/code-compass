"""Command-line interface: ``code-compass index | search | ask | bench``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .index import DEFAULT_EMBEDDING, Index, IndexConfig, build_index

DEFAULT_INDEX_DIR = Path(".index")


def _config(args) -> IndexConfig:
    return IndexConfig(chunker=args.chunker, header=not args.no_header, embedding=args.embedding)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--chunker", choices=["ast", "lines"], default="ast")
    p.add_argument("--no-header", action="store_true", help="do not prepend 'path :: symbol' to chunks")
    p.add_argument("--embedding", default=DEFAULT_EMBEDDING, help="'minilm' or a sentence-transformers model")


def _print_hits(hits) -> None:
    for i, h in enumerate(hits, 1):
        c = h.chunk
        where = f"{c.path}:{c.start_line}-{c.end_line}"
        print(f"{i:>2}. {where}  {c.symbol or '(module)'}  [{c.kind}]  score={h.score:.3f}")


def cmd_index(args) -> None:
    build_index(args.repo, args.index_dir, _config(args))


def cmd_search(args) -> None:
    index = Index(args.index_dir, _config(args))
    _print_hits(index.retriever(args.mode).search(args.query, args.k))


def cmd_ask(args) -> None:
    from .generate import answer

    index = Index(args.index_dir, _config(args))
    hits = index.retriever(args.mode).search(args.question, args.k)
    result = answer(args.question, hits, model=args.model)
    print(result.text)
    if result.citations:
        print("\nSources:")
        for i, c in enumerate(result.citations, 1):
            print(f"  [{i}] {c['source']}")
    else:
        print("\nRetrieved:")
        _print_hits(hits)


def cmd_bench(args) -> None:
    from .evaluate import load_questions, evaluate

    questions = load_questions(args.questions, args.repo)
    configs = [IndexConfig(chunker, header, args.embedding) for chunker in ("lines", "ast") for header in (False, True)]
    results = []
    for cfg in configs:
        if args.build or not (args.index_dir / f"{cfg.name}.jsonl").exists():
            build_index(args.repo, args.index_dir, cfg)
        index = Index(args.index_dir, cfg)
        for mode in ("bm25", "dense", "hybrid"):
            res = evaluate(index.retriever(mode), questions)
            results.append({"index": cfg.name, "chunker": cfg.chunker, "header": cfg.header, "mode": mode,
                            "chunks": len(index.chunks), **res})
            s = res["summary"]["all"]
            print(f"{cfg.name:<22} {mode:<7} R@1={s['recall@1']:.2f} R@5={s['recall@5']:.2f} "
                  f"R@10={s['recall@10']:.2f} MRR={s['mrr@10']:.3f} fileR@5={s['file_recall@5']:.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="code-compass", description="Search and question a code base.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="chunk and embed a repository")
    p.add_argument("repo", type=Path)
    _add_config_args(p)
    p.set_defaults(func=cmd_index)

    for name, helptext in (("search", "show the retrieved chunks"), ("ask", "answer with Claude, with citations")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("query" if name == "search" else "question")
        p.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
        p.add_argument("-k", type=int, default=8)
        _add_config_args(p)
        if name == "ask":
            p.add_argument("--model", default="claude-opus-5-5")
        p.set_defaults(func=cmd_search if name == "search" else cmd_ask)

    p = sub.add_parser("bench", help="run the retrieval benchmark over all configurations")
    p.add_argument("repo", type=Path)
    p.add_argument("--questions", type=Path, default=Path("eval/questions.yaml"))
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--embedding", default=DEFAULT_EMBEDDING)
    p.add_argument("--build", action="store_true", help="rebuild indexes even if they exist")
    p.add_argument("--out", type=Path, default=Path("results/bench.json"))
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
