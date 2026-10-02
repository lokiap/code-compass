"""Split a source tree into retrievable chunks.

Two strategies are compared in the benchmark:

* ``LineChunker``: fixed windows of N lines with overlap. The usual baseline,
  blind to code structure (a window can start in the middle of a function).
* ``ASTChunker``: one chunk per function / method, plus one "summary" chunk
  per class (signature, docstring, attributes and the list of its methods)
  and chunks for module-level code. Parsing is done with tree-sitter.
  Functions longer than ``max_lines`` are split into windows that all repeat
  the signature, so every piece keeps its context.

Every chunk carries its path, line range and qualified symbol name
(``KMeans.fit``), which is what the evaluation and the citations rely on.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import tree_sitter_python
from tree_sitter import Language, Node, Parser

PY_LANGUAGE = Language(tree_sitter_python.language())

DEFAULT_EXCLUDES = ("tests", "test", "externals", "_vendor", "vendor", "build", "docs", "examples")


@dataclass
class Chunk:
    path: str  # relative to the repository root, POSIX separators
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    symbol: str = ""  # qualified name, e.g. "KMeans.fit"; "" for module-level code
    kind: str = "lines"  # function | method | class | module | lines
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        raw = f"{self.path}:{self.start_line}-{self.end_line}:{self.symbol}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def header(self) -> str:
        """Short context line prepended to the text before indexing ("contextual header")."""
        where = f"{self.path}" + (f" :: {self.symbol}" if self.symbol else "")
        return f"# {where} ({self.kind}, lines {self.start_line}-{self.end_line})\n"

    def indexed_text(self, with_header: bool) -> str:
        return (self.header() + self.text) if with_header else self.text


def iter_source_files(
    root: Path, suffixes: Iterable[str] = (".py",), excludes: Iterable[str] = DEFAULT_EXCLUDES
) -> Iterator[Path]:
    excludes = set(excludes)
    for path in sorted(root.rglob("*")):
        if path.suffix not in suffixes or not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(p in excludes or p.startswith(".") for p in parts[:-1]):
            continue
        if path.name.startswith("test_") or path.name == "conftest.py":
            continue
        yield path


class LineChunker:
    name = "lines"

    def __init__(self, lines: int = 60, overlap: int = 15):
        if overlap >= lines:
            raise ValueError("overlap must be smaller than the window size")
        self.lines, self.overlap = lines, overlap

    def chunk_file(self, rel_path: str, source: str) -> list[Chunk]:
        lines = source.splitlines()
        chunks, step = [], self.lines - self.overlap
        for start in range(0, max(len(lines), 1), step):
            window = lines[start : start + self.lines]
            if not any(line.strip() for line in window):
                continue
            chunks.append(Chunk(rel_path, start + 1, start + len(window), "\n".join(window)))
            if start + self.lines >= len(lines):
                break
        return chunks


class ASTChunker:
    name = "ast"

    def __init__(self, max_lines: int = 80, overlap: int = 10, module_block_lines: int = 40):
        self.max_lines, self.overlap, self.module_block_lines = max_lines, overlap, module_block_lines
        self.parser = Parser(PY_LANGUAGE)

    # -- public -----------------------------------------------------------
    def chunk_file(self, rel_path: str, source: str) -> list[Chunk]:
        src = source.encode()
        tree = self.parser.parse(src)
        lines = source.splitlines()
        chunks: list[Chunk] = []
        module_rows: list[int] = []  # rows (0-based) of top-level code outside defs/classes

        for node in tree.root_node.children:
            target = _unwrap_decorated(node)
            if target.type == "function_definition":
                chunks += self._function(rel_path, lines, node, target, prefix="", kind="function")
            elif target.type == "class_definition":
                chunks += self._class(rel_path, lines, node, target, prefix="")
            else:
                module_rows += range(node.start_point[0], node.end_point[0] + 1)

        chunks += self._module_blocks(rel_path, lines, module_rows)
        return sorted(chunks, key=lambda c: (c.start_line, c.end_line))

    # -- helpers ----------------------------------------------------------
    def _function(self, path, lines, outer: Node, fn: Node, prefix: str, kind: str) -> list[Chunk]:
        name = _name(fn)
        symbol = f"{prefix}{name}"
        start, end = outer.start_point[0], outer.end_point[0]
        body = fn.child_by_field_name("body")
        sig_end = body.start_point[0] - 1 if body is not None and body.start_point[0] > fn.start_point[0] else fn.start_point[0]
        signature = lines[start : sig_end + 1]
        return self._split(path, lines, start, end, symbol, kind, signature)

    def _class(self, path, lines, outer: Node, cls: Node, prefix: str) -> list[Chunk]:
        name = _name(cls)
        symbol = f"{prefix}{name}"
        body = cls.child_by_field_name("body")
        chunks: list[Chunk] = []
        summary_rows: list[int] = list(range(outer.start_point[0], (body.start_point[0] if body else cls.end_point[0])))
        method_sigs: list[str] = []
        if body is not None:
            for child in body.children:
                target = _unwrap_decorated(child)
                if target.type == "function_definition":
                    chunks += self._function(path, lines, child, target, prefix=f"{symbol}.", kind="method")
                    method_sigs.append(lines[target.start_point[0]].strip())
                elif target.type == "class_definition":
                    chunks += self._class(path, lines, child, target, prefix=f"{symbol}.")
                    method_sigs.append(lines[target.start_point[0]].strip())
                else:  # docstring, class attributes, comments
                    summary_rows += range(child.start_point[0], child.end_point[0] + 1)

        summary = [lines[r] for r in sorted(set(summary_rows)) if r < len(lines)]
        if len(summary) > self.max_lines:  # very long docstrings (sklearn!) get truncated
            summary = summary[: self.max_lines] + ["    ..."]
        if method_sigs:
            summary += ["", "    # methods:"] + [f"    # {s}" for s in method_sigs]
        # The summary's line range is the class header + docstring/attributes, not the whole class,
        # so that it does not "overlap" every method during evaluation.
        header_end = max(summary_rows) if summary_rows else outer.start_point[0]
        chunks.append(Chunk(path, outer.start_point[0] + 1, header_end + 1, "\n".join(summary), symbol, "class"))
        return chunks

    def _split(self, path, lines, start, end, symbol, kind, signature) -> list[Chunk]:
        """Emit [start, end] as one chunk, or as overlapping windows repeating the signature."""
        n = end - start + 1
        if n <= self.max_lines:
            return [Chunk(path, start + 1, end + 1, "\n".join(lines[start : end + 1]), symbol, kind)]
        chunks, step, sig_len = [], self.max_lines - self.overlap, len(signature)
        part = 0
        for s in range(start, end + 1, step):
            e = min(s + self.max_lines - 1, end)
            body = lines[s : e + 1]
            text = body if s == start else signature + ["    # ..."] + body
            chunks.append(
                Chunk(path, s + 1, e + 1, "\n".join(text), symbol, kind, meta={"part": part, "signature_lines": sig_len})
            )
            part += 1
            if e == end:
                break
        return chunks

    def _module_blocks(self, path, lines, rows: list[int]) -> list[Chunk]:
        """Group contiguous top-level rows (imports, constants, scripts) into blocks."""
        rows = sorted(set(r for r in rows if r < len(lines)))
        chunks, block = [], []
        for r in rows:
            if block and (r != block[-1] + 1 or len(block) >= self.module_block_lines):
                chunks.append(self._module_chunk(path, lines, block))
                block = []
            block.append(r)
        if block:
            chunks.append(self._module_chunk(path, lines, block))
        # Drop crumbs (a lone comment or blank line between two functions).
        return [c for c in chunks if sum(1 for line in c.text.splitlines() if line.strip()) >= 2]

    @staticmethod
    def _module_chunk(path, lines, block) -> Chunk:
        return Chunk(path, block[0] + 1, block[-1] + 1, "\n".join(lines[block[0] : block[-1] + 1]), "", "module")


def _unwrap_decorated(node: Node) -> Node:
    if node.type == "decorated_definition":
        definition = node.child_by_field_name("definition")
        if definition is not None:
            return definition
    return node


def _name(node: Node) -> str:
    name = node.child_by_field_name("name")
    return name.text.decode() if name is not None else "<anonymous>"


def make_chunker(name: str):
    if name == "ast":
        return ASTChunker()
    if name == "lines":
        return LineChunker()
    raise ValueError(f"unknown chunker: {name!r} (expected 'ast' or 'lines')")


def chunk_repository(root: Path, chunker, excludes: Iterable[str] = DEFAULT_EXCLUDES) -> list[Chunk]:
    root = Path(root)
    chunks: list[Chunk] = []
    for path in iter_source_files(root, excludes=excludes):
        source = path.read_text(encoding="utf-8", errors="replace")
        chunks += chunker.chunk_file(path.relative_to(root).as_posix(), source)
    return chunks
