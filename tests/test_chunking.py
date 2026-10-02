import textwrap

from codecompass.chunking import ASTChunker, LineChunker

SOURCE = textwrap.dedent(
    '''
    """Module docstring."""
    import os

    CONSTANT = 3
    OTHER = 4


    def helper(x):
        return x + 1


    class Model:
        """A model."""

        alpha = 0.5

        def fit(self, X):
            return self

        @property
        def size(self):
            return 1

        class Inner:
            def run(self):
                pass
    '''
).lstrip()


def symbols(chunks):
    return {(c.kind, c.symbol) for c in chunks}


def test_ast_chunker_emits_one_chunk_per_definition():
    chunks = ASTChunker().chunk_file("m.py", SOURCE)
    assert {("function", "helper"), ("method", "Model.fit"), ("method", "Model.size"),
            ("class", "Model"), ("class", "Model.Inner"), ("method", "Model.Inner.run")} <= symbols(chunks)
    assert any(c.kind == "module" and "CONSTANT" in c.text for c in chunks)


def test_decorated_method_keeps_its_decorator():
    size = next(c for c in ASTChunker().chunk_file("m.py", SOURCE) if c.symbol == "Model.size")
    assert size.text.lstrip().startswith("@property")


def test_class_summary_lists_methods_but_not_their_bodies():
    model = next(c for c in ASTChunker().chunk_file("m.py", SOURCE) if c.symbol == "Model")
    assert "alpha = 0.5" in model.text and "def fit(self, X):" in model.text
    assert "return self" not in model.text
    assert model.end_line < SOURCE.splitlines().index("    def fit(self, X):") + 1


def test_long_function_is_split_and_every_part_repeats_the_signature():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    src = f"def long(a, b):\n{body}\n    return a\n"
    parts = ASTChunker(max_lines=50, overlap=5).chunk_file("l.py", src)
    assert len(parts) > 3 and all(p.symbol == "long" for p in parts)
    assert all(p.text.startswith("def long(a, b):") for p in parts)
    assert parts[-1].end_line == len(src.splitlines())


def test_line_chunker_covers_the_whole_file_with_overlap():
    src = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = LineChunker(lines=30, overlap=10).chunk_file("f.py", src)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 100
    assert all(b.start_line == a.start_line + 20 for a, b in zip(chunks, chunks[1:]))


def test_chunk_ids_are_unique_and_stable():
    a = ASTChunker().chunk_file("m.py", SOURCE)
    b = ASTChunker().chunk_file("m.py", SOURCE)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)
