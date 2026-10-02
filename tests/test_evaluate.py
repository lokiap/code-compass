import textwrap

from codecompass.chunking import Chunk
from codecompass.evaluate import KS, Question, Target, first_relevant_rank, ndcg, summarize, symbol_table


def test_symbol_table_resolves_nested_and_decorated_definitions():
    src = textwrap.dedent('''
        def f():
            pass

        class A:
            @staticmethod
            def g():
                return 1
    ''').lstrip()
    table = symbol_table(src)
    assert table["f"] == (1, 2)
    assert table["A"] == (4, 7)
    assert table["A.g"] == (5, 7)


def test_relevance_is_line_overlap_in_the_expected_file():
    q = Question("q", "?", "semantic", [Target("a.py", "f", 10, 20)])
    hits = [Chunk("b.py", 10, 20, ""), Chunk("a.py", 1, 9, ""), Chunk("a.py", 18, 40, "")]
    assert first_relevant_rank(q, hits) == 3
    assert first_relevant_rank(q, hits[:2]) is None
    assert 0 < ndcg(q, hits, 10) < 1


def test_summary_metrics():
    rows = [{"category": "semantic", "rank": 1, "ndcg": 1.0}, {"category": "keyword", "rank": 4, "ndcg": 0.4},
            {"category": "keyword", "rank": None, "ndcg": 0.0}]
    s = summarize(rows)
    assert s["all"]["recall@1"] == 1 / 3 and s["all"]["recall@5"] == 2 / 3
    assert abs(s["all"]["mrr@10"] - (1 + 0.25) / 3) < 1e-9
    assert s["keyword"]["n"] == 2 and set(f"recall@{k}" for k in KS) <= set(s["semantic"])
