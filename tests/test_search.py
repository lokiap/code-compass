from codecompass.chunking import Chunk
from codecompass.search import BM25Retriever, HybridRetriever, Hit, code_tokens


def test_code_tokens_split_identifiers_but_keep_them_whole():
    toks = code_tokens("def check_is_fitted(estimator): KMeansPlusPlus")
    assert {"check_is_fitted", "check", "fitted", "kmeansplusplus", "k", "means", "plus"} <= set(toks)


def test_code_tokens_drop_question_stopwords():
    assert code_tokens("How is the model fitted?") == ["model", "fitted"]


def _chunks():
    texts = ["def check_is_fitted(estimator): raise NotFittedError",
             "def train_test_split(*arrays, test_size=None): shuffle and split",
             "class KMeans: lloyd iterations and inertia"]
    return [Chunk(f"f{i}.py", 1, 1, t, symbol=f"s{i}") for i, t in enumerate(texts)], texts


def test_bm25_ranks_the_lexical_match_first():
    chunks, texts = _chunks()
    hits = BM25Retriever(chunks, texts).search("is the estimator fitted?", k=3)
    assert hits[0].chunk.symbol == "s0"
    assert BM25Retriever(chunks, texts).search("unknownword") == []


class _Fixed:
    name = "fixed"

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k=10):
        return [Hit(c, 1.0) for c in self.chunks[:k]]


def test_rrf_rewards_documents_ranked_well_by_both_retrievers():
    chunks, _ = _chunks()
    a, b, c = chunks
    hybrid = HybridRetriever([_Fixed([a, b, c]), _Fixed([b, c, a])])
    assert [h.chunk.symbol for h in hybrid.search("q", 3)][0] == "s1"
