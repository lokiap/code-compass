from types import SimpleNamespace

from codecompass.chunking import Chunk
from codecompass.generate import answer
from codecompass.search import Hit


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_answer_sends_chunks_as_citable_documents_and_collects_citations():
    citation = SimpleNamespace(document_title="a.py::f (lines 1-2)", cited_text="return 1")
    response = SimpleNamespace(stop_reason="end_turn", content=[
        SimpleNamespace(type="text", text="f returns one", citations=[citation]),
        SimpleNamespace(type="text", text=".", citations=None),
    ])
    messages = FakeMessages(response)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    hit = Hit(Chunk("a.py", 1, 2, "def f():\n    return 1", symbol="f", kind="function"), 1.0)

    result = answer("what does f return?", [hit], client=client)

    docs = messages.kwargs["messages"][0]["content"]
    assert docs[0]["type"] == "document" and docs[0]["citations"] == {"enabled": True}
    assert docs[0]["title"] == "a.py::f (lines 1-2)" and docs[-1]["text"] == "what does f return?"
    assert result.text == "f returns one [1]." and result.citations[0]["quote"] == "return 1"
