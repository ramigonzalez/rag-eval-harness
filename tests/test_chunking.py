import numpy as np

from raglab.chunking import FixedChunker, RecursiveChunker, SemanticChunker, WholeChunker
from raglab.textutil import split_sentences, word_count
from raglab.types import Document


def test_fixed_windows_overlap_and_cover_the_tail():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = FixedChunker(chunk_size=4, overlap=1).split_text(text)
    assert chunks == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]


def test_fixed_short_text_is_one_chunk():
    assert FixedChunker(100, 25).split_text("only a few words") == ["only a few words"]


def test_fixed_rejects_overlap_that_does_not_advance():
    try:
        FixedChunker(chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("overlap == chunk_size should fail")


def test_recursive_packs_on_sentence_boundaries():
    text = "One two three. Four five six. Seven eight nine."
    chunks = RecursiveChunker(chunk_size=6, overlap=3).split_text(text)
    assert chunks == [
        "One two three. Four five six.",
        "Four five six. Seven eight nine.",
    ]
    assert all(word_count(chunk) <= 6 for chunk in chunks)


def test_recursive_respects_a_large_overlap_without_hanging():
    text = "aa bb cc. dd ee ff gg hh ii jj. kk ll mm nn oo pp."
    chunks = RecursiveChunker(chunk_size=10, overlap=8).split_text(text)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_whole_document_keeps_title_and_body():
    document = Document("d1", "Title", "Body text here.")
    chunks = WholeChunker().chunk_document(document)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1::c0"
    assert chunks[0].text == "Title\n\nBody text here."


class _StubEmbedder:
    model_name = "stub"

    def __init__(self, mapping):
        self.mapping = mapping

    def embed_documents(self, texts):
        return np.stack([np.asarray(self.mapping[text], dtype=np.float32) for text in texts])


def test_semantic_splits_where_the_topic_shifts():
    text = "Red apples are sweet. Red apples are crisp. Blue oceans are vast. Blue oceans are deep."
    sentences = split_sentences(text)
    assert len(sentences) == 4
    mapping = {
        sentences[0]: [1.0, 0.0],
        sentences[1]: [1.0, 0.0],
        sentences[2]: [0.0, 1.0],
        sentences[3]: [0.0, 1.0],
    }
    chunker = SemanticChunker(breakpoint_percentile=50, max_chunk_words=40, min_chunk_words=1)
    document = Document("fruit", "", text)
    chunks = chunker.chunk_corpus([document], _StubEmbedder(mapping))
    assert len(chunks) == 2
    assert "sweet" in chunks[0].text and "crisp" in chunks[0].text
    assert "oceans" in chunks[1].text
    assert "oceans" not in chunks[0].text


def test_semantic_caps_a_long_single_topic_document():
    sentence = "Alpha beta gamma delta."
    text = " ".join([sentence] * 8)
    chunker = SemanticChunker(breakpoint_percentile=95, max_chunk_words=6, min_chunk_words=1)
    chunks = chunker.chunk_corpus(
        [Document("d", "", text)],
        _StubEmbedder({sentence: [1.0, 0.0]}),
    )
    assert len(chunks) > 1
    assert all(word_count(chunk.text) <= 6 for chunk in chunks)
