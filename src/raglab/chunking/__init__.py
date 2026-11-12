"""Chunking strategies. Importing this package registers the built-in chunkers."""

from raglab.chunking.fixed import FixedChunker
from raglab.chunking.markdown import MarkdownChunker
from raglab.chunking.parent_document import ParentDocumentChunker
from raglab.chunking.recursive import RecursiveChunker
from raglab.chunking.registry import CHUNKERS, build_chunker, register_chunker
from raglab.chunking.semantic import SemanticChunker
from raglab.chunking.sentence_window import SentenceWindowChunker
from raglab.chunking.whole import WholeChunker

__all__ = [
    "CHUNKERS",
    "FixedChunker",
    "MarkdownChunker",
    "ParentDocumentChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceWindowChunker",
    "WholeChunker",
    "build_chunker",
    "register_chunker",
]
