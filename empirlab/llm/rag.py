"""RAG Pipeline for Economic Literature Review.

Status: API stub — full implementation coming in v0.3.
"""
from __future__ import annotations


class LitReviewRAG:
    """RAG pipeline for economics literature review over PDFs / arXiv.

    Parameters
    ----------
    llm          : str, default 'gpt-4o'
    embedder     : str, default 'text-embedding-3-small'
    chunk_size   : int, default 512
    chunk_overlap: int, default 64
    k            : int, default 5   Top-k retrieved chunks per query.

    Usage (coming in v0.3)
    ----------------------
    >>> rag = LitReviewRAG()
    >>> rag.ingest(pdf_paths=['paper1.pdf', 'paper2.pdf'])
    >>> answer = rag.query('What identification strategy does the paper use?')
    """

    def __init__(self, llm="gpt-4o", embedder="text-embedding-3-small",
                 chunk_size=512, chunk_overlap=64, k=5):
        self.llm          = llm
        self.embedder     = embedder
        self.chunk_size   = chunk_size
        self.chunk_overlap= chunk_overlap
        self.k            = k

    def ingest(self, pdf_paths: list[str] = None,
               arxiv_ids: list[str] = None) -> None:
        raise NotImplementedError("LitReviewRAG coming in v0.3")

    def query(self, question: str) -> str:
        raise NotImplementedError

    def batch_query(self, questions: list[str]) -> list[str]:
        raise NotImplementedError
