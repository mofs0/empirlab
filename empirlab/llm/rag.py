"""
RAG Pipeline for Economic Literature Review
=============================================
Retrieval-Augmented Generation over PDF papers / arXiv abstracts.

Dependencies (install with pip install empirlab[llm]):
    openai, langchain, langchain-community, faiss-cpu, pypdf, tiktoken

Usage
-----
>>> from empirlab.llm import LitReviewRAG
>>> rag = LitReviewRAG(llm="gpt-4o")
>>> rag.ingest_pdfs(["paper1.pdf", "paper2.pdf"])
>>> answer = rag.query("What identification strategies are used for ATE?")
>>> print(answer)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class LitReviewRAG:
    """RAG pipeline for economics literature review over PDFs.

    Parameters
    ----------
    llm : str, default 'gpt-4o'
        OpenAI model for generation. Can also use 'gpt-4o-mini' for speed.
    embedder : str, default 'text-embedding-3-small'
        OpenAI embedding model.
    chunk_size : int, default 512
        Token chunk size for text splitting.
    chunk_overlap : int, default 64
        Overlap between consecutive chunks.
    k : int, default 5
        Number of top-k chunks retrieved per query.
    openai_api_key : str or None
        Defaults to OPENAI_API_KEY env variable.

    Examples
    --------
    >>> rag = LitReviewRAG(llm="gpt-4o-mini", k=3)
    >>> rag.ingest_pdfs(["chernozhukov2018.pdf"])
    >>> print(rag.query("What is the key identification assumption?"))
    >>> print(rag.query("How is cross-fitting used?"))
    >>> # Explore retrieved chunks directly
    >>> docs = rag.retrieve("cross-fitting")
    >>> for doc in docs:
    ...     print(doc.page_content[:200])
    """

    def __init__(self, llm: str = "gpt-4o",
                 embedder: str = "text-embedding-3-small",
                 chunk_size: int = 512, chunk_overlap: int = 64,
                 k: int = 5, openai_api_key: Optional[str] = None):
        self.llm_name    = llm
        self.embedder    = embedder
        self.chunk_size  = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k           = k
        self.api_key     = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._vectorstore = None

    # ------------------------------------------------------------------
    def ingest_pdfs(self, pdf_paths: list[str]) -> "LitReviewRAG":
        """Load and index a list of PDF files.

        Parameters
        ----------
        pdf_paths : list of str or Path
            Paths to PDF files to ingest.

        Returns
        -------
        self
        """
        self._require_deps()
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        all_docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(str(path))
            pages  = loader.load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)
            print(f"  Ingested {Path(path).name}: {len(pages)} pages → {len(chunks)} chunks")

        embeddings = OpenAIEmbeddings(
            model=self.embedder, openai_api_key=self.api_key
        )
        self._vectorstore = FAISS.from_documents(all_docs, embeddings)
        print(f"Total chunks indexed: {len(all_docs)}")
        return self

    # ------------------------------------------------------------------
    def ingest_texts(self, texts: list[str],
                     metadatas: Optional[list[dict]] = None) -> "LitReviewRAG":
        """Index raw text strings (e.g. arXiv abstracts).

        Parameters
        ----------
        texts     : list of str
        metadatas : optional list of dicts (one per text)
        """
        self._require_deps()
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            chunks = splitter.create_documents([text], metadatas=[meta])
            docs.extend(chunks)

        embeddings = OpenAIEmbeddings(model=self.embedder, openai_api_key=self.api_key)
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(docs, embeddings)
        else:
            self._vectorstore.add_documents(docs)
        return self

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> list:
        """Return top-k chunks most relevant to query."""
        self._check_indexed()
        return self._vectorstore.similarity_search(query, k=self.k)

    # ------------------------------------------------------------------
    def query(self, question: str,
              system_prompt: Optional[str] = None) -> str:
        """Answer a question using retrieved context.

        Parameters
        ----------
        question      : str  The research question.
        system_prompt : str or None  Custom system prompt.

        Returns
        -------
        str  The generated answer.
        """
        self._check_indexed()
        self._require_deps()
        from openai import OpenAI

        docs = self.retrieve(question)
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source','?')} p{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )
        system = system_prompt or (
            "You are an expert economist and research assistant. "
            "Answer the user's question based ONLY on the provided context. "
            "Be precise, cite equation numbers or page references when possible, "
            "and say 'Not found in context' if the answer is not supported."
        )
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the FAISS index to disk."""
        self._check_indexed()
        self._vectorstore.save_local(path)
        print(f"Index saved to {path}")

    def load(self, path: str) -> "LitReviewRAG":
        """Load a previously saved FAISS index."""
        self._require_deps()
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        embeddings = OpenAIEmbeddings(model=self.embedder, openai_api_key=self.api_key)
        self._vectorstore = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
        return self

    # ------------------------------------------------------------------
    def _require_deps(self):
        try:
            import langchain        # noqa: F401
            import openai           # noqa: F401
            import faiss            # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"LitReviewRAG requires: pip install empirlab[llm]\n"
                f"Missing: {e}"
            ) from e

    def _check_indexed(self):
        if self._vectorstore is None:
            raise RuntimeError("No documents indexed. Call ingest_pdfs() or ingest_texts() first.")
