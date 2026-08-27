"""Vector Store abstraction and FAISS implementation."""

from __future__ import annotations

from typing import Protocol

class Document:
    def __init__(self, content: str):
        self.page_content = content

class VectorStore(Protocol):
    async def similarity_search(self, embedding: list[float], top_k: int) -> list[Document]:
        ...

class FAISSVectorStore:
    """FAISS adapter for the vector store."""
    def __init__(self):
        pass

    async def similarity_search(self, embedding: list[float], top_k: int) -> list[Document]:
        # Mock implementation returning empty for now
        return [Document("mock document content")]
