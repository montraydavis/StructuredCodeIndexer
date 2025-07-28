"""Embedding services interfaces."""

from abc import ABC, abstractmethod
from typing import List


class IEmbeddingService(ABC):
    """Interface for generating vector embeddings."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for a single text."""
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for multiple texts."""
        pass 