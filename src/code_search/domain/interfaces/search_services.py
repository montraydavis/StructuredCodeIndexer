"""Search services interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict

from ..models import SearchConfiguration, SearchResponse


class ISearchService(ABC):
    """Abstract interface for search services."""
    
    @abstractmethod
    async def search_by_text(self, query: str, config: SearchConfiguration) -> SearchResponse:
        """Search using natural language text query."""
        pass
    
    @abstractmethod
    async def search_by_embedding(self, embedding: List[float], config: SearchConfiguration) -> SearchResponse:
        """Search using pre-computed embedding vector."""
        pass
    
    @abstractmethod
    async def search_similar_code(self, code_content: str, config: SearchConfiguration) -> SearchResponse:
        """Find code similar to the provided code content."""
        pass
    
    @abstractmethod
    async def search_by_multiple_terms(self, terms: List[str], config: SearchConfiguration) -> SearchResponse:
        """Search using multiple terms combined into a single embedding."""
        pass
    
    @abstractmethod
    async def search_with_context(self, query: str, context: str, config: SearchConfiguration) -> SearchResponse:
        """Search with additional context to improve embedding quality."""
        pass
    
    @abstractmethod
    def clear_embedding_cache(self) -> None:
        """Clear the embedding service cache."""
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the embedding cache."""
        pass 