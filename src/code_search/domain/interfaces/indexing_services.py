"""Indexing services interfaces."""

from abc import ABC, abstractmethod


class ICodeIndexer(ABC):
    """Interface for code indexing operations."""
    
    @abstractmethod
    async def index_project(self, project_directory: str) -> None:
        """Execute the complete indexing pipeline for a project."""
        pass

    @abstractmethod
    async def get_project_stats(self) -> dict:
        """Return statistics about the indexed project."""
        pass 