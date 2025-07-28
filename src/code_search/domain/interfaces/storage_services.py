"""Storage services interfaces."""

from abc import ABC, abstractmethod
from typing import List, Optional, Set

from ..models import FileIndex, CodeMember, CodeMethod, SearchResult, SearchConfiguration


class IVectorStore(ABC):
    """Interface for vector storage operations."""
    
    @abstractmethod
    async def store_file_index(self, file_index: FileIndex) -> None:
        """Store file index with embedding."""
        pass

    @abstractmethod
    async def store_code_member(self, code_member: CodeMember) -> None:
        """Store code member with embedding."""
        pass

    @abstractmethod
    async def store_code_method(self, code_method: CodeMethod) -> None:
        """Store code method with embedding."""
        pass

    @abstractmethod
    async def get_file_by_id(self, file_id: str) -> Optional[FileIndex]:
        """Retrieve file index by ID."""
        pass

    @abstractmethod
    async def get_members_by_file_id(self, file_id: str) -> List[CodeMember]:
        """Retrieve all members for a specific file."""
        pass

    @abstractmethod
    async def get_methods_by_member_id(self, member_id: str) -> List[CodeMethod]:
        """Retrieve all methods for a specific member."""
        pass

    @abstractmethod
    async def file_exists(self, file_path: str, content_hash: str) -> Optional[FileIndex]:
        """Check if file exists with given path and content hash."""
        pass

    @abstractmethod
    async def member_exists(self, content_hash: str) -> Optional[CodeMember]:
        """Check if member exists with given content hash."""
        pass

    @abstractmethod
    async def method_exists(self, content_hash: str) -> Optional[CodeMethod]:
        """Check if method exists with given content hash."""
        pass

    @abstractmethod
    async def get_all_file_hashes(self) -> Set[str]:
        """Get all existing file content hashes for resume capability."""
        pass

    @abstractmethod
    async def search_files(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search files by embedding similarity."""
        pass

    @abstractmethod
    async def search_members(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search members by embedding similarity."""
        pass

    @abstractmethod
    async def search_methods(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search methods by embedding similarity."""
        pass 