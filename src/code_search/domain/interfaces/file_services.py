"""File services interfaces."""

from abc import ABC, abstractmethod
from typing import List

from ..models import FileIndex


class IFileLoader(ABC):
    """Interface for loading source code files."""
    
    @abstractmethod
    async def load_files(self, project_directory: str) -> List[FileIndex]:
        """Load all source code files from the specified directory."""
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass 