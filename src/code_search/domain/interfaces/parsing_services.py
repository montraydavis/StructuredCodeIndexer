"""Parsing services interfaces."""

from abc import ABC, abstractmethod
from typing import List

from ..models import FileIndex, CodeMember


class ICodeParser(ABC):
    """Interface for parsing code structure."""
    
    @abstractmethod
    async def parse_file(self, file_index: FileIndex) -> List[CodeMember]:
        """Extract classes, interfaces, and enums from file content."""
        pass

    @abstractmethod
    def supports_file_type(self, file_path: str) -> bool:
        """Check if the file type can be parsed."""
        pass


class IAICodeParsingService(ABC):
    """Abstract interface for AI-powered code parsing services."""
    
    @abstractmethod
    async def parse_code_to_members(self, file_content: str, file_path: str) -> List[CodeMember]:
        """Parse code content using AI and return structured code members."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the AI service."""
        pass 