"""File-related domain models."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, TYPE_CHECKING
import uuid
import hashlib

if TYPE_CHECKING:
    from .code_models import CodeMember


@dataclass
class FileIndex:
    """Represents an indexed source code file."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    content: str = ""
    content_hash: str = ""
    content_embedding: Optional[List[float]] = None
    members: List['CodeMember'] = field(default_factory=list)

    def __post_init__(self):
        if self.content_hash == "" and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class CodeParsingResult:
    """Result from AI code parsing."""
    members: List[Dict] = field(default_factory=list)
    success: bool = True
    error_message: str = "" 