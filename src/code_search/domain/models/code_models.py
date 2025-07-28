"""Code structure domain models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import uuid
import hashlib


class MemberType(Enum):
    """Type of code member."""
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"


@dataclass
class CodeMember:
    """Represents a code member (class, interface, enum)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_id: str = ""
    type: MemberType = MemberType.CLASS
    name: str = ""
    summary: str = ""
    content_hash: str = ""
    summary_embedding: Optional[List[float]] = None
    methods: List['CodeMethod'] = field(default_factory=list)

    def __post_init__(self):
        if self.content_hash == "" and self.summary:
            self.content_hash = hashlib.sha256(f"{self.file_id}:{self.name}:{self.summary}".encode()).hexdigest()


@dataclass
class CodeMethod:
    """Represents a method within a code member."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    member_id: str = ""
    name: str = ""
    summary: str = ""
    content_hash: str = ""
    summary_embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.content_hash == "" and self.summary:
            self.content_hash = hashlib.sha256(f"{self.member_id}:{self.name}:{self.summary}".encode()).hexdigest() 