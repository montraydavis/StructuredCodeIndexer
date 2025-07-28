"""VectorDB document schemas."""

from docarray import BaseDoc
from docarray.typing import NdArray
import numpy as np
from typing import Any


class FileDoc(BaseDoc):
    """Document schema for file storage in VectorDB."""
    file_path: str
    content: str
    content_hash: str
    embedding: NdArray


class MemberDoc(BaseDoc):
    """Document schema for code member storage in VectorDB."""
    file_id: str
    type: str
    name: str
    text_content: str  # Avoids conflict with BaseDoc.summary() method
    content_hash: str
    embedding: NdArray


class MethodDoc(BaseDoc):
    """Document schema for method storage in VectorDB."""
    member_id: str
    name: str
    text_content: str  # Avoids conflict with BaseDoc.summary() method
    content_hash: str
    embedding: NdArray
