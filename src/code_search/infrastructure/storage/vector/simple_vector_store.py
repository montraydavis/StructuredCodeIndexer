"""Simple vector store implementation using numpy for better compatibility."""

from typing import List, Optional, Set, Dict, Any
from pathlib import Path
import json
import pickle
import numpy as np
from dataclasses import asdict

from ....domain.interfaces import IVectorStore
from ....domain.models import FileIndex, CodeMember, CodeMethod, SearchResult, MemberType


class SimpleVectorStore(IVectorStore):
    """Simple vector storage implementation using numpy arrays and pickle files."""

    def __init__(self, workspace_path: str = "./simple_vectordb_workspace"):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(exist_ok=True)

        # Storage files
        self.files_data_path = self.workspace_path / "files_data.pkl"
        self.members_data_path = self.workspace_path / "members_data.pkl"
        self.methods_data_path = self.workspace_path / "methods_data.pkl"

        # In-memory storage
        self._files_data: List[Dict[str, Any]] = []
        self._members_data: List[Dict[str, Any]] = []
        self._methods_data: List[Dict[str, Any]] = []

        # In-memory caches for existence checking
        self._file_hashes: Set[str] = set()
        self._member_hashes: Set[str] = set()
        self._method_hashes: Set[str] = set()
        self._cache_loaded = False

    async def _load_data(self):
        """Load existing data from disk."""
        if self._cache_loaded:
            return

        try:
            if self.files_data_path.exists():
                with open(self.files_data_path, 'rb') as f:
                    self._files_data = pickle.load(f)
                    self._file_hashes = {item['content_hash'] for item in self._files_data}
        except Exception as e:
            print(f"⚠️  Failed to load files data: {e}")
            self._files_data = []
            self._file_hashes = set()

        try:
            if self.members_data_path.exists():
                with open(self.members_data_path, 'rb') as f:
                    self._members_data = pickle.load(f)
                    self._member_hashes = {item['content_hash'] for item in self._members_data}
        except Exception as e:
            print(f"⚠️  Failed to load members data: {e}")
            self._members_data = []
            self._member_hashes = set()

        try:
            if self.methods_data_path.exists():
                with open(self.methods_data_path, 'rb') as f:
                    self._methods_data = pickle.load(f)
                    self._method_hashes = {item['content_hash'] for item in self._methods_data}
        except Exception as e:
            print(f"⚠️  Failed to load methods data: {e}")
            self._methods_data = []
            self._method_hashes = set()

        self._cache_loaded = True

    async def _save_data(self):
        """Save data to disk."""
        try:
            with open(self.files_data_path, 'wb') as f:
                pickle.dump(self._files_data, f)
        except Exception as e:
            print(f"⚠️  Failed to save files data: {e}")

        try:
            with open(self.members_data_path, 'wb') as f:
                pickle.dump(self._members_data, f)
        except Exception as e:
            print(f"⚠️  Failed to save members data: {e}")

        try:
            with open(self.methods_data_path, 'wb') as f:
                pickle.dump(self._methods_data, f)
        except Exception as e:
            print(f"⚠️  Failed to save methods data: {e}")

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)

        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _search_data(self, data: List[Dict[str, Any]], embedding: List[float],
                     limit: int, threshold: float, result_type: str) -> List[SearchResult]:
        """Search through data using cosine similarity."""
        results = []

        for item in data:
            item_embedding = item.get('embedding', [])
            if not item_embedding:
                continue

            similarity = self._cosine_similarity(embedding, item_embedding)

            if similarity >= threshold:
                metadata = item.get('metadata', {})
                results.append(SearchResult(
                    id=item.get('id', ''),
                    type=result_type,
                    name=metadata.get('name', ''),
                    summary=item.get('summary', item.get('content', '')[:200] + '...' if len(item.get('content', '')) > 200 else item.get('content', '')),
                    file_path=item.get('file_path', metadata.get('file_path', '')),
                    score=similarity,
                    content=item.get('content', item.get('summary', ''))
                ))

        # Sort by similarity score (descending) and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    # IVectorStore interface implementation

    async def store_file_index(self, file_index: FileIndex) -> None:
        """Store file index with embedding."""
        await self._load_data()

        if file_index.content_hash not in self._file_hashes and file_index.content_embedding:
            file_data = {
                'id': file_index.id,
                'file_path': file_index.file_path,
                'content': file_index.content,
                'content_hash': file_index.content_hash,
                'embedding': file_index.content_embedding,
                'metadata': {
                    'id': file_index.id,
                    'file_path': file_index.file_path
                }
            }

            self._files_data.append(file_data)
            self._file_hashes.add(file_index.content_hash)
            await self._save_data()

    async def store_code_member(self, code_member: CodeMember) -> None:
        """Store code member with embedding."""
        await self._load_data()

        if code_member.content_hash not in self._member_hashes and code_member.summary_embedding:
            member_data = {
                'id': code_member.id,
                'file_id': code_member.file_id,
                'content_hash': code_member.content_hash,
                'summary': code_member.summary,
                'embedding': code_member.summary_embedding,
                'metadata': {
                    'id': code_member.id,
                    'file_id': code_member.file_id,
                    'type': code_member.type.value if isinstance(code_member.type, MemberType) else code_member.type,
                    'name': code_member.name
                }
            }

            self._members_data.append(member_data)
            self._member_hashes.add(code_member.content_hash)
            await self._save_data()

    async def store_code_method(self, code_method: CodeMethod) -> None:
        """Store code method with embedding."""
        await self._load_data()

        if code_method.content_hash not in self._method_hashes and code_method.summary_embedding:
            method_data = {
                'id': code_method.id,
                'member_id': code_method.member_id,
                'content_hash': code_method.content_hash,
                'summary': code_method.summary,
                'embedding': code_method.summary_embedding,
                'metadata': {
                    'id': code_method.id,
                    'member_id': code_method.member_id,
                    'name': code_method.name
                }
            }

            self._methods_data.append(method_data)
            self._method_hashes.add(code_method.content_hash)
            await self._save_data()

    async def get_file_by_id(self, file_id: str) -> Optional[FileIndex]:
        """Retrieve file index by ID."""
        await self._load_data()

        for item in self._files_data:
            if item.get('id') == file_id:
                return FileIndex(
                    id=item['id'],
                    file_path=item['file_path'],
                    content=item['content'],
                    content_hash=item['content_hash'],
                    content_embedding=item['embedding']
                )
        return None

    async def get_members_by_file_id(self, file_id: str) -> List[CodeMember]:
        """Retrieve all members for a specific file."""
        await self._load_data()

        members = []
        for item in self._members_data:
            if item.get('file_id') == file_id:
                member_type = MemberType(item['metadata']['type']) if 'type' in item['metadata'] else MemberType.CLASS
                members.append(CodeMember(
                    id=item['id'],
                    file_id=item['file_id'],
                    type=member_type,
                    name=item['metadata'].get('name', ''),
                    summary=item['summary'],
                    content_hash=item['content_hash'],
                    summary_embedding=item['embedding']
                ))
        return members

    async def get_methods_by_member_id(self, member_id: str) -> List[CodeMethod]:
        """Retrieve all methods for a specific member."""
        await self._load_data()

        methods = []
        for item in self._methods_data:
            if item.get('member_id') == member_id:
                methods.append(CodeMethod(
                    id=item['id'],
                    member_id=item['member_id'],
                    name=item['metadata'].get('name', ''),
                    summary=item['summary'],
                    content_hash=item['content_hash'],
                    summary_embedding=item['embedding']
                ))
        return methods

    async def file_exists(self, file_path: str, content_hash: str) -> Optional[FileIndex]:
        """Check if file exists with given path and content hash."""
        await self._load_data()

        for item in self._files_data:
            if item.get('file_path') == file_path and item.get('content_hash') == content_hash:
                return FileIndex(
                    id=item['id'],
                    file_path=item['file_path'],
                    content=item['content'],
                    content_hash=item['content_hash'],
                    content_embedding=item['embedding']
                )
        return None

    async def member_exists(self, content_hash: str) -> Optional[CodeMember]:
        """Check if member exists with given content hash."""
        await self._load_data()

        for item in self._members_data:
            if item.get('content_hash') == content_hash:
                member_type = MemberType(item['metadata']['type']) if 'type' in item['metadata'] else MemberType.CLASS
                return CodeMember(
                    id=item['id'],
                    file_id=item['file_id'],
                    type=member_type,
                    name=item['metadata'].get('name', ''),
                    summary=item['summary'],
                    content_hash=item['content_hash'],
                    summary_embedding=item['embedding']
                )
        return None

    async def method_exists(self, content_hash: str) -> Optional[CodeMethod]:
        """Check if method exists with given content hash."""
        await self._load_data()

        for item in self._methods_data:
            if item.get('content_hash') == content_hash:
                return CodeMethod(
                    id=item['id'],
                    member_id=item['member_id'],
                    name=item['metadata'].get('name', ''),
                    summary=item['summary'],
                    content_hash=item['content_hash'],
                    summary_embedding=item['embedding']
                )
        return None

    async def get_all_file_hashes(self) -> Set[str]:
        """Get all existing file content hashes for resume capability."""
        await self._load_data()
        return self._file_hashes.copy()

    async def search_files(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar files using the provided embedding."""
        await self._load_data()
        return self._search_data(self._files_data, embedding, limit, threshold, 'file')

    async def search_members(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar code members using the provided embedding."""
        await self._load_data()
        return self._search_data(self._members_data, embedding, limit, threshold, 'member')

    async def search_methods(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar code methods using the provided embedding."""
        await self._load_data()
        return self._search_data(self._methods_data, embedding, limit, threshold, 'method')

    async def get_statistics(self) -> dict:
        """Get statistics about the vector store."""
        await self._load_data()
        return {
            'total_files': len(self._files_data),
            'total_members': len(self._members_data),
            'total_methods': len(self._methods_data),
            'workspace_path': str(self.workspace_path)
        }
