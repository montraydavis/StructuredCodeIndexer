"""Vector store implementation using VectorDB."""

from typing import List, Optional, Set, Any
from pathlib import Path
import numpy as np

from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
from docarray.typing import ID  # Add proper ID type import

from ....domain.interfaces import IVectorStore
from ....domain.models import FileIndex, CodeMember, CodeMethod, SearchResult, MemberType
from .document_schemas import FileDoc, MemberDoc, MethodDoc


class VectorStore(IVectorStore):
    """Implementation of vector storage using VectorDB."""

    def __init__(self, workspace_path: str = "./vectordb_workspace", use_hnsw: bool = True):
        self.workspace_path = workspace_path
        self.use_hnsw = use_hnsw
        self._files_db: Any = None
        self._members_db: Any = None
        self._methods_db: Any = None

        # In-memory caches for existence checking
        self._file_hashes: Set[str] = set()
        self._member_hashes: Set[str] = set()
        self._method_hashes: Set[str] = set()
        self._cache_loaded = False

    async def _ensure_databases(self):
        """Initialize VectorDB instances if not already created."""
        if self._files_db is None:
            try:
                if self.use_hnsw:
                    self._files_db = HNSWVectorDB(workspace=f"{self.workspace_path}/files")
                else:
                    self._files_db = InMemoryExactNNVectorDB(workspace=f"{self.workspace_path}/files")
            except Exception as e:
                print(f"⚠️  Failed to initialize HNSW files DB, falling back to InMemory: {e}")
                self._files_db = InMemoryExactNNVectorDB(workspace=f"{self.workspace_path}/files")
                self.use_hnsw = False

        if self._members_db is None:
            try:
                if self.use_hnsw:
                    self._members_db = HNSWVectorDB(workspace=f"{self.workspace_path}/members")
                else:
                    self._members_db = InMemoryExactNNVectorDB(workspace=f"{self.workspace_path}/members")
            except Exception as e:
                print(f"⚠️  Failed to initialize HNSW members DB, falling back to InMemory: {e}")
                self._members_db = InMemoryExactNNVectorDB(workspace=f"{self.workspace_path}/members")
                self.use_hnsw = False

        if self._methods_db is None:
            try:
                if self.use_hnsw:
                    self._methods_db = HNSWVectorDB(workspace=f"{self.workspace_path}/methods")
                else:
                    self._methods_db = InMemoryExactNNVectorDB(workspace=f"{self.workspace_path}/methods")
            except Exception as e:
                print(f"⚠️  Failed to initialize HNSW methods DB, falling back to InMemory: {e}")
                self._methods_db = InMemoryExactNNVectorDB(workspace=f"{self.workspace_path}/methods")
                self.use_hnsw = False

    async def _load_existing_hashes(self):
        """Load existing content hashes into memory for fast existence checking."""
        if self._cache_loaded:
            return

        await self._ensure_databases()

        try:
            # Load file hashes
            dummy_file_query = FileDoc(
                file_path="",
                content="",
                content_hash="",
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            file_results = self._files_db.search(docs=[dummy_file_query], limit=10000)
            if hasattr(file_results, 'matches') and file_results.matches:
                for result in file_results.matches:
                    self._file_hashes.add(result.content_hash)

            # Load member hashes
            dummy_member_query = MemberDoc(
                file_id="",
                type="",
                name="",
                text_content="",
                content_hash="",
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            member_results = self._members_db.search(docs=[dummy_member_query], limit=10000)
            if hasattr(member_results, 'matches') and member_results.matches:
                for result in member_results.matches:
                    self._member_hashes.add(result.content_hash)

            # Load method hashes
            dummy_method_query = MethodDoc(
                member_id="",
                name="",
                text_content="",
                content_hash="",
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            method_results = self._methods_db.search(docs=[dummy_method_query], limit=10000)
            if hasattr(method_results, 'matches') and method_results.matches:
                for result in method_results.matches:
                    self._method_hashes.add(result.content_hash)

        except Exception as e:
            print(f"Warning: Could not load existing hashes: {e}")

        self._cache_loaded = True

    def _file_to_doc(self, file_index: FileIndex) -> FileDoc:
        """Convert FileIndex dataclass to VectorDB FileDoc."""
        return FileDoc(
            id=ID(file_index.id),  # Convert string to DocArray ID
            file_path=file_index.file_path,
            content=file_index.content,
            content_hash=file_index.content_hash,
            embedding=np.array(file_index.content_embedding, dtype=np.float32)  # type: ignore
        )

    def _member_to_doc(self, code_member: CodeMember) -> MemberDoc:
        """Convert CodeMember dataclass to VectorDB MemberDoc."""
        return MemberDoc(
            id=ID(code_member.id),  # Convert string to DocArray ID
            file_id=code_member.file_id,
            type=code_member.type.value,
            name=code_member.name,
            text_content=code_member.summary,
            content_hash=code_member.content_hash,
            embedding=np.array(code_member.summary_embedding, dtype=np.float32)  # type: ignore
        )

    def _method_to_doc(self, code_method: CodeMethod) -> MethodDoc:
        """Convert CodeMethod dataclass to VectorDB MethodDoc."""
        return MethodDoc(
            id=ID(code_method.id),  # Convert string to DocArray ID
            member_id=code_method.member_id,
            name=code_method.name,
            text_content=code_method.summary,
            content_hash=code_method.content_hash,
            embedding=np.array(code_method.summary_embedding, dtype=np.float32)  # type: ignore
        )

    def _doc_to_file(self, doc: FileDoc) -> FileIndex:
        """Convert VectorDB FileDoc to FileIndex dataclass."""
        return FileIndex(
            id=str(doc.id) if doc.id else "",  # Convert DocArray ID to string
            file_path=doc.file_path,
            content=doc.content,
            content_hash=doc.content_hash,
            content_embedding=doc.embedding.tolist() if hasattr(doc.embedding, 'tolist') else list(doc.embedding)
        )

    def _doc_to_member(self, doc: MemberDoc) -> CodeMember:
        """Convert VectorDB MemberDoc to CodeMember dataclass."""
        return CodeMember(
            id=str(doc.id) if doc.id else "",  # Convert DocArray ID to string
            file_id=doc.file_id,
            type=MemberType(doc.type),
            name=doc.name,
            summary=doc.text_content,
            content_hash=doc.content_hash,
            summary_embedding=doc.embedding.tolist() if hasattr(doc.embedding, 'tolist') else list(doc.embedding)
        )

    def _doc_to_method(self, doc: MethodDoc) -> CodeMethod:
        """Convert VectorDB MethodDoc to CodeMethod dataclass."""
        return CodeMethod(
            id=str(doc.id) if doc.id else "",  # Convert DocArray ID to string
            member_id=doc.member_id,
            name=doc.name,
            summary=doc.text_content,
            content_hash=doc.content_hash,
            summary_embedding=doc.embedding.tolist() if hasattr(doc.embedding, 'tolist') else list(doc.embedding)
        )

    async def store_file_index(self, file_index: FileIndex) -> None:
        """Store file index in VectorDB."""
        await self._ensure_databases()
        file_doc = self._file_to_doc(file_index)
        self._files_db.index(docs=[file_doc])
        # Update cache
        self._file_hashes.add(file_index.content_hash)

    async def store_code_member(self, code_member: CodeMember) -> None:
        """Store code member in VectorDB."""
        await self._ensure_databases()
        member_doc = self._member_to_doc(code_member)
        self._members_db.index(docs=[member_doc])
        # Update cache
        self._member_hashes.add(code_member.content_hash)

    async def store_code_method(self, code_method: CodeMethod) -> None:
        """Store code method in VectorDB."""
        await self._ensure_databases()
        method_doc = self._method_to_doc(code_method)
        self._methods_db.index(docs=[method_doc])
        # Update cache
        self._method_hashes.add(code_method.content_hash)

    async def get_file_by_id(self, file_id: str) -> Optional[FileIndex]:
        """Retrieve file index by ID."""
        await self._ensure_databases()
        try:
            # Create a dummy query vector for exact ID search
            dummy_query = FileDoc(
                file_path="",
                content="",
                content_hash="",
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            results = self._files_db.search(docs=[dummy_query], limit=1000)

            # Filter results to find exact ID match
            if hasattr(results, 'matches') and results.matches:
                for result in results.matches:
                    if hasattr(result, 'id') and result.id == file_id:
                        return self._doc_to_file(result)
            return None
        except Exception:
            return None

    async def get_members_by_file_id(self, file_id: str) -> List[CodeMember]:
        """Retrieve all members for a specific file."""
        await self._ensure_databases()
        try:
            # Create a dummy query vector for search
            dummy_query = MemberDoc(
                file_id=file_id,
                type="",
                name="",
                text_content="",
                content_hash="",
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            results = self._members_db.search(docs=[dummy_query], limit=1000)

            # Filter results to find members with matching file_id
            members = []
            if hasattr(results, 'matches') and results.matches:
                for result in results.matches:
                    if hasattr(result, 'file_id') and result.file_id == file_id:
                        members.append(self._doc_to_member(result))
            return members
        except Exception:
            return []

    async def get_methods_by_member_id(self, member_id: str) -> List[CodeMethod]:
        """Retrieve all methods for a specific member."""
        await self._ensure_databases()
        try:
            # Create a dummy query vector for search
            dummy_query = MethodDoc(
                member_id=member_id,
                name="",
                text_content="",
                content_hash="",
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            results = self._methods_db.search(docs=[dummy_query], limit=1000)

            # Filter results to find methods with matching member_id
            methods = []
            if hasattr(results, 'matches') and results.matches:
                for result in results.matches:
                    if hasattr(result, 'member_id') and result.member_id == member_id:
                        methods.append(self._doc_to_method(result))
            return methods
        except Exception:
            return []

    async def file_exists(self, file_path: str, content_hash: str) -> Optional[FileIndex]:
        """Check if file exists with given path and content hash."""
        await self._load_existing_hashes()

        if content_hash not in self._file_hashes:
            return None

        # File with this hash exists, now find it by searching
        try:
            await self._ensure_databases()
            dummy_query = FileDoc(
                file_path=file_path,
                content="",
                content_hash=content_hash,
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            results = self._files_db.search(docs=[dummy_query], limit=1000)

            if hasattr(results, 'matches') and results.matches:
                for result in results.matches:
                    if (hasattr(result, 'content_hash') and hasattr(result, 'file_path') and
                        result.content_hash == content_hash and result.file_path == file_path):
                        return self._doc_to_file(result)
            return None
        except Exception:
            return None

    async def member_exists(self, content_hash: str) -> Optional[CodeMember]:
        """Check if member exists with given content hash."""
        await self._load_existing_hashes()

        if content_hash not in self._member_hashes:
            return None

        try:
            await self._ensure_databases()
            dummy_query = MemberDoc(
                file_id="",
                type="",
                name="",
                text_content="",
                content_hash=content_hash,
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            results = self._members_db.search(docs=[dummy_query], limit=1000)

            if hasattr(results, 'matches') and results.matches:
                for result in results.matches:
                    if hasattr(result, 'content_hash') and result.content_hash == content_hash:
                        return self._doc_to_member(result)
            return None
        except Exception:
            return None

    async def method_exists(self, content_hash: str) -> Optional[CodeMethod]:
        """Check if method exists with given content hash."""
        await self._load_existing_hashes()

        if content_hash not in self._method_hashes:
            return None

        try:
            await self._ensure_databases()
            dummy_query = MethodDoc(
                member_id="",
                name="",
                text_content="",
                content_hash=content_hash,
                embedding=np.zeros(3072, dtype=np.float32)  # type: ignore
            )
            results = self._methods_db.search(docs=[dummy_query], limit=1000)

            if hasattr(results, 'matches') and results.matches:
                for result in results.matches:
                    if hasattr(result, 'content_hash') and result.content_hash == content_hash:
                        return self._doc_to_method(result)
            return None
        except Exception:
            return None

    async def get_all_file_hashes(self) -> Set[str]:
        """Get all existing file content hashes for resume capability."""
        await self._load_existing_hashes()
        return self._file_hashes.copy()

    async def search_files(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search files by embedding similarity."""
        await self._ensure_databases()
        try:
            # Create query with the provided embedding
            query_doc = FileDoc(
                file_path="",
                content="",
                content_hash="",
                embedding=np.array(embedding, dtype=np.float32)  # type: ignore
            )

            results = self._files_db.search(docs=[query_doc], limit=limit)
            search_results = []

            if hasattr(results, 'matches') and results.matches:
                for idx, match in enumerate(results.matches):
                    # Get score - different libraries may have different formats
                    score = 0.0
                    if hasattr(results, 'scores') and results.scores and len(results.scores) > idx:
                        score = float(results.scores[idx])
                    elif hasattr(match, 'score'):
                        score = float(match.score)

                    if score >= threshold:
                        search_result = SearchResult(
                            id=getattr(match, 'id', ''),
                            type="file",
                            name=Path(match.file_path).name,
                            summary=f"File: {match.file_path}",
                            file_path=match.file_path,
                            score=score,
                            content=match.content[:500] + "..." if len(match.content) > 500 else match.content
                        )
                        search_results.append(search_result)

            return search_results
        except Exception as e:
            print(f"Error searching files: {e}")
            return []

    async def search_members(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search members by embedding similarity."""
        await self._ensure_databases()
        try:
            # Create query with the provided embedding
            query_doc = MemberDoc(
                file_id="",
                type="",
                name="",
                text_content="",
                content_hash="",
                embedding=np.array(embedding, dtype=np.float32)  # type: ignore
            )

            results = self._members_db.search(docs=[query_doc], limit=limit)
            search_results = []

            if hasattr(results, 'matches') and results.matches:
                for idx, match in enumerate(results.matches):
                    # Get score
                    score = 0.0
                    if hasattr(results, 'scores') and results.scores and len(results.scores) > idx:
                        score = float(results.scores[idx])
                    elif hasattr(match, 'score'):
                        score = float(match.score)

                    if score >= threshold:
                        # Get file info for context
                        file_info = await self.get_file_by_id(match.file_id)
                        file_path = file_info.file_path if file_info else "Unknown"

                        search_result = SearchResult(
                            id=getattr(match, 'id', ''),
                            type="member",
                            name=match.name,
                            summary=match.text_content,
                            file_path=file_path,
                            score=score,
                            content=match.text_content
                        )
                        search_results.append(search_result)

            return search_results
        except Exception as e:
            print(f"Error searching members: {e}")
            return []

    async def search_methods(self, embedding: List[float], limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search methods by embedding similarity."""
        await self._ensure_databases()
        try:
            # Create query with the provided embedding
            query_doc = MethodDoc(
                member_id="",
                name="",
                text_content="",
                content_hash="",
                embedding=np.array(embedding, dtype=np.float32)  # type: ignore
            )

            results = self._methods_db.search(docs=[query_doc], limit=limit)
            search_results = []

            if hasattr(results, 'matches') and results.matches:
                for idx, match in enumerate(results.matches):
                    # Get score
                    score = 0.0
                    if hasattr(results, 'scores') and results.scores and len(results.scores) > idx:
                        score = float(results.scores[idx])
                    elif hasattr(match, 'score'):
                        score = float(match.score)

                    if score >= threshold:
                        # Get member and file info for context (simplified)
                        member_name = "Unknown"
                        file_path = "Unknown"

                        search_result = SearchResult(
                            id=getattr(match, 'id', ''),
                            type="method",
                            name=match.name,
                            summary=match.text_content,
                            file_path=file_path,
                            member_name=member_name,
                            score=score,
                            content=match.text_content
                        )
                        search_results.append(search_result)

            return search_results
        except Exception as e:
            print(f"Error searching methods: {e}")
            return []
