"""Search-related domain models."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchConfiguration:
    """Configuration for search operations."""
    use_files: bool = True
    use_members: bool = True
    use_methods: bool = True
    max_results_per_type: int = 10
    similarity_threshold: float = 0.7


@dataclass
class SearchResult:
    """Individual search result."""
    id: str
    type: str  # 'file', 'member', 'method'
    name: str
    summary: str
    file_path: str = ""
    member_name: str = ""  # For methods, the parent member name
    score: float = 0.0
    content: str = ""  # Full content for files, summary for others


@dataclass
class SearchResponse:
    """Complete search response containing all results."""
    query: str
    total_results: int
    file_results: List[SearchResult] = field(default_factory=list)
    member_results: List[SearchResult] = field(default_factory=list)
    method_results: List[SearchResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    @property
    def all_results(self) -> List[SearchResult]:
        """Get all results sorted by score."""
        all_results = self.file_results + self.member_results + self.method_results
        return sorted(all_results, key=lambda x: x.score, reverse=True) 