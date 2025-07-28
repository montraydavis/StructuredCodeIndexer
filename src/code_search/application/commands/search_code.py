"""Search code command handler."""

from dataclasses import dataclass
from typing import List

from ...domain.interfaces import ISearchService
from ...domain.models import SearchConfiguration, SearchResponse


@dataclass
class SearchCodeCommand:
    """Command to search for code."""
    query: str
    use_files: bool = True
    use_members: bool = True
    use_methods: bool = True
    max_results_per_type: int = 10
    similarity_threshold: float = 0.7


@dataclass
class SearchSimilarCodeCommand:
    """Command to search for similar code."""
    code_content: str
    use_files: bool = True
    use_members: bool = True
    use_methods: bool = True
    max_results_per_type: int = 10
    similarity_threshold: float = 0.7


@dataclass
class SearchMultipleTermsCommand:
    """Command to search using multiple terms."""
    terms: List[str]
    use_files: bool = True
    use_members: bool = True
    use_methods: bool = True
    max_results_per_type: int = 10
    similarity_threshold: float = 0.7


class SearchCodeCommandHandler:
    """Handler for search code commands."""

    def __init__(self, search_service: ISearchService):
        self._search_service = search_service

    async def handle_text_search(self, command: SearchCodeCommand) -> SearchResponse:
        """Handle text-based search command."""
        config = SearchConfiguration(
            use_files=command.use_files,
            use_members=command.use_members,
            use_methods=command.use_methods,
            max_results_per_type=command.max_results_per_type,
            similarity_threshold=command.similarity_threshold
        )

        return await self._search_service.search_by_text(command.query, config)

    async def handle_similar_code_search(self, command: SearchSimilarCodeCommand) -> SearchResponse:
        """Handle similar code search command."""
        config = SearchConfiguration(
            use_files=command.use_files,
            use_members=command.use_members,
            use_methods=command.use_methods,
            max_results_per_type=command.max_results_per_type,
            similarity_threshold=command.similarity_threshold
        )

        return await self._search_service.search_similar_code(command.code_content, config)

    async def handle_multiple_terms_search(self, command: SearchMultipleTermsCommand) -> SearchResponse:
        """Handle multiple terms search command."""
        config = SearchConfiguration(
            use_files=command.use_files,
            use_members=command.use_members,
            use_methods=command.use_methods,
            max_results_per_type=command.max_results_per_type,
            similarity_threshold=command.similarity_threshold
        )

        return await self._search_service.search_by_multiple_terms(command.terms, config)
