"""Search service implementation."""

from typing import List, Dict
import time

from ...domain.interfaces import ISearchService, IVectorStore, IEmbeddingService
from ...domain.models import SearchConfiguration, SearchResponse


class SearchService(ISearchService):
    """Implementation of search service using vector embeddings."""

    def __init__(self, vector_store: IVectorStore, embedding_service: IEmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    async def search_by_text(self, query: str, config: SearchConfiguration) -> SearchResponse:
        """Search using natural language text query."""
        start_time = time.time()

        try:
            print(f"🔍 Converting query to embeddings: '{query[:50]}{'...' if len(query) > 50 else ''}'")

            # Generate embedding for the query text (embedding service handles caching)
            query_embedding = await self.embedding_service.generate_embedding(query)

            print(f"✅ Embedding generated (dimension: {len(query_embedding)})")

            # Perform embedding-based search
            response = await self.search_by_embedding(query_embedding, config)
            response.query = query
            response.execution_time_ms = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            print(f"Error in text search: {e}")
            return SearchResponse(
                query=query,
                total_results=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def search_by_embedding(self, embedding: List[float], config: SearchConfiguration) -> SearchResponse:
        """Search using pre-computed embedding vector."""
        start_time = time.time()

        file_results = []
        member_results = []
        method_results = []

        try:
            print(f"🔍 Searching vector databases with embedding (dim: {len(embedding)})")
            print(f"📊 Search config - Files: {config.use_files}, Members: {config.use_members}, Methods: {config.use_methods}")
            print(f"🎯 Similarity threshold: {config.similarity_threshold}, Max results per type: {config.max_results_per_type}")

            # Search files if enabled
            if config.use_files:
                print("   📁 Searching files...")
                file_results = await self.vector_store.search_files(
                    embedding,
                    limit=config.max_results_per_type,
                    threshold=config.similarity_threshold
                )
                print(f"   📁 Found {len(file_results)} file matches")

            # Search members if enabled
            if config.use_members:
                print("   🏗️  Searching members...")
                member_results = await self.vector_store.search_members(
                    embedding,
                    limit=config.max_results_per_type,
                    threshold=config.similarity_threshold
                )
                print(f"   🏗️  Found {len(member_results)} member matches")

            # Search methods if enabled
            if config.use_methods:
                print("   ⚙️  Searching methods...")
                method_results = await self.vector_store.search_methods(
                    embedding,
                    limit=config.max_results_per_type,
                    threshold=config.similarity_threshold
                )
                print(f"   ⚙️  Found {len(method_results)} method matches")

            total_results = len(file_results) + len(member_results) + len(method_results)

            return SearchResponse(
                query="",  # Will be set by caller
                total_results=total_results,
                file_results=file_results,
                member_results=member_results,
                method_results=method_results,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            print(f"Error in embedding search: {e}")
            return SearchResponse(
                query="",
                total_results=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def search_similar_code(self, code_content: str, config: SearchConfiguration) -> SearchResponse:
        """Find code similar to the provided code content."""
        start_time = time.time()

        try:
            print(f"🔍 Finding similar code for: '{code_content[:50]}{'...' if len(code_content) > 50 else ''}'")

            # Generate embedding for the code content (embedding service handles caching)
            code_embedding = await self.embedding_service.generate_embedding(code_content)

            print(f"✅ Code embedding generated (dimension: {len(code_embedding)})")

            # Perform embedding-based search
            response = await self.search_by_embedding(code_embedding, config)
            response.query = f"Similar to: {code_content[:100]}..." if len(code_content) > 100 else code_content
            response.execution_time_ms = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            print(f"Error in similar code search: {e}")
            return SearchResponse(
                query=f"Similar to: {code_content[:100]}...",
                total_results=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def search_by_multiple_terms(self, terms: List[str], config: SearchConfiguration) -> SearchResponse:
        """Search using multiple terms combined into a single embedding."""
        start_time = time.time()

        try:
            # Combine terms into a single query
            combined_query = " ".join(terms)
            print(f"🔍 Multi-term search: {terms}")

            # Generate embedding for the combined query (embedding service handles caching)
            query_embedding = await self.embedding_service.generate_embedding(combined_query)

            # Perform embedding-based search
            response = await self.search_by_embedding(query_embedding, config)
            response.query = f"Multi-term: {', '.join(terms)}"
            response.execution_time_ms = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            print(f"Error in multi-term search: {e}")
            return SearchResponse(
                query=f"Multi-term: {', '.join(terms)}",
                total_results=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def search_with_context(self, query: str, context: str, config: SearchConfiguration) -> SearchResponse:
        """Search with additional context to improve embedding quality."""
        start_time = time.time()

        try:
            # Combine query with context for better embeddings
            enhanced_query = f"Context: {context}\nQuery: {query}"
            print(f"🔍 Context-enhanced search - Query: '{query}', Context: '{context[:30]}{'...' if len(context) > 30 else ''}'")

            # Generate embedding for the enhanced query (embedding service handles caching)
            query_embedding = await self.embedding_service.generate_embedding(enhanced_query)

            # Perform embedding-based search
            response = await self.search_by_embedding(query_embedding, config)
            response.query = f"{query} (with context)"
            response.execution_time_ms = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            print(f"Error in context search: {e}")
            return SearchResponse(
                query=f"{query} (with context)",
                total_results=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def clear_embedding_cache(self) -> None:
        """Clear the embedding service cache."""
        # Access the concrete implementation's cache if it exists
        if hasattr(self.embedding_service, '_embedding_cache'):
            cache = getattr(self.embedding_service, '_embedding_cache')
            if hasattr(cache, 'clear'):
                cache.clear()
                print("🗑️  Embedding cache cleared")
            else:
                print("⚠️  Embedding cache found but cannot be cleared")
        else:
            print("⚠️  No embedding cache found to clear")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the embedding cache."""
        embedding_cache_size = 0

        # Access the concrete implementation's cache if it exists
        if hasattr(self.embedding_service, '_embedding_cache'):
            cache = getattr(self.embedding_service, '_embedding_cache')
            if hasattr(cache, '__len__'):
                embedding_cache_size = len(cache)

        return {
            'embedding_cache_size': embedding_cache_size
        }
