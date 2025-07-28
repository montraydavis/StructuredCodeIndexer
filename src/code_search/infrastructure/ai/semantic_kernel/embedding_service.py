"""Embedding service implementation using OpenAI."""

from typing import List, Dict, Optional
import hashlib

from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

from ....domain.interfaces import IEmbeddingService


class EmbeddingService(IEmbeddingService):
    """Implementation of embedding service using OpenAI embeddings."""

    def __init__(self, model_name: str = "text-embedding-3-large", api_key: Optional[str] = None):
        if api_key:
            self._embedding_service = OpenAITextEmbedding(ai_model_id=model_name, api_key=api_key)
        else:
            # Try to get from environment
            self._embedding_service = OpenAITextEmbedding(ai_model_id=model_name)
        # In-memory cache for embeddings to avoid regenerating
        self._embedding_cache: Dict[str, List[float]] = {}

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for a single text."""
        # Use content hash as cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        result = await self._embedding_service.generate_embeddings([text])
        # Convert numpy floats to Python floats for Pydantic validation
        embedding = [float(x) for x in result[0]]

        # Cache the result
        self._embedding_cache[text_hash] = embedding
        return embedding

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for multiple texts."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[text_hash])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder

        # Generate embeddings for uncached texts
        if uncached_texts:
            results = await self._embedding_service.generate_embeddings(uncached_texts)
            for i, embedding in enumerate(results):
                converted_embedding = [float(x) for x in embedding]
                actual_index = uncached_indices[i]
                embeddings[actual_index] = converted_embedding

                # Cache the result
                text_hash = hashlib.sha256(texts[actual_index].encode()).hexdigest()
                self._embedding_cache[text_hash] = converted_embedding

        return embeddings
