"""Direct OpenAI embedding service implementation."""

from typing import List, Dict, Optional
import hashlib
import openai

from ....domain.interfaces import IEmbeddingService


class OpenAIEmbeddingService(IEmbeddingService):
    """OpenAI embedding service implementation with direct API calls."""

    def __init__(self, model_name: str = "text-embedding-3-large", api_key: Optional[str] = None):
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
        # In-memory cache for embeddings to avoid regenerating
        self._embedding_cache: Dict[str, List[float]] = {}

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for a single text."""
        # Handle empty or None text
        if not text or not text.strip():
            # Return a zero vector for empty content
            return [0.0] * 3072

        # Use content hash as cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        try:
            client = openai.OpenAI()
            response = client.embeddings.create(
                model=self.model_name,
                input=[text]
            )

            embedding = response.data[0].embedding
            # Convert to list of floats
            embedding_list = [float(x) for x in embedding]

            # Cache the result
            self._embedding_cache[text_hash] = embedding_list
            return embedding_list

        except Exception as e:
            print(f"⚠️  Error generating embedding for text: {e}")
            # Return zero vector as fallback
            return [0.0] * 3072

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for multiple texts."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first and handle empty texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                embeddings.append([0.0] * 3072)
                continue

            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[text_hash])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder

        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                client = openai.OpenAI()
                response = client.embeddings.create(
                    model=self.model_name,
                    input=uncached_texts
                )

                for i, embedding_data in enumerate(response.data):
                    embedding_list = [float(x) for x in embedding_data.embedding]
                    actual_index = uncached_indices[i]
                    embeddings[actual_index] = embedding_list

                    # Cache the result
                    text_hash = hashlib.sha256(texts[actual_index].encode()).hexdigest()
                    self._embedding_cache[text_hash] = embedding_list

            except Exception as e:
                print(f"⚠️  Error generating batch embeddings: {e}")
                # Fill remaining with zero vectors
                for i in uncached_indices:
                    if embeddings[i] is None:
                        embeddings[i] = [0.0] * 3072

        return embeddings
