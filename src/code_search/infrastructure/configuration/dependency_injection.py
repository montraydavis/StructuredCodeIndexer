"""Dependency injection configuration."""

import os
from typing import Tuple

from ...domain.interfaces import (
    IFileLoader, ICodeParser, IEmbeddingService, IVectorStore,
    ICodeIndexer, IAICodeParsingService, ISearchService
)
from ..storage.file import FileLoader
from ..parsing import CodeParser
from ..ai.semantic_kernel import SemanticKernelCodeParsingService
from ..ai.openai import OpenAIEmbeddingService
from ..storage.vector import SimpleVectorStore
from ...application.services import SearchService, CodeIndexer
from .settings import get_settings


def get_api_key() -> str:
    """Get OpenAI API key from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or not api_key.startswith('sk-'):
        raise ValueError("OPENAI_API_KEY not found or invalid. Please set your OpenAI API key in .env file.")
    return api_key


async def initialize_services(api_key: str) -> Tuple[ICodeIndexer, ISearchService]:
    """Initialize all required services for the application."""
    # Load configuration
    config = get_settings()

    print("🤖 Initializing AI code parsing service...")
    ai_parsing_service = SemanticKernelCodeParsingService(
        api_key=api_key,
        model_name=config.ai_model
    )
    await ai_parsing_service.initialize()
    print("✅ AI parsing service initialized")

    # Initialize core services with configuration
    file_loader = FileLoader(supported_extensions=config.supported_extensions)
    code_parser = CodeParser(ai_service=ai_parsing_service)
    embedding_service = OpenAIEmbeddingService(
        model_name=config.embedding_model,
        api_key=api_key
    )
    vector_store = SimpleVectorStore(workspace_path=config.vector_store_path)

    # Initialize search service
    search_service = SearchService(vector_store=vector_store, embedding_service=embedding_service)

    # Create indexer
    indexer = CodeIndexer(
        file_loader=file_loader,
        code_parser=code_parser,
        embedding_service=embedding_service,
        vector_store=vector_store,
        resume_mode=True
    )

    return indexer, search_service
