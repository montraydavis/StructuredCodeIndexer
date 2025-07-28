"""
Search diagnostics script to identify why search is returning 0 results.
"""

import asyncio
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from src.code_search.infrastructure.configuration import initialize_services, get_api_key, get_settings
from src.code_search.infrastructure.storage.vector import SimpleVectorStore
from src.code_search.infrastructure.ai.openai import OpenAIEmbeddingService


async def diagnose_vector_store():
    """Diagnose vector store data and search issues."""
    print("🔬 SEARCH DIAGNOSTICS")
    print("=" * 50)

    # Get configuration
    config = get_settings()
    api_key = get_api_key()

    print(f"📁 Vector store path: {config.vector_store_path}")
    print(f"🎯 Similarity threshold: {config.similarity_threshold}")

    # Initialize vector store
    vector_store = SimpleVectorStore(workspace_path=config.vector_store_path)
    embedding_service = OpenAIEmbeddingService(
        model_name=config.embedding_model,
        api_key=api_key
    )

    # Force load data
    await vector_store._load_data()

    print(f"\n📊 Data Status:")
    print(f"   Files in memory: {len(vector_store._files_data)}")
    print(f"   Members in memory: {len(vector_store._members_data)}")
    print(f"   Methods in memory: {len(vector_store._methods_data)}")

    # Check file paths exist
    files_exist = vector_store.files_data_path.exists()
    members_exist = vector_store.members_data_path.exists()
    methods_exist = vector_store.methods_data_path.exists()

    print(f"\n💾 File System Status:")
    print(f"   files_data.pkl exists: {files_exist}")
    print(f"   members_data.pkl exists: {members_exist}")
    print(f"   methods_data.pkl exists: {methods_exist}")

    if files_exist:
        print(f"   files_data.pkl size: {vector_store.files_data_path.stat().st_size} bytes")
    if members_exist:
        print(f"   members_data.pkl size: {vector_store.members_data_path.stat().st_size} bytes")
    if methods_exist:
        print(f"   methods_data.pkl size: {vector_store.methods_data_path.stat().st_size} bytes")

    # Check data content
    if vector_store._files_data:
        print(f"\n📁 Sample File Data:")
        sample_file = vector_store._files_data[0]
        print(f"   ID: {sample_file.get('id', 'N/A')}")
        print(f"   Path: {sample_file.get('file_path', 'N/A')}")
        print(f"   Content length: {len(sample_file.get('content', ''))}")
        print(f"   Embedding length: {len(sample_file.get('embedding', []))}")
        print(f"   Embedding type: {type(sample_file.get('embedding', []))}")

        # Check if embedding is valid
        embedding = sample_file.get('embedding', [])
        if embedding:
            embedding_array = np.array(embedding)
            print(f"   Embedding shape: {embedding_array.shape}")
            print(f"   Embedding mean: {embedding_array.mean():.6f}")
            print(f"   Embedding std: {embedding_array.std():.6f}")
            print(f"   Embedding norm: {np.linalg.norm(embedding_array):.6f}")

    if vector_store._members_data:
        print(f"\n🏗️ Sample Member Data:")
        sample_member = vector_store._members_data[0]
        print(f"   ID: {sample_member.get('id', 'N/A')}")
        print(f"   Name: {sample_member.get('metadata', {}).get('name', 'N/A')}")
        print(f"   Summary: {sample_member.get('summary', 'N/A')[:50]}...")
        print(f"   Embedding length: {len(sample_member.get('embedding', []))}")

    # Test similarity calculation
    print(f"\n🧮 Testing Similarity Calculations:")

    if vector_store._files_data or vector_store._members_data:
        # Generate test query embedding
        test_query = "prompt"
        query_embedding = await embedding_service.generate_embedding(test_query)
        print(f"   Query: '{test_query}'")
        print(f"   Query embedding length: {len(query_embedding)}")

        # Test against file embeddings
        if vector_store._files_data:
            print(f"\n   📁 File Similarity Scores:")
            for i, file_data in enumerate(vector_store._files_data[:3]):  # Test first 3
                file_embedding = file_data.get('embedding', [])
                if file_embedding:
                    similarity = vector_store._cosine_similarity(query_embedding, file_embedding)
                    print(f"      File {i+1} ({file_data.get('file_path', 'Unknown')}): {similarity:.6f}")
                    print(f"         Threshold {config.similarity_threshold}: {'✅ PASS' if similarity >= config.similarity_threshold else '❌ FAIL'}")

        # Test against member embeddings
        if vector_store._members_data:
            print(f"\n   🏗️ Member Similarity Scores:")
            for i, member_data in enumerate(vector_store._members_data[:3]):  # Test first 3
                member_embedding = member_data.get('embedding', [])
                if member_embedding:
                    similarity = vector_store._cosine_similarity(query_embedding, member_embedding)
                    member_name = member_data.get('metadata', {}).get('name', f'Member {i+1}')
                    print(f"      {member_name}: {similarity:.6f}")
                    print(f"         Threshold {config.similarity_threshold}: {'✅ PASS' if similarity >= config.similarity_threshold else '❌ FAIL'}")

    # Test with lower threshold
    print(f"\n🔍 Testing with Lower Threshold (0.1):")

    if vector_store._files_data:
        query_embedding = await embedding_service.generate_embedding("prompt")
        file_results = await vector_store.search_files(query_embedding, limit=3, threshold=0.1)
        print(f"   📁 Files found with threshold 0.1: {len(file_results)}")
        for result in file_results:
            print(f"      {result.name}: {result.score:.6f}")

    if vector_store._members_data:
        member_results = await vector_store.search_members(query_embedding, limit=3, threshold=0.1)
        print(f"   🏗️ Members found with threshold 0.1: {len(member_results)}")
        for result in member_results:
            print(f"      {result.name}: {result.score:.6f}")

    # Test embedding cache
    print(f"\n🗂️ Embedding Cache Status:")
    if hasattr(embedding_service, '_embedding_cache'):
        cache_size = len(embedding_service._embedding_cache)
        print(f"   Cache size: {cache_size} entries")
        if cache_size > 0:
            # Show a few cache keys
            cache_keys = list(embedding_service._embedding_cache.keys())[:3]
            for key in cache_keys:
                print(f"   Sample key: {key[:16]}...")
    else:
        print("   No embedding cache found")


async def test_specific_search():
    """Test search with specific terms that should match the indexed content."""
    print(f"\n🎯 Testing Specific Search Terms:")
    print("-" * 40)

    api_key = get_api_key()
    indexer, search_service = await initialize_services(api_key)

    from src.code_search.application.commands import SearchCodeCommand, SearchCodeCommandHandler
    search_handler = SearchCodeCommandHandler(search_service)

    # Test terms that should definitely match based on file names
    test_terms = [
        "PromptResult",
        "PullRequest",
        "Vector",
        "class",
        "model",
        "result"
    ]

    for term in test_terms:
        print(f"\n🔍 Testing: '{term}'")

        command = SearchCodeCommand(
            query=term,
            max_results_per_type=3,
            similarity_threshold=0.1  # Very low threshold
        )

        response = await search_handler.handle_text_search(command)
        print(f"   Results: {response.total_results} (Files: {len(response.file_results)}, Members: {len(response.member_results)}, Methods: {len(response.method_results)})")

        if response.all_results:
            top_result = response.all_results[0]
            print(f"   Top result: {top_result.name} ({top_result.type}) - Score: {top_result.score:.6f}")


async def main():
    """Run diagnostics."""
    load_dotenv()

    try:
        await diagnose_vector_store()
        await test_specific_search()

        print(f"\n✅ Diagnostics completed!")

    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
