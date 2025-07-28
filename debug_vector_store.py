"""Debug script for testing vector store functionality."""

import asyncio
import numpy as np
from dotenv import load_dotenv

from src.code_search.infrastructure.storage.vector import SimpleVectorStore
from src.code_search.infrastructure.ai.openai import OpenAIEmbeddingService
from src.code_search.infrastructure.configuration import get_api_key, get_settings
from src.code_search.domain.models import FileIndex, CodeMember, CodeMethod, MemberType


async def create_test_data():
    """Create test data for vector store testing."""
    # Create test file
    test_file = FileIndex(
        file_path="test/sample.py",
        content="class TestClass:\n    def test_method(self):\n        pass",
        content_embedding=[0.1] * 3072  # Mock embedding
    )

    # Create test member
    test_member = CodeMember(
        file_id=test_file.id,
        type=MemberType.CLASS,
        name="TestClass",
        summary="A test class for debugging purposes",
        summary_embedding=[0.2] * 3072  # Mock embedding
    )

    # Create test method
    test_method = CodeMethod(
        member_id=test_member.id,
        name="test_method",
        summary="A test method that does nothing",
        summary_embedding=[0.3] * 3072  # Mock embedding
    )

    test_member.methods = [test_method]
    test_file.members = [test_member]

    return test_file, test_member, test_method


async def test_vector_store_basic_operations(vector_store: SimpleVectorStore):
    """Test basic vector store operations."""
    print(f"\n🗄️ Testing basic vector store operations")
    print("-" * 50)

    # Create test data
    test_file, test_member, test_method = await create_test_data()

    try:
        # Test storing
        print("1. Testing storage operations...")
        await vector_store.store_file_index(test_file)
        print("   ✅ File stored successfully")

        await vector_store.store_code_member(test_member)
        print("   ✅ Member stored successfully")

        await vector_store.store_code_method(test_method)
        print("   ✅ Method stored successfully")

        # Test retrieval
        print("\n2. Testing retrieval operations...")
        retrieved_file = await vector_store.get_file_by_id(test_file.id)
        if retrieved_file:
            print(f"   ✅ File retrieved: {retrieved_file.file_path}")
        else:
            print("   ❌ File not found")

        members = await vector_store.get_members_by_file_id(test_file.id)
        print(f"   ✅ Members retrieved: {len(members)} found")

        methods = await vector_store.get_methods_by_member_id(test_member.id)
        print(f"   ✅ Methods retrieved: {len(methods)} found")

        # Test existence checking
        print("\n3. Testing existence checks...")
        file_exists = await vector_store.file_exists(test_file.file_path, test_file.content_hash)
        print(f"   ✅ File exists check: {'Found' if file_exists else 'Not found'}")

        member_exists = await vector_store.member_exists(test_member.content_hash)
        print(f"   ✅ Member exists check: {'Found' if member_exists else 'Not found'}")

        method_exists = await vector_store.method_exists(test_method.content_hash)
        print(f"   ✅ Method exists check: {'Found' if method_exists else 'Not found'}")

        return True

    except Exception as e:
        print(f"   ❌ Vector store operation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vector_search_operations(vector_store: SimpleVectorStore, embedding_service: OpenAIEmbeddingService):
    """Test vector search operations."""
    print(f"\n🔍 Testing vector search operations")
    print("-" * 50)

    try:
        # Generate embeddings for search queries
        print("1. Generating search embeddings...")

        search_queries = [
            "test class implementation",
            "method functionality",
            "database connection"
        ]

        for query in search_queries:
            print(f"\n   🔍 Searching for: '{query}'")

            # Generate embedding for query
            query_embedding = await embedding_service.generate_embedding(query)
            print(f"      Embedding dimension: {len(query_embedding)}")

            # Search files
            file_results = await vector_store.search_files(
                query_embedding,
                limit=5,
                threshold=0.5
            )
            print(f"      📁 File results: {len(file_results)}")

            # Search members
            member_results = await vector_store.search_members(
                query_embedding,
                limit=5,
                threshold=0.5
            )
            print(f"      🏗️  Member results: {len(member_results)}")

            # Search methods
            method_results = await vector_store.search_methods(
                query_embedding,
                limit=5,
                threshold=0.5
            )
            print(f"      ⚙️  Method results: {len(method_results)}")

            # Show top results
            all_results = file_results + member_results + method_results
            if all_results:
                all_results.sort(key=lambda x: x.score, reverse=True)
                top_result = all_results[0]
                print(f"      🥇 Top result: {top_result.name} (score: {top_result.score:.3f})")

        return True

    except Exception as e:
        print(f"   ❌ Vector search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vector_store_performance(vector_store: SimpleVectorStore, embedding_service: OpenAIEmbeddingService):
    """Test vector store performance with multiple items."""
    print(f"\n⚡ Testing vector store performance")
    print("-" * 50)

    try:
        import time

        # Create multiple test items
        print("1. Creating test data...")
        num_items = 10
        test_files = []
        test_members = []
        test_methods = []

        for i in range(num_items):
            # Generate embeddings for variety
            file_text = f"File content {i} with various code structures"
            member_text = f"Class {i} for testing purposes with different functionality"
            method_text = f"Method {i} that performs specific operations"

            file_embedding = await embedding_service.generate_embedding(file_text)
            member_embedding = await embedding_service.generate_embedding(member_text)
            method_embedding = await embedding_service.generate_embedding(method_text)

            # Create test objects
            test_file = FileIndex(
                file_path=f"test/file_{i}.py",
                content=f"# Test file {i}\nclass TestClass{i}:\n    pass",
                content_embedding=file_embedding
            )

            test_member = CodeMember(
                file_id=test_file.id,
                type=MemberType.CLASS,
                name=f"TestClass{i}",
                summary=f"Test class {i} for performance testing",
                summary_embedding=member_embedding
            )

            test_method = CodeMethod(
                member_id=test_member.id,
                name=f"test_method_{i}",
                summary=f"Test method {i} for performance evaluation",
                summary_embedding=method_embedding
            )

            test_files.append(test_file)
            test_members.append(test_member)
            test_methods.append(test_method)

        # Test bulk storage performance
        print(f"\n2. Testing bulk storage of {num_items} items...")

        start_time = time.time()
        for test_file in test_files:
            await vector_store.store_file_index(test_file)
        storage_time = time.time() - start_time
        print(f"   📁 File storage: {storage_time:.3f}s ({num_items/storage_time:.1f} items/sec)")

        start_time = time.time()
        for test_member in test_members:
            await vector_store.store_code_member(test_member)
        storage_time = time.time() - start_time
        print(f"   🏗️  Member storage: {storage_time:.3f}s ({num_items/storage_time:.1f} items/sec)")

        start_time = time.time()
        for test_method in test_methods:
            await vector_store.store_code_method(test_method)
        storage_time = time.time() - start_time
        print(f"   ⚙️  Method storage: {storage_time:.3f}s ({num_items/storage_time:.1f} items/sec)")

        # Test search performance
        print(f"\n3. Testing search performance...")

        search_embedding = await embedding_service.generate_embedding("test class method")

        start_time = time.time()
        file_results = await vector_store.search_files(search_embedding, limit=5)
        search_time = time.time() - start_time
        print(f"   📁 File search: {search_time*1000:.1f}ms, {len(file_results)} results")

        start_time = time.time()
        member_results = await vector_store.search_members(search_embedding, limit=5)
        search_time = time.time() - start_time
        print(f"   🏗️  Member search: {search_time*1000:.1f}ms, {len(member_results)} results")

        start_time = time.time()
        method_results = await vector_store.search_methods(search_embedding, limit=5)
        search_time = time.time() - start_time
        print(f"   ⚙️  Method search: {search_time*1000:.1f}ms, {len(method_results)} results")

        return True

    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vector_store_edge_cases(vector_store: SimpleVectorStore):
    """Test vector store edge cases."""
    print(f"\n🧪 Testing edge cases")
    print("-" * 30)

    try:
        # Test with non-existent IDs
        print("1. Testing non-existent lookups...")

        fake_id = "non-existent-id"
        result = await vector_store.get_file_by_id(fake_id)
        print(f"   📁 Non-existent file: {'Found' if result else 'Not found (expected)'}")

        members = await vector_store.get_members_by_file_id(fake_id)
        print(f"   🏗️  Members for fake file: {len(members)} (expected: 0)")

        methods = await vector_store.get_methods_by_member_id(fake_id)
        print(f"   ⚙️  Methods for fake member: {len(methods)} (expected: 0)")

        # Test with empty embeddings
        print("\n2. Testing empty embeddings...")
        try:
            empty_results = await vector_store.search_files([], limit=1)
            print(f"   📁 Empty embedding search: {len(empty_results)} results")
        except Exception as e:
            print(f"   📁 Empty embedding search failed (expected): {type(e).__name__}")

        # Test with invalid thresholds
        print("\n3. Testing edge threshold values...")
        test_embedding = [0.1] * 3072

        high_threshold_results = await vector_store.search_files(test_embedding, threshold=0.99)
        print(f"   📁 High threshold (0.99): {len(high_threshold_results)} results")

        zero_threshold_results = await vector_store.search_files(test_embedding, threshold=0.0)
        print(f"   📁 Zero threshold (0.0): {len(zero_threshold_results)} results")

        return True

    except Exception as e:
        print(f"   ❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Debug vector store functionality."""
    load_dotenv()

    try:
        # Get API key and configuration
        api_key = get_api_key()
        config = get_settings()

        print(f"🔧 DEBUG MODE: Vector Store")
        print(f"🔑 API key configured: {'Yes' if api_key else 'No'}")
        print(f"⚙️  Configuration loaded: {config.vector_store_path}")

        # Initialize services with configuration
        vector_store = SimpleVectorStore(workspace_path=config.vector_store_path)
        embedding_service = OpenAIEmbeddingService(
            model_name=config.embedding_model,
            api_key=api_key
        )

        print("\n" + "="*60)
        print("🗄️ VECTOR STORE DEBUG TESTS")
        print("="*60)

        # Run tests
        test_results = []

        test_results.append(await test_vector_store_basic_operations(vector_store))
        test_results.append(await test_vector_search_operations(vector_store, embedding_service))
        test_results.append(await test_vector_store_performance(vector_store, embedding_service))
        test_results.append(await test_vector_store_edge_cases(vector_store))

        # Summary
        print(f"\n📊 Test Summary")
        print("-" * 25)
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        print(f"   ✅ Passed: {passed_tests}/{total_tests}")
        print(f"   {'🎉 All tests passed!' if passed_tests == total_tests else '⚠️  Some tests failed'}")

        # Cache stats
        if hasattr(embedding_service, '_embedding_cache'):
            cache_size = len(embedding_service._embedding_cache)
            print(f"   🗂️  Embedding cache: {cache_size} items")

        print("\n✅ Vector store debug session completed!")

    except Exception as e:
        print(f"❌ Vector Store Debug Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
