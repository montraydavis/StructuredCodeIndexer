"""Main CLI application entry point."""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

from ...infrastructure.configuration import initialize_services, get_api_key
from ...application.commands import (
    IndexProjectCommand, IndexProjectCommandHandler,
    SearchCodeCommand, SearchCodeCommandHandler
)
from ...domain.models import SearchConfiguration


async def demonstrate_basic_search(search_handler: SearchCodeCommandHandler) -> None:
    """Demonstrate basic text search functionality."""
    print("\n🔤 Single Query Search")
    print("-" * 25)

    search_queries = ["prompt"]

    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")

        command = SearchCodeCommand(
            query=query,
            max_results_per_type=3,
            similarity_threshold=0.6
        )

        response = await search_handler.handle_text_search(command)

        print(f"   📊 Found {response.total_results} results in {response.execution_time_ms:.1f}ms")

        # Show top result
        for result in response.all_results[:1]:
            print(f"   📄 {result.type.upper()}: {result.name}")
            print(f"      📁 {result.file_path}")
            print(f"      ⭐ Score: {result.score:.3f}")


async def demonstrate_advanced_search(search_handler: SearchCodeCommandHandler) -> None:
    """Demonstrate advanced search capabilities."""
    print("\n🔀 Advanced Search Demonstrations")
    print("-" * 35)

    # Similar code search
    print("\n🔄 Similar Code Search")
    print("-" * 22)
    sample_code = """
    public async Task<User> GetUserByIdAsync(int userId)
    {
        return await _repository.GetByIdAsync(userId);
    }
    """
    print(f"\n🔍 Finding code similar to async repository pattern")

    from ...application.commands.search_code import SearchSimilarCodeCommand
    similar_command = SearchSimilarCodeCommand(
        code_content=sample_code,
        max_results_per_type=3,
        similarity_threshold=0.6
    )

    response = await search_handler.handle_similar_code_search(similar_command)

    print(f"   📊 Found {response.total_results} results in {response.execution_time_ms:.1f}ms")
    for result in response.all_results[:2]:
        print(f"   📄 {result.type.upper()}: {result.name}")
        print(f"      ⭐ Score: {result.score:.3f}")


async def demonstrate_database_specific_search(search_handler: SearchCodeCommandHandler) -> None:
    """Demonstrate database-specific search configurations."""
    print("\n⚙️ Database-Specific Search")
    print("-" * 32)

    configs = [
        ("Files Only", {"use_files": True, "use_members": False, "use_methods": False}),
        ("Members Only", {"use_files": False, "use_members": True, "use_methods": False}),
        ("Methods Only", {"use_files": False, "use_members": False, "use_methods": True})
    ]

    query = "prompt"
    for config_name, search_config in configs:
        print(f"\n🔍 {config_name} search for: '{query}'")

        command = SearchCodeCommand(
            query=query,
            **search_config
        )

        response = await search_handler.handle_text_search(command)

        print(f"   📊 Total results: {response.total_results}")
        print(f"   ⏱️  Time: {response.execution_time_ms:.1f}ms")

        if response.all_results:
            top_result = response.all_results[0]
            print(f"   🥇 Top result: {top_result.name} ({top_result.type}) - Score: {top_result.score:.3f}")


async def main():
    """Main CLI application function."""
    load_dotenv()

    try:
        # Get API key and initialize services
        api_key = get_api_key()

        project_directory = "C:\\Users\\montr\\Downloads\\ADOPullRequestTools\\ADOPullRequestTools\\Models"

        # Initialize all services
        indexer, search_service = await initialize_services(api_key)

        # Create command handlers
        index_handler = IndexProjectCommandHandler(indexer)
        search_handler = SearchCodeCommandHandler(search_service)

        # Run indexing (commented out - uncomment to re-index)
        # index_command = IndexProjectCommand(project_directory=project_directory)
        # await index_handler.handle(index_command)

        # Run search demonstrations
        print("\n" + "="*60)
        print("🔍 SEARCH DEMONSTRATIONS")
        print("="*60)

        await demonstrate_basic_search(search_handler)
        await demonstrate_advanced_search(search_handler)
        await demonstrate_database_specific_search(search_handler)

        print("\n✅ CLI demonstrations completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
