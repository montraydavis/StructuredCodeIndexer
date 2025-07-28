"""Debug script for testing the full indexing and search pipeline."""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.code_search.infrastructure.configuration import initialize_services, get_api_key
from src.code_search.application.commands import (
    IndexProjectCommand, IndexProjectCommandHandler,
    SearchCodeCommand, SearchCodeCommandHandler
)


async def main():
    """Main debug function with indexing enabled."""
    load_dotenv()

    try:
        # Get API key and initialize services
        api_key = get_api_key()

        # Configure project directory - adjust this path as needed
        project_directory = "C:\\Users\\montr\\Downloads\\ADOPullRequestTools\\ADOPullRequestTools\\Models"

        print(f"🔧 DEBUG MODE: Full pipeline with indexing")
        print(f"📂 Project directory: {Path(project_directory).absolute()}")
        print(f"🔑 API key configured: {'Yes' if api_key else 'No'}")

        # Initialize all services
        indexer, search_service = await initialize_services(api_key)

        # Create command handlers
        index_handler = IndexProjectCommandHandler(indexer)
        search_handler = SearchCodeCommandHandler(search_service)

        # Run indexing (ENABLED for debugging)
        print("\n🚀 Starting indexing process...")
        index_command = IndexProjectCommand(project_directory=project_directory)
        await index_handler.handle(index_command)

        # Run search demonstrations
        print("\n" + "="*60)
        print("🔍 SEARCH DEMONSTRATIONS")
        print("="*60)

        # Basic search test
        print("\n🔤 Basic Search Test")
        print("-" * 25)

        search_command = SearchCodeCommand(
            query="trade",
            max_results_per_type=3,
            similarity_threshold=0.6
        )

        response = await search_handler.handle_text_search(search_command)

        print(f"📊 Found {response.total_results} results in {response.execution_time_ms:.1f}ms")

        # Show results
        for result in response.all_results[:5]:
            print(f"📄 {result.type.upper()}: {result.name}")
            print(f"   📁 {result.file_path}")
            print(f"   ⭐ Score: {result.score:.3f}")
            print(f"   📝 {result.summary[:100]}...")
            print()

        print("✅ Debug session completed successfully!")

    except Exception as e:
        print(f"❌ Debug Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
