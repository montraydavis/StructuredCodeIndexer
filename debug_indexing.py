"""Debug script for testing indexing functionality only."""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.code_search.infrastructure.configuration import initialize_services, get_api_key
from src.code_search.application.commands import IndexProjectCommand, IndexProjectCommandHandler


async def main():
    """Debug indexing functionality."""
    load_dotenv()

    try:
        # Get API key and initialize services
        api_key = get_api_key()

        # Configure project directory - adjust this path as needed
        project_directory = "C:\\Users\\montr\\Downloads\\ADOPullRequestTools\\ADOPullRequestTools\\Models"

        print(f"🔧 DEBUG MODE: Indexing Only")
        print(f"📂 Project directory: {Path(project_directory).absolute()}")
        print(f"🔑 API key configured: {'Yes' if api_key else 'No'}")

        # Initialize services
        indexer, _ = await initialize_services(api_key)

        # Create command handler
        index_handler = IndexProjectCommandHandler(indexer)

        # Run indexing with detailed logging
        print("\n🚀 Starting indexing process...")
        print("=" * 50)

        # Manually create command to test different scenarios
        index_command = IndexProjectCommand(project_directory=project_directory)

        # Set breakpoint here for debugging
        await index_handler.handle(index_command)

        # Get final statistics
        stats = await indexer.get_project_stats()

        print("\n📊 Final Indexing Statistics:")
        print("=" * 40)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        # Test individual components
        print("\n🔍 Component Tests:")
        print("-" * 25)

        # Test file loading
        from src.code_search.infrastructure.storage.file import FileLoader
        file_loader = FileLoader()
        files = await file_loader.load_files(project_directory)
        print(f"✅ File loading: {len(files)} files found")

        if files:
            print(f"   📄 Sample file: {files[0].file_path}")
            print(f"   📏 Content length: {len(files[0].content)} chars")
            print(f"   🔒 Hash: {files[0].content_hash[:16]}...")

        print("\n✅ Indexing debug session completed!")

    except Exception as e:
        print(f"❌ Indexing Debug Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
