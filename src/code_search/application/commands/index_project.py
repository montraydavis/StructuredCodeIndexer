"""Index project command handler."""

from dataclasses import dataclass
from pathlib import Path

from ...domain.interfaces import ICodeIndexer


@dataclass
class IndexProjectCommand:
    """Command to index a project directory."""
    project_directory: str
    

class IndexProjectCommandHandler:
    """Handler for index project commands."""
    
    def __init__(self, indexer: ICodeIndexer):
        self._indexer = indexer
    
    async def handle(self, command: IndexProjectCommand) -> None:
        """Execute the project indexing process."""
        print("🚀 Starting code indexing...")
        print(f"📂 Project directory: {Path(command.project_directory).absolute()}")
        
        await self._indexer.index_project(command.project_directory)
        
        # Get indexing statistics
        stats = await self._indexer.get_project_stats()
        print("\n📊 Indexing Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}") 