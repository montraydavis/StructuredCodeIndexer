"""File loader implementation."""

from typing import List, Optional
from pathlib import Path

from ....domain.interfaces import IFileLoader
from ....domain.models import FileIndex


class FileLoader(IFileLoader):
    """Implementation for loading source code files from filesystem."""

    def __init__(self, supported_extensions: Optional[List[str]] = None):
        self._supported_extensions = supported_extensions or ['.py', '.cs', '.ts', '.js']

    async def load_files(self, project_directory: str) -> List[FileIndex]:
        """Load all source code files from the specified directory."""
        files = []
        project_path = Path(project_directory)

        for file_path in project_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self._supported_extensions:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    file_index = FileIndex(
                        file_path=str(file_path.relative_to(project_path)),
                        content=content
                    )
                    # Content hash is automatically generated in __post_init__
                    files.append(file_index)
                except (UnicodeDecodeError, PermissionError):
                    continue

        return files

    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return self._supported_extensions
