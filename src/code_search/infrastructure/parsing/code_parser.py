"""Code parser implementation."""

from typing import List
from pathlib import Path

from ...domain.interfaces import ICodeParser, IAICodeParsingService
from ...domain.models import FileIndex, CodeMember


class CodeParser(ICodeParser):
    """Implementation for parsing code structure using AI service."""
    
    def __init__(self, ai_service: IAICodeParsingService):
        self.ai_service = ai_service

    async def parse_file(self, file_index: FileIndex) -> List[CodeMember]:
        """Parse code content using AI and return structured code members."""
        try:
            print(f"   🤖 AI parsing: {file_index.file_path}")
            members = await self.ai_service.parse_code_to_members(
                file_index.content, 
                file_index.file_path
            )
            
            # Set correct file_id for all members
            for member in members:
                member.file_id = file_index.id
                # Regenerate content hashes with correct file_id
                member.__post_init__()
                for method in member.methods:
                    method.member_id = member.id
                    method.__post_init__()
            
            return members
                    
        except Exception as e:
            print(f"   ❌ AI parsing failed for {file_index.file_path}: {e}")
            return []  # Return empty list on failure

    def supports_file_type(self, file_path: str) -> bool:
        """Check if file type is supported by AI parsing."""
        # AI can handle most common programming languages
        supported_extensions = {'.py', '.cs', '.ts', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb', '.swift', '.kt'}
        return Path(file_path).suffix.lower() in supported_extensions 