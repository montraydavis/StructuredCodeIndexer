"""AI code parsing service implementation using Semantic Kernel."""

from typing import List, Optional, Dict
from pathlib import Path
import uuid
import json

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig

from ....domain.interfaces import IAICodeParsingService
from ....domain.models import CodeMember, CodeMethod, MemberType


class SemanticKernelCodeParsingService(IAICodeParsingService):
    """AI code parsing service using Semantic Kernel."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.model_name = model_name
        self.kernel: Optional[Kernel] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the AI service."""
        if self._initialized:
            return

        try:
            self.kernel = Kernel()

            # Configure chat completion service
            chat_service = OpenAIChatCompletion(
                service_id="code_parser",
                ai_model_id=self.model_name,
                api_key=self.api_key
            )

            self.kernel.add_service(chat_service)
            self._initialized = True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize AI service: {e}")

    def _get_parsing_prompt(self) -> str:
        """Get the prompt template for code parsing."""
        return '''
## Role
You are an expert code analyzer that extracts structured information from source code.

## Instructions
Analyze the provided source code and extract all classes, interfaces, and enums with their methods.
Return the results as a valid JSON object with the exact structure specified below.

## Required JSON Structure
```json
{
  "members": [
    {
      "type": "class|interface|enum",
      "name": "MemberName",
      "summary": "Clear description of what this member does or represents",
      "methods": [
        {
          "name": "methodName",
          "summary": "Clear description of what this method does"
        }
      ]
    }
  ]
}
```

## Rules
1. **type**: Must be exactly "class", "interface", or "enum"
2. **name**: Extract the actual name from the code
3. **summary**: Write a concise, clear description (1-2 sentences)
4. **methods**: Include all public methods, functions, or procedures
5. For enums: methods array should be empty []
6. Skip private/internal methods unless they're the only methods
7. Return only valid JSON - no markdown, no explanations

## File Information
- File: {{$file_path}}
- Language: {{$language}}

## Source Code
```
{{$source_code}}
```

Return only the JSON response:'''

    async def parse_code_to_members(self, file_content: str, file_path: str) -> List[CodeMember]:
        """Parse code content using AI and return structured code members."""
        if not self._initialized:
            await self.initialize()

        if not self.kernel:
            raise RuntimeError("Kernel not initialized")

        try:
            # Determine language from file extension
            language = self._get_language_from_path(file_path)

            # Prepare the prompt
            prompt = self._get_parsing_prompt()

            # Create arguments
            arguments = KernelArguments(
                file_path=file_path,
                language=language,
                source_code=file_content
            )

            # Create prompt template config
            template_config = PromptTemplateConfig(
                template=prompt,
                template_format="semantic-kernel",
                name="CodeParsingTemplate",
                description="Extracts structured code information"
            )

                                    # Create and invoke function
            function = self.kernel.add_function(
                plugin_name="CodeParser",
                function_name="parse_code",
                prompt_template_config=template_config
            )

            # Get AI response
            try:
                response = await self.kernel.invoke(function, arguments)  # type: ignore
            except Exception as e:
                # Try alternative invocation if first method fails
                if hasattr(function, 'invoke_async'):
                    response = await function.invoke_async(self.kernel, arguments)  # type: ignore
                else:
                    raise e

            if not response or not response.value:
                raise ValueError("Empty response from AI service")

            # Parse JSON response
            json_result = self._parse_json_response(response.value[0].content)

            # Convert to CodeMember objects
            return self._convert_to_code_members(json_result, file_path)

        except Exception as e:
            print(f"AI parsing failed for {file_path}: {e}")
            return []  # Fall back to empty list on error

    def _get_language_from_path(self, file_path: str) -> str:
        """Determine programming language from file extension."""
        extension = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'Python',
            '.cs': 'C#',
            '.ts': 'TypeScript',
            '.js': 'JavaScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C'
        }
        return language_map.get(extension, 'Unknown')

    def _parse_json_response(self, response: str) -> Dict:
        """Parse and validate JSON response from AI."""
        try:
            # Clean response - remove any markdown formatting
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]

            # Parse JSON
            result = json.loads(cleaned.strip())

            # Validate structure
            if not isinstance(result, dict) or 'members' not in result:
                raise ValueError("Invalid JSON structure - missing 'members' key")

            if not isinstance(result['members'], list):
                raise ValueError("Invalid JSON structure - 'members' must be a list")

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    def _convert_to_code_members(self, json_result: Dict, file_path: str) -> List[CodeMember]:
        """Convert JSON result to CodeMember objects."""
        members = []
        file_id = str(uuid.uuid4())  # Generate file ID for this parsing session

        for member_data in json_result.get('members', []):
            try:
                # Validate required fields
                if not all(key in member_data for key in ['type', 'name', 'summary']):
                    continue

                # Convert type string to enum
                member_type_str = member_data['type'].lower()
                if member_type_str == 'class':
                    member_type = MemberType.CLASS
                elif member_type_str == 'interface':
                    member_type = MemberType.INTERFACE
                elif member_type_str == 'enum':
                    member_type = MemberType.ENUM
                else:
                    continue  # Skip unknown types

                # Create methods
                methods = []
                for method_data in member_data.get('methods', []):
                    if 'name' in method_data and 'summary' in method_data:
                        method = CodeMethod(
                            member_id="",  # Will be set after member creation
                            name=method_data['name'],
                            summary=method_data['summary']
                        )
                        methods.append(method)

                # Create member
                member = CodeMember(
                    file_id=file_id,
                    type=member_type,
                    name=member_data['name'],
                    summary=member_data['summary'],
                    methods=methods
                )

                # Set member_id for methods
                for method in member.methods:
                    method.member_id = member.id

                members.append(member)

            except Exception as e:
                print(f"Failed to convert member data: {e}")
                continue

        return members
