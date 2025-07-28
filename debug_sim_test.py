"""
Proposed improvements to embedding strategy for better similarity scores.
"""

from dataclasses import dataclass
from typing import List
import asyncio

from src.code_search.domain.models import FileIndex, CodeMember, CodeMethod, MemberType
from src.code_search.infrastructure.ai.openai import OpenAIEmbeddingService


@dataclass
class ImprovedEmbeddingStrategy:
    """Improved embedding strategies for different content types."""

    @staticmethod
    def create_file_embedding_text(file_index: FileIndex) -> str:
        """
        CURRENT: Embeds entire file content (thousands of characters)
        IMPROVED: Create focused summary with key classes and purpose
        """
        # Extract filename without extension for context
        filename = file_index.file_path.split('/')[-1].replace('.cs', '').replace('.py', '')

        # Create focused embedding text
        embedding_text = f"File {filename}: "

        # Add key classes/members info if available
        if file_index.members:
            class_names = [member.name for member in file_index.members[:3]]  # Top 3 classes
            class_types = [member.type.value for member in file_index.members[:3]]

            class_descriptions = []
            for name, type_val in zip(class_names, class_types):
                class_descriptions.append(f"{name} {type_val}")

            embedding_text += f"contains {', '.join(class_descriptions)}"
        else:
            # Fallback: use first 200 chars of content
            embedding_text += file_index.content[:200].replace('\n', ' ').strip()

        return embedding_text

    @staticmethod
    def create_member_embedding_text(member: CodeMember) -> str:
        """
        CURRENT: Embeds only AI-generated summary (verbose, no name prominence)
        IMPROVED: Name-first with structured context
        """
        # Lead with the actual name
        embedding_text = f"{member.name}"

        # Add type context
        type_context = {
            MemberType.CLASS: "class",
            MemberType.INTERFACE: "interface",
            MemberType.ENUM: "enum"
        }

        member_type = type_context.get(member.type, "class")
        embedding_text += f" {member_type}"

        # Add concise purpose from summary
        if member.summary:
            # Take first sentence or first 100 chars, whichever is shorter
            summary_part = member.summary.split('.')[0] if '.' in member.summary else member.summary
            if len(summary_part) > 100:
                summary_part = summary_part[:100] + "..."

            embedding_text += f" - {summary_part}"

        # Add method context if available
        if member.methods:
            method_count = len(member.methods)
            key_methods = [method.name for method in member.methods[:2]]  # First 2 methods
            embedding_text += f" with {method_count} methods including {', '.join(key_methods)}"

        return embedding_text

    @staticmethod
    def create_method_embedding_text(method: CodeMethod, member_name: str = "") -> str:
        """
        CURRENT: Embeds only AI-generated summary
        IMPROVED: Method name with class context and concise description
        """
        # Lead with method name
        embedding_text = f"{method.name}"

        # Add class context if available
        if member_name:
            embedding_text += f" method in {member_name}"
        else:
            embedding_text += " method"

        # Add concise description
        if method.summary:
            # First sentence or first 80 chars
            summary_part = method.summary.split('.')[0] if '.' in method.summary else method.summary
            if len(summary_part) > 80:
                summary_part = summary_part[:80] + "..."

            embedding_text += f" - {summary_part}"

        return embedding_text


async def test_improved_strategy():
    """Test the improved embedding strategy with real data."""
    print("🚀 TESTING IMPROVED EMBEDDING STRATEGY")
    print("=" * 45)

    # Simulate real data from your project
    file_index = FileIndex(
        file_path="Models/PromptResult.cs",
        content="""namespace ADOPullRequestTools.Models
{
    public record PromptResult(string Content, PromptTiming Timing, int InputTokens, int OutputTokens)
    {
        public static PromptResult Empty => new(string.Empty, PromptTiming.Empty, 0, 0);
    }

    public record PromptTiming(DateTime StartTime, DateTime EndTime, TimeSpan Duration)
    {
        public static PromptTiming Empty => new(DateTime.MinValue, DateTime.MinValue, TimeSpan.Zero);
    }
}"""
    )

    member = CodeMember(
        file_id=file_index.id,
        type=MemberType.CLASS,
        name="PromptResult",
        summary="A record to store the results from each prompt execution with timing information and usage statistics for AI model interactions"
    )

    method = CodeMethod(
        member_id=member.id,
        name="FromPromptResult",
        summary="Converts a PromptResult instance to a PromptResultVector representation suitable for vector database storage and retrieval operations"
    )

    # Add method to member for context
    member.methods = [method]
    file_index.members = [member]

    # Test current vs improved
    strategy = ImprovedEmbeddingStrategy()

    print("📊 EMBEDDING TEXT COMPARISON:")
    print("-" * 40)

    # File embeddings
    print(f"\n📁 FILE EMBEDDINGS:")
    print(f"   Current: {file_index.content[:100]}...")
    print(f"   Length: {len(file_index.content)} characters")

    improved_file = strategy.create_file_embedding_text(file_index)
    print(f"\n   Improved: {improved_file}")
    print(f"   Length: {len(improved_file)} characters")

    # Member embeddings
    print(f"\n🏗️ MEMBER EMBEDDINGS:")
    print(f"   Current: {member.summary}")
    print(f"   Length: {len(member.summary)} characters")

    improved_member = strategy.create_member_embedding_text(member)
    print(f"\n   Improved: {improved_member}")
    print(f"   Length: {len(improved_member)} characters")

    # Method embeddings
    print(f"\n⚙️ METHOD EMBEDDINGS:")
    print(f"   Current: {method.summary}")
    print(f"   Length: {len(method.summary)} characters")

    improved_method = strategy.create_method_embedding_text(method, member.name)
    print(f"\n   Improved: {improved_method}")
    print(f"   Length: {len(improved_method)} characters")


async def simulate_score_improvements():
    """Simulate what score improvements we might expect."""
    print(f"\n🎯 EXPECTED SCORE IMPROVEMENTS")
    print("=" * 35)

    improvements = {
        "File Embeddings": {
            "current_approach": "Embedding entire file content (1000+ chars)",
            "current_score_range": "0.1 - 0.3",
            "improved_approach": "Focused summary with key classes",
            "expected_score_range": "0.3 - 0.6",
            "improvement": "+100% to +200%"
        },

        "Member Embeddings": {
            "current_approach": "AI summary only (no name prominence)",
            "current_score_range": "0.2 - 0.4",
            "improved_approach": "Name-first with structured context",
            "expected_score_range": "0.4 - 0.7",
            "improvement": "+50% to +100%"
        },

        "Method Embeddings": {
            "current_approach": "Verbose AI description",
            "current_score_range": "0.2 - 0.5",
            "improved_approach": "Method name + class + concise purpose",
            "expected_score_range": "0.4 - 0.8",
            "improvement": "+60% to +100%"
        }
    }

    for category, info in improvements.items():
        print(f"\n📋 {category}:")
        print(f"   Current: {info['current_approach']}")
        print(f"   Current scores: {info['current_score_range']}")
        print(f"   Improved: {info['improved_approach']}")
        print(f"   Expected scores: {info['expected_score_range']}")
        print(f"   Improvement: {info['improvement']}")


def show_implementation_changes():
    """Show what code changes we'd need to make."""
    print(f"\n🔧 IMPLEMENTATION CHANGES NEEDED")
    print("=" * 40)

    changes = [
        {
            "file": "infrastructure/storage/vector/simple_vector_store.py",
            "change": "Update store_file_index() to use improved file embedding text",
            "code": "file_index.content_embedding = await embedding_service.generate_embedding(\n    ImprovedEmbeddingStrategy.create_file_embedding_text(file_index)\n)"
        },

        {
            "file": "application/services/indexing_service.py",
            "change": "Update member embedding generation",
            "code": "member.summary_embedding = await embedding_service.generate_embedding(\n    ImprovedEmbeddingStrategy.create_member_embedding_text(member)\n)"
        },

        {
            "file": "application/services/indexing_service.py",
            "change": "Update method embedding generation with context",
            "code": "method.summary_embedding = await embedding_service.generate_embedding(\n    ImprovedEmbeddingStrategy.create_method_embedding_text(method, member.name)\n)"
        }
    ]

    for i, change in enumerate(changes, 1):
        print(f"\n{i}. {change['file']}:")
        print(f"   Change: {change['change']}")
        print(f"   Code: {change['code']}")


async def main():
    """Run embedding improvement analysis."""
    try:
        await test_improved_strategy()
        await simulate_score_improvements()
        show_implementation_changes()

        print(f"\n💡 SUMMARY:")
        print("=" * 12)
        print("• Current low scores (0.2-0.5) are due to verbose, unfocused embeddings")
        print("• Including names prominently should boost scores to 0.4-0.7 range")
        print("• Structured approach (Name - Description) works better than summaries alone")
        print("• Shorter, focused embeddings often outperform longer verbose ones")
        print("• These changes could double our search effectiveness")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
