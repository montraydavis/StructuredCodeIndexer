"""Code indexing service implementation."""

from typing import Dict

from ...domain.interfaces import ICodeIndexer, IFileLoader, ICodeParser, IEmbeddingService, IVectorStore


class CodeIndexer(ICodeIndexer):
    """Implementation of code indexing operations."""

    def __init__(self, file_loader: IFileLoader, code_parser: ICodeParser,
                 embedding_service: IEmbeddingService, vector_store: IVectorStore,
                 resume_mode: bool = True):
        self._file_loader = file_loader
        self._code_parser = code_parser
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._resume_mode = resume_mode
        self._stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'members_extracted': 0,
            'members_skipped': 0,
            'methods_extracted': 0,
            'methods_skipped': 0,
            'embeddings_generated': 0,
            'embeddings_cached': 0
        }

    async def index_project(self, project_directory: str) -> None:
        """Execute the complete indexing pipeline for a project with duplicate detection."""
        print(f"🚀 Starting indexing for project: {project_directory}")
        print(f"📋 Resume mode: {'Enabled' if self._resume_mode else 'Disabled'}")

        # Load existing state if in resume mode
        if self._resume_mode:
            print("🔄 Loading existing project state...")
            existing_hashes = await self._vector_store.get_all_file_hashes()
            print(f"   Found {len(existing_hashes)} existing files in index")

        # Step 1: Load files
        print("📁 Loading source files...")
        file_indices = await self._file_loader.load_files(project_directory)
        print(f"   Found {len(file_indices)} files in project")

        # Step 2: Process files with duplicate detection
        print("🔢 Processing files and generating embeddings...")
        processed_files = []

        for file_index in file_indices:
            # Check if file already exists with same content
            existing_file = await self._vector_store.file_exists(file_index.file_path, file_index.content_hash)

            if existing_file and self._resume_mode:
                print(f"   ⏭️  Skipping unchanged file: {file_index.file_path}")
                self._stats['files_skipped'] += 1
                # Use existing file data including cached embedding
                processed_files.append(existing_file)
                continue

            # Generate optimized embedding for new/changed file
            print(f"   🔢 Processing file: {file_index.file_path}")

            # Create focused file embedding text
            filename = file_index.file_path.split('/')[-1].replace('.cs', '').replace('.py', '').replace('.ts', '').replace('.js', '')
            file_embedding_text = f"File {filename}"

            # Add key classes info if we have parsed members
            if hasattr(file_index, 'members') and file_index.members:
                class_names = [member.name for member in file_index.members[:3]]  # Top 3 classes
                class_types = [member.type.value for member in file_index.members[:3]]
                class_descriptions = [f"{name} {type_val}" for name, type_val in zip(class_names, class_types)]
                file_embedding_text += f": contains {', '.join(class_descriptions)}"
            else:
                # Fallback: use first 200 chars of content for context
                content_preview = file_index.content[:200].replace('\n', ' ').strip()
                file_embedding_text += f": {content_preview}"

            file_index.content_embedding = await self._embedding_service.generate_embedding(file_embedding_text)
            await self._vector_store.store_file_index(file_index)
            processed_files.append(file_index)
            self._stats['files_processed'] += 1
            self._stats['embeddings_generated'] += 1

        print(f"   📊 Files - Processed: {self._stats['files_processed']}, Skipped: {self._stats['files_skipped']}")

        # Step 3: Parse code structure with duplicate detection
        print("🔍 Parsing code structure...")
        all_members = []

        for file_index in processed_files:
            if self._code_parser.supports_file_type(file_index.file_path):
                members = await self._code_parser.parse_file(file_index)

                for member in members:
                    # Check if member already exists
                    existing_member = await self._vector_store.member_exists(member.content_hash)

                    if existing_member and self._resume_mode:
                        print(f"   ⏭️  Skipping unchanged member: {member.name}")
                        self._stats['members_skipped'] += 1
                        all_members.append(existing_member)
                        continue

                    # Generate optimized embedding for new/changed member
                    print(f"   🔍 Processing member: {member.name} ({member.type.value})")

                    # Create name-first structured embedding text
                    member_embedding_text = f"{member.name} {member.type.value}"

                    # Add concise description from summary
                    if member.summary:
                        # Take first sentence or first 100 chars, whichever is shorter
                        summary_part = member.summary.split('.')[0] if '.' in member.summary else member.summary
                        if len(summary_part) > 100:
                            summary_part = summary_part[:100] + "..."
                        member_embedding_text += f" - {summary_part}"

                    # Add method context if available
                    if member.methods:
                        method_count = len(member.methods)
                        key_methods = [method.name for method in member.methods[:2]]  # First 2 methods
                        member_embedding_text += f" with {method_count} methods including {', '.join(key_methods)}"

                    member.summary_embedding = await self._embedding_service.generate_embedding(member_embedding_text)
                    await self._vector_store.store_code_member(member)
                    all_members.append(member)
                    self._stats['members_extracted'] += 1
                    self._stats['embeddings_generated'] += 1

        print(f"   📊 Members - Extracted: {self._stats['members_extracted']}, Skipped: {self._stats['members_skipped']}")

        # Step 4: Process methods with duplicate detection
        print("⚙️ Processing methods...")
        all_methods = []

        for member in all_members:
            for method in member.methods:
                # Check if method already exists
                existing_method = await self._vector_store.method_exists(method.content_hash)

                if existing_method and self._resume_mode:
                    print(f"   ⏭️  Skipping unchanged method: {method.name}")
                    self._stats['methods_skipped'] += 1
                    all_methods.append(existing_method)
                    continue

                # Generate optimized embedding for new/changed method
                print(f"   ⚙️ Processing method: {method.name}")

                # Create method embedding with name, context, and purpose
                method_embedding_text = f"{method.name} method"

                # Add class context
                parent_member = None
                for member in all_members:
                    if method.member_id == member.id:
                        parent_member = member
                        break

                if parent_member:
                    method_embedding_text += f" in {parent_member.name}"

                # Add concise description
                if method.summary:
                    # First sentence or first 80 chars
                    summary_part = method.summary.split('.')[0] if '.' in method.summary else method.summary
                    if len(summary_part) > 80:
                        summary_part = summary_part[:80] + "..."
                    method_embedding_text += f" - {summary_part}"

                method.summary_embedding = await self._embedding_service.generate_embedding(method_embedding_text)
                await self._vector_store.store_code_method(method)
                all_methods.append(method)
                self._stats['methods_extracted'] += 1
                self._stats['embeddings_generated'] += 1

        print(f"   📊 Methods - Extracted: {self._stats['methods_extracted']}, Skipped: {self._stats['methods_skipped']}")

        print("✅ Indexing completed successfully!")
        print(f"📊 Final Stats: {self._stats}")

    async def get_project_stats(self) -> Dict:
        """Return statistics about the indexed project."""
        return self._stats.copy()
