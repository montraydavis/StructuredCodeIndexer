"""Debug script for testing AI parsing service functionality."""

import asyncio
from dotenv import load_dotenv

from src.code_search.infrastructure.configuration import get_api_key
from src.code_search.infrastructure.ai.semantic_kernel import SemanticKernelCodeParsingService
from src.code_search.domain.models import CodeMember, MemberType


# Sample code snippets for testing
SAMPLE_PYTHON_CODE = """
class UserService:
    \"\"\"Service for managing user operations.\"\"\"
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def get_user_by_id(self, user_id: int):
        \"\"\"Retrieve a user by their ID.\"\"\"
        return await self.db.fetch_one("SELECT * FROM users WHERE id = ?", user_id)
    
    async def create_user(self, user_data: dict):
        \"\"\"Create a new user.\"\"\"
        return await self.db.execute("INSERT INTO users...", user_data)
    
    def validate_email(self, email: str) -> bool:
        \"\"\"Validate email format.\"\"\"
        import re
        pattern = r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$'
        return bool(re.match(pattern, email))

class DatabaseConnection:
    \"\"\"Database connection wrapper.\"\"\"
    
    async def connect(self):
        \"\"\"Establish database connection.\"\"\"
        pass
    
    async def disconnect(self):
        \"\"\"Close database connection.\"\"\"
        pass
"""

SAMPLE_CSHARP_CODE = """
public interface IUserRepository
{
    Task<User> GetByIdAsync(int userId);
    Task<User> CreateAsync(User user);
    Task UpdateAsync(User user);
    Task DeleteAsync(int userId);
}

public class UserService : IUserService
{
    private readonly IUserRepository _userRepository;
    private readonly ILogger<UserService> _logger;
    
    public UserService(IUserRepository userRepository, ILogger<UserService> logger)
    {
        _userRepository = userRepository;
        _logger = logger;
    }
    
    public async Task<User> GetUserByIdAsync(int userId)
    {
        _logger.LogInformation("Getting user with ID: {UserId}", userId);
        return await _userRepository.GetByIdAsync(userId);
    }
    
    public async Task<User> CreateUserAsync(CreateUserRequest request)
    {
        var user = new User
        {
            Name = request.Name,
            Email = request.Email
        };
        
        return await _userRepository.CreateAsync(user);
    }
    
    public bool ValidateEmail(string email)
    {
        var regex = new Regex(@"^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$");
        return regex.IsMatch(email);
    }
}

public enum UserRole
{
    Admin,
    User,
    Guest
}
"""

SAMPLE_TYPESCRIPT_CODE = """
interface UserRepository {
    getUserById(userId: number): Promise<User>;
    createUser(user: CreateUserRequest): Promise<User>;
    updateUser(user: User): Promise<void>;
    deleteUser(userId: number): Promise<void>;
}

class UserService implements IUserService {
    private userRepository: UserRepository;
    private logger: Logger;
    
    constructor(userRepository: UserRepository, logger: Logger) {
        this.userRepository = userRepository;
        this.logger = logger;
    }
    
    async getUserById(userId: number): Promise<User> {
        this.logger.info(`Getting user with ID: ${userId}`);
        return await this.userRepository.getUserById(userId);
    }
    
    async createUser(request: CreateUserRequest): Promise<User> {
        const user: User = {
            name: request.name,
            email: request.email,
            role: UserRole.User
        };
        
        return await this.userRepository.createUser(user);
    }
    
    validateEmail(email: string): boolean {
        const regex = /^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$/;
        return regex.test(email);
    }
}

enum UserRole {
    Admin = "admin",
    User = "user", 
    Guest = "guest"
}
"""


async def test_ai_parsing_service(ai_service: SemanticKernelCodeParsingService, code: str, file_path: str, language: str):
    """Test AI parsing on a code sample."""
    print(f"\n🤖 Testing AI parsing - {language}")
    print(f"📁 File: {file_path}")
    print("-" * 60)
    
    try:
        # Parse the code
        members = await ai_service.parse_code_to_members(code, file_path)
        
        print(f"✅ Parsing successful - Found {len(members)} members")
        
        # Display results
        for i, member in enumerate(members, 1):
            print(f"\n   {i}. {member.type.value.upper()}: {member.name}")
            print(f"      📝 Summary: {member.summary}")
            print(f"      🔒 Hash: {member.content_hash[:16]}...")
            print(f"      ⚙️  Methods: {len(member.methods)}")
            
            # Show methods
            for j, method in enumerate(member.methods[:3], 1):  # Show max 3 methods
                print(f"         {j}. {method.name}: {method.summary}")
                if j == 3 and len(member.methods) > 3:
                    print(f"         ... and {len(member.methods) - 3} more methods")
        
        return members
        
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return []


async def test_parsing_edge_cases(ai_service: SemanticKernelCodeParsingService):
    """Test AI parsing with edge cases."""
    print(f"\n🧪 Testing edge cases")
    print("-" * 40)
    
    # Test empty file
    print("\n1. Empty file:")
    members = await ai_service.parse_code_to_members("", "empty.py")
    print(f"   Result: {len(members)} members")
    
    # Test comments only
    print("\n2. Comments only:")
    comment_code = """
    # This is just a comment file
    # No actual code here
    # Just testing parsing
    """
    members = await ai_service.parse_code_to_members(comment_code, "comments.py")
    print(f"   Result: {len(members)} members")
    
    # Test malformed code
    print("\n3. Malformed code:")
    malformed_code = """
    class BrokenClass
        def method_without_colon()
            return "this won't parse"
        
    class AnotherClass:
        def valid_method(self):
            return "this should parse"
    """
    members = await ai_service.parse_code_to_members(malformed_code, "malformed.py")
    print(f"   Result: {len(members)} members")


async def test_parsing_performance(ai_service: SemanticKernelCodeParsingService):
    """Test parsing performance with larger code."""
    print(f"\n⏱️ Testing parsing performance")
    print("-" * 40)
    
    # Create a larger code sample
    large_code = SAMPLE_PYTHON_CODE * 5  # Repeat 5 times
    
    import time
    start_time = time.time()
    
    members = await ai_service.parse_code_to_members(large_code, "large_file.py")
    
    end_time = time.time()
    parsing_time = (end_time - start_time) * 1000
    
    print(f"   📏 Code size: {len(large_code)} characters")
    print(f"   ⏱️  Parsing time: {parsing_time:.1f}ms")
    print(f"   📊 Members found: {len(members)}")
    print(f"   🚀 Speed: {len(large_code) / parsing_time * 1000:.0f} chars/second")


async def main():
    """Debug AI parsing functionality."""
    load_dotenv()
    
    try:
        # Get API key
        api_key = get_api_key()
        
        print(f"🔧 DEBUG MODE: AI Parsing Service")
        print(f"🔑 API key configured: {'Yes' if api_key else 'No'}")
        
        # Initialize AI parsing service
        ai_service = SemanticKernelCodeParsingService(api_key, model_name="gpt-4o")
        await ai_service.initialize()
        
        print("\n" + "="*60)
        print("🤖 AI PARSING DEBUG TESTS")
        print("="*60)
        
        # Test different languages
        test_cases = [
            (SAMPLE_PYTHON_CODE, "user_service.py", "Python"),
            (SAMPLE_CSHARP_CODE, "UserService.cs", "C#"),
            (SAMPLE_TYPESCRIPT_CODE, "userService.ts", "TypeScript")
        ]
        
        all_members = []
        for code, file_path, language in test_cases:
            members = await test_ai_parsing_service(ai_service, code, file_path, language)
            all_members.extend(members)
        
        # Test edge cases
        await test_parsing_edge_cases(ai_service)
        
        # Test performance
        await test_parsing_performance(ai_service)
        
        # Summary
        print(f"\n📊 Summary")
        print("-" * 25)
        print(f"   Total members parsed: {len(all_members)}")
        
        # Count by type
        type_counts = {}
        method_count = 0
        for member in all_members:
            type_name = member.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            method_count += len(member.methods)
        
        for type_name, count in type_counts.items():
            print(f"   {type_name.title()}es: {count}")
        print(f"   Total methods: {method_count}")
        
        print("\n✅ AI parsing debug session completed!")
        
    except Exception as e:
        print(f"❌ AI Parsing Debug Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main()) 