"""Application configuration settings."""

import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class AppSettings:
    """Application configuration settings."""

    # Vector store settings
    vector_store_path: str = field(default="./vectordb_workspace")

    # AI model settings
    embedding_model: str = field(default="text-embedding-3-large")
    ai_model: str = field(default="gpt-4o")

    # Search settings
    similarity_threshold: float = field(default=0.3)
    max_search_results: int = field(default=10)

    # File processing settings
    supported_extensions: list = field(default_factory=lambda: ['.py', '.cs', '.ts', '.js', '.java', '.cpp'])

    # Performance settings
    batch_size: int = field(default=100)
    cache_size: int = field(default=1000)

    @classmethod
    def load_from_json(cls, config_path: Optional[str] = None) -> 'AppSettings':
        """Load settings from JSON file."""
        if config_path is None:
            config_path = os.getenv('CONFIG_PATH', './config/app_config.json')

        config_file = Path(config_path)

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return cls(**config_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                print("Using default settings...")

        return cls()

    @classmethod
    def load_from_env(cls) -> 'AppSettings':
        """Load settings from environment variables."""
        return cls(
            vector_store_path=os.getenv('VECTOR_STORE_PATH', './vectordb_workspace'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large'),
            ai_model=os.getenv('AI_MODEL', 'gpt-4o'),
            similarity_threshold=float(os.getenv('SIMILARITY_THRESHOLD', '0.3')),
            max_search_results=int(os.getenv('MAX_SEARCH_RESULTS', '10')),
            supported_extensions=os.getenv('SUPPORTED_EXTENSIONS', '.py,.cs,.ts,.js,.java,.cpp').split(','),
            batch_size=int(os.getenv('BATCH_SIZE', '100')),
            cache_size=int(os.getenv('CACHE_SIZE', '1000'))
        )

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'AppSettings':
        """Load settings with fallback priority: JSON -> Environment -> Defaults."""
        # Try JSON first
        settings = cls.load_from_json(config_path)

        # Override with environment variables (environment takes precedence)
        env_settings = cls.load_from_env()

        # Merge settings (environment overrides JSON)
        for field_name in settings.__dataclass_fields__:
            env_value = getattr(env_settings, field_name)
            if env_value is not None:
                setattr(settings, field_name, env_value)

        return settings

    def save_to_json(self, config_path: Optional[str] = None) -> None:
        """Save current settings to JSON file."""
        if config_path is None:
            config_path = os.getenv('CONFIG_PATH', './config/app_config.json')

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
            print(f"✅ Configuration saved to {config_path}")
        except Exception as e:
            print(f"❌ Could not save configuration to {config_path}: {e}")


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings.load()
    return _settings


def reload_settings(config_path: Optional[str] = None) -> AppSettings:
    """Reload settings from configuration source."""
    global _settings
    _settings = AppSettings.load(config_path)
    return _settings
