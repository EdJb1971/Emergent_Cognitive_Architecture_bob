import logging
import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Multi-Agent Cognitive Architecture API"
    ENVIRONMENT: str = "development"
    DEBUG_MODE: bool = False

    # LLM Integration Service Settings
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    LLM_MODEL_NAME: str = "models/gemini-2.5-flash-lite" # Default LLM model for text generation by agents
    EMBEDDING_MODEL_NAME: str = "models/embedding-001" # Default LLM model for embedding generation
    LLM_MODEL_FOR_RESPONSE_GENERATION: str = "models/gemini-2.5-flash-lite" # Specific model for Cognitive Brain's final response
    LLM_MODEL_FOR_MODERATION: str = "models/gemini-2.0-flash-lite" # Specific model for content moderation

    # LLM Rate limiting & concurrency
    LLM_MAX_CONCURRENCY_PER_MODEL: int = Field(2, env="LLM_MAX_CONCURRENCY_PER_MODEL")
    LLM_GLOBAL_MAX_CONCURRENCY: int = Field(6, env="LLM_GLOBAL_MAX_CONCURRENCY")
    LLM_429_BASE_DELAY_SEC: float = Field(10.0, env="LLM_429_BASE_DELAY_SEC")
    LLM_429_JITTER_SEC: float = Field(0.5, env="LLM_429_JITTER_SEC")

    # Web Browsing Service Settings (for WebBrowsingService)
    SERPAPI_API_KEY: Optional[str] = Field(None, env="SERPAPI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    GOOGLE_CSE_ID: Optional[str] = Field(None, env="GOOGLE_CSE_ID")

    # Local Vector Database Settings (for MemoryService)
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_CYCLES: str = "cognitive_cycles"
    CHROMA_COLLECTION_PATTERNS: str = "discovered_patterns"

    # Security Settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_KEY_HEADER_NAME: str = "X-API-Key"
    # SEC-003, CQ-002 Fix: Replaced API_KEYS_CSV with a single API_KEY for system authentication.
    # This is a step towards a more robust system by removing the insecure list management,
    # and will be paired with a fixed user ID in dependencies.py.
    API_KEY: str = Field(..., env="API_KEY", description="A single, strong API key for system authentication.")


    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "./logs/app.log"

    # Phase 7 Feature Flags
    ATTENTION_CONTROLLER_ENABLED: bool = Field(False, env="ATTENTION_CONTROLLER_ENABLED")
    ATTENTION_CONTROLLER_SHADOW_MODE: bool = Field(True, env="ATTENTION_CONTROLLER_SHADOW_MODE")
    SALIENCE_NETWORK_ENABLED: bool = Field(False, env="SALIENCE_NETWORK_ENABLED")

@lru_cache
def get_settings():
    logger.info("Loading application settings...")
    return Settings()

settings = get_settings()
