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
    # Optional: only required when a Gemini-backed provider is actually selected.
    GEMINI_API_KEY: Optional[str] = Field(None, env="GEMINI_API_KEY")
    # Model identifiers verified against the Gemini model list on 2026-08-01.
    LLM_MODEL_NAME: str = "models/gemini-3.5-flash-lite" # Default LLM model for text generation by agents
    EMBEDDING_MODEL_NAME: str = "models/gemini-embedding-001" # Default LLM model for embedding generation
    LLM_MODEL_FOR_RESPONSE_GENERATION: str = "models/gemini-3.5-flash-lite" # Specific model for Cognitive Brain's final response
    LLM_MODEL_FOR_MODERATION: str = "models/gemini-3.5-flash-lite" # Specific model for content moderation
    LLM_PROVIDER: str = Field("gemini", env="LLM_PROVIDER")
    OLLAMA_BASE_URL: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_CHAT_MODEL: str = Field("", env="OLLAMA_CHAT_MODEL")
    EMBEDDING_PROVIDER: str = Field("gemini", env="EMBEDDING_PROVIDER")
    OLLAMA_EMBEDDING_MODEL: str = Field("", env="OLLAMA_EMBEDDING_MODEL")
    OLLAMA_MAX_INTERACTIVE_REQUESTS: int = Field(1, env="OLLAMA_MAX_INTERACTIVE_REQUESTS")
    OLLAMA_MAX_BACKGROUND_REQUESTS: int = Field(1, env="OLLAMA_MAX_BACKGROUND_REQUESTS")
    # Reasoning models consume the whole output budget on thinking tokens and return an empty response.
    OLLAMA_THINKING: bool = Field(False, env="OLLAMA_THINKING")
    # Ollama silently truncates to a small default context, which makes long prompts return nothing.
    OLLAMA_NUM_CTX: int = Field(16384, env="OLLAMA_NUM_CTX")
    MODERATION_PROVIDER: str = Field("gemini", env="MODERATION_PROVIDER")
    # Empty means the final response is synthesised by the same provider as the agents.
    SYNTHESIS_PROVIDER: str = Field("", env="SYNTHESIS_PROVIDER")
    # Refuses to start if any role resolves to a provider that leaves the machine.
    LOCAL_ONLY_MODE: bool = Field(False, env="LOCAL_ONLY_MODE")
    # Fails startup when a collection's stored embedding identity differs from the active provider.
    EMBEDDING_IDENTITY_ENFORCED: bool = Field(True, env="EMBEDDING_IDENTITY_ENFORCED")

    # LLM Rate limiting & concurrency
    LLM_MAX_CONCURRENCY_PER_MODEL: int = Field(2, env="LLM_MAX_CONCURRENCY_PER_MODEL")
    LLM_GLOBAL_MAX_CONCURRENCY: int = Field(6, env="LLM_GLOBAL_MAX_CONCURRENCY")
    LLM_429_BASE_DELAY_SEC: float = Field(10.0, env="LLM_429_BASE_DELAY_SEC")
    LLM_429_JITTER_SEC: float = Field(0.5, env="LLM_429_JITTER_SEC")
    META_COGNITIVE_MAX_OUTPUT_TOKENS: int = Field(64, env="META_COGNITIVE_MAX_OUTPUT_TOKENS")
    META_COGNITIVE_MAX_RESPONSE_WORDS: int = Field(40, env="META_COGNITIVE_MAX_RESPONSE_WORDS")

    # External research is a separate, disabled-by-default capability.
    RESEARCH_ENABLED: bool = Field(False, env="RESEARCH_ENABLED")
    RESEARCH_PROVIDER: str = Field("disabled", env="RESEARCH_PROVIDER")
    # A model is preselected so a configured Gemini key can be enabled from the
    # operator UI; provider access itself remains disabled until explicitly toggled.
    RESEARCH_MODEL: str = Field("models/gemini-3.5-flash-lite", env="RESEARCH_MODEL")
    RESEARCH_LOW_CONFIDENCE_THRESHOLD: float = Field(0.55, env="RESEARCH_LOW_CONFIDENCE_THRESHOLD")
    RESEARCH_MAX_QUERIES: int = Field(3, env="RESEARCH_MAX_QUERIES")
    RESEARCH_MAX_QUERY_CHARS: int = Field(500, env="RESEARCH_MAX_QUERY_CHARS")
    RESEARCH_TIMEOUT_SECONDS: float = Field(30.0, env="RESEARCH_TIMEOUT_SECONDS")
    COGNITIVE_RESEARCH_DRIVE_ENABLED: bool = Field(False, env="COGNITIVE_RESEARCH_DRIVE_ENABLED")
    COGNITIVE_RESEARCH_DRIVE_SHADOW_MODE: bool = Field(True, env="COGNITIVE_RESEARCH_DRIVE_SHADOW_MODE")
    COGNITIVE_RESEARCH_DEEPEN_THRESHOLD: float = Field(0.28, env="COGNITIVE_RESEARCH_DEEPEN_THRESHOLD")
    COGNITIVE_RESEARCH_UNCERTAINTY_THRESHOLD: float = Field(0.48, env="COGNITIVE_RESEARCH_UNCERTAINTY_THRESHOLD")
    COGNITIVE_RESEARCH_INQUIRY_THRESHOLD: float = Field(0.64, env="COGNITIVE_RESEARCH_INQUIRY_THRESHOLD")
    COGNITIVE_RESEARCH_EXTERNAL_THRESHOLD: float = Field(0.78, env="COGNITIVE_RESEARCH_EXTERNAL_THRESHOLD")
    COGNITIVE_RESEARCH_COOLDOWN_MINUTES: float = Field(30.0, env="COGNITIVE_RESEARCH_COOLDOWN_MINUTES")
    COGNITIVE_RESEARCH_HYSTERESIS_MINUTES: float = Field(15.0, env="COGNITIVE_RESEARCH_HYSTERESIS_MINUTES")
    INQUIRY_QUEUE_ENABLED: bool = Field(True, env="INQUIRY_QUEUE_ENABLED")
    INQUIRY_DB_PATH: str = Field("./chroma_db/inquiry_queue.sqlite3", env="INQUIRY_DB_PATH")
    INQUIRY_TTL_DAYS: int = Field(14, env="INQUIRY_TTL_DAYS")
    INQUIRY_REQUIRE_USER_APPROVAL: bool = Field(True, env="INQUIRY_REQUIRE_USER_APPROVAL")

    # Legacy search credentials are retained for migration only. ResearchService does not read them.
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
    SALIENCE_NETWORK_SHADOW_MODE: bool = Field(True, env="SALIENCE_NETWORK_SHADOW_MODE")
    SALIENCE_NETWORK_TOP_K: int = Field(3, ge=1, le=20, env="SALIENCE_NETWORK_TOP_K")
    SALIENCE_RECENCY_HALF_LIFE_DAYS: float = Field(
        30.0,
        gt=0.0,
        env="SALIENCE_RECENCY_HALF_LIFE_DAYS",
    )

@lru_cache
def get_settings():
    logger.info("Loading application settings...")
    return Settings()

settings = get_settings()
