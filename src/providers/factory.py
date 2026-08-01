"""Resolves generation, embedding, and moderation providers from configuration."""

from __future__ import annotations

from typing import Any, Optional, Union

from src.core.config import settings
from src.core.exceptions import ConfigurationError
from src.providers.base import EmbeddingModelIdentity
from src.providers.composite_provider import CompositeProvider
from src.providers.execution_scheduler import ModelExecutionScheduler
from src.providers.gemini_provider import GeminiProvider
from src.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from src.providers.ollama_provider import OllamaProvider

ActiveProvider = Union[GeminiProvider, CompositeProvider]


def _require_ollama_model(name: str, setting: str) -> str:
    if not name:
        raise ConfigurationError(detail=f"{setting} must be set when Ollama is selected.")
    return name


def build_embedding_provider(scheduler: ModelExecutionScheduler, gemini: Optional[GeminiProvider] = None):
    """Build only the embedding role; used by tools that never generate text."""
    choice = settings.EMBEDDING_PROVIDER.lower()
    if choice == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=_require_ollama_model(settings.OLLAMA_EMBEDDING_MODEL, "OLLAMA_EMBEDDING_MODEL"),
            scheduler=scheduler,
        )
    if choice == "gemini":
        return gemini or GeminiProvider()
    raise ConfigurationError(detail=f"Unsupported EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'.")


def build_embedding_provider_for_identity(
    identity: EmbeddingModelIdentity, scheduler: ModelExecutionScheduler
):
    """Reconstruct the provider that produced a stored vector space, for like-for-like queries."""
    if identity.provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=identity.model,
            scheduler=scheduler,
        )
    if identity.provider == "gemini":
        return GeminiProvider()
    raise ConfigurationError(detail=f"No adapter available for embedding provider '{identity.provider}'.")


def build_synthesis_provider(scheduler: ModelExecutionScheduler, generation_provider: Any) -> Any:
    """The final response may be synthesised by a different provider than the agents.

    An empty SYNTHESIS_PROVIDER keeps synthesis on the agent provider, so the local-only
    promise holds by default; setting it to 'gemini' is an explicit, per-turn cloud call.
    """
    choice = (settings.SYNTHESIS_PROVIDER or "").lower()
    if not choice:
        return generation_provider
    if choice == "ollama":
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=_require_ollama_model(settings.OLLAMA_CHAT_MODEL, "OLLAMA_CHAT_MODEL"),
            scheduler=scheduler,
        )
    if choice == "gemini":
        return GeminiProvider(model=settings.LLM_MODEL_FOR_RESPONSE_GENERATION)
    raise ConfigurationError(detail=f"Unsupported SYNTHESIS_PROVIDER '{settings.SYNTHESIS_PROVIDER}'.")


def enforce_local_only(**roles: Any) -> None:
    """Fails startup rather than letting a cloud provider slip into a local-only deployment."""
    if not settings.LOCAL_ONLY_MODE:
        return
    remote = sorted(
        f"{role}={provider.capabilities.provider}"
        for role, provider in roles.items()
        if provider is not None and not provider.capabilities.is_local
    )
    if remote:
        raise ConfigurationError(
            detail=f"LOCAL_ONLY_MODE is enabled but these roles are not local: {', '.join(remote)}."
        )


def build_active_provider(scheduler: ModelExecutionScheduler) -> ActiveProvider:
    """Each role is selected independently; a uniform selection skips the composite wrapper."""
    cached_gemini: Optional[GeminiProvider] = None

    def gemini() -> GeminiProvider:
        nonlocal cached_gemini
        if cached_gemini is None:
            cached_gemini = GeminiProvider()
        return cached_gemini

    generation_choice = settings.LLM_PROVIDER.lower()
    if generation_choice == "ollama":
        generation = OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=_require_ollama_model(settings.OLLAMA_CHAT_MODEL, "OLLAMA_CHAT_MODEL"),
            scheduler=scheduler,
            num_ctx=settings.OLLAMA_NUM_CTX,
            thinking=settings.OLLAMA_THINKING,
        )
    elif generation_choice == "gemini":
        generation = gemini()
    else:
        raise ConfigurationError(detail=f"Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}'.")

    embedding = build_embedding_provider(scheduler, gemini() if settings.EMBEDDING_PROVIDER.lower() == "gemini" else None)

    moderation_choice = settings.MODERATION_PROVIDER.lower()
    if moderation_choice == "ollama":
        safety = generation if generation_choice == "ollama" else OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=_require_ollama_model(settings.OLLAMA_CHAT_MODEL, "OLLAMA_CHAT_MODEL"),
            scheduler=scheduler,
        )
    elif moderation_choice == "gemini":
        safety = gemini()
    else:
        raise ConfigurationError(detail=f"Unsupported MODERATION_PROVIDER '{settings.MODERATION_PROVIDER}'.")

    if generation is embedding is safety:
        return generation
    return CompositeProvider(generation, embedding, safety)