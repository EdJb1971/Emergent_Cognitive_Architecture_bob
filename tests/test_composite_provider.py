from unittest.mock import AsyncMock, MagicMock

import pytest

from src.providers.base import EmbeddingModelIdentity, ProviderCapabilities
from src.providers.composite_provider import CompositeProvider


def _generation_provider():
    generation = AsyncMock()
    generation.capabilities = ProviderCapabilities(
        provider="ollama", model="gemma4:e4b", is_local=True
    )
    generation.generate_text.return_value = "local text"
    return generation


def _embedding_provider():
    embedding = MagicMock()
    embedding.capabilities = ProviderCapabilities(
        provider="ollama", model="embeddinggemma", is_local=True, supports_embeddings=True
    )
    embedding.identity = EmbeddingModelIdentity(
        provider="ollama", model="embeddinggemma", vector_dimension=768
    )
    embedding.embed = AsyncMock(return_value=[0.1, 0.2])
    embedding.verify = AsyncMock(return_value=embedding.identity)
    return embedding


def _safety_provider(is_local: bool):
    safety = AsyncMock()
    safety.capabilities = ProviderCapabilities(
        provider="ollama" if is_local else "gemini",
        model="gemma4:e4b" if is_local else "models/gemini-2.0-flash-lite",
        is_local=is_local,
    )
    safety.moderate_content.return_value = {"is_safe": True}
    return safety


@pytest.mark.asyncio
async def test_composite_provider_routes_each_capability_to_its_provider():
    generation = _generation_provider()
    embedding = _embedding_provider()
    safety = _safety_provider(is_local=True)
    provider = CompositeProvider(generation, embedding, safety)

    assert await provider.generate_text("hello") == "local text"
    assert await provider.generate_embedding("hello") == [0.1, 0.2]
    assert await provider.moderate_content(text="hello") == {"is_safe": True}
    assert provider.capabilities.is_local is True
    generation.generate_text.assert_awaited_once()
    embedding.embed.assert_awaited_once_with("hello")
    safety.moderate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_composite_provider_is_not_local_when_safety_is_cloud_backed():
    provider = CompositeProvider(
        _generation_provider(), _embedding_provider(), _safety_provider(is_local=False)
    )

    assert provider.capabilities.is_local is False


@pytest.mark.asyncio
async def test_composite_provider_exposes_embedding_identity():
    embedding = _embedding_provider()
    provider = CompositeProvider(_generation_provider(), embedding, _safety_provider(is_local=True))

    assert provider.identity.describe() == "ollama/embeddinggemma@768d"
    assert (await provider.verify()).vector_dimension == 768
    embedding.verify.assert_awaited_once()
