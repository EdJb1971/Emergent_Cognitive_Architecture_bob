from unittest.mock import AsyncMock

import pytest

from src.core.exceptions import LLMServiceException
from src.providers.base import ProviderPurpose, ProviderRequest, ProviderResult
from src.providers.execution_scheduler import ModelExecutionScheduler
from src.providers.ollama_provider import OllamaProvider


@pytest.mark.asyncio
async def test_ollama_provider_schedules_text_generation():
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        scheduler=ModelExecutionScheduler(),
    )
    provider._generate = AsyncMock(
        return_value=ProviderResult(
            provider="ollama",
            model="gemma4:e4b",
            content="local response",
            latency_ms=1.0,
        )
    )

    result = await provider.generate(
        ProviderRequest(purpose=ProviderPurpose.INTERACTIVE, prompt="hello")
    )

    assert result.content == "local response"
    provider._generate.assert_awaited_once()
    assert provider.capabilities.is_local is True
    assert provider.capabilities.supports_embeddings is False


@pytest.mark.asyncio
async def test_ollama_provider_rejects_embedding_and_unverified_image_requests():
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        scheduler=ModelExecutionScheduler(),
    )

    with pytest.raises(LLMServiceException, match="embedding provider"):
        await provider.generate_embedding("hello")
    with pytest.raises(LLMServiceException, match="no verified vision capability"):
        await provider.generate(
            ProviderRequest(
                purpose=ProviderPurpose.INTERACTIVE,
                prompt="describe this",
                image_base64="image-data",
            )
        )
    with pytest.raises(LLMServiceException, match="no verified audio capability"):
        await provider.generate(
            ProviderRequest(
                purpose=ProviderPurpose.INTERACTIVE,
                prompt="listen conservatively",
                audio_base64="wav-data",
            )
        )


@pytest.mark.asyncio
async def test_ollama_provider_accepts_image_only_after_capability_verification():
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        scheduler=ModelExecutionScheduler(),
        supports_images=True,
    )
    provider._generate = AsyncMock(
        return_value=ProviderResult(
            provider="ollama",
            model="gemma4:e4b",
            content='{"description":"local"}',
            latency_ms=1.0,
        )
    )

    await provider.generate(
        ProviderRequest(
            purpose=ProviderPurpose.INTERACTIVE,
            prompt="observe",
            image_base64="image-data",
        )
    )

    request = provider._generate.await_args.args[0]
    assert request.image_base64 == "image-data"
    assert provider.capabilities.supports_images is True


@pytest.mark.asyncio
async def test_ollama_provider_accepts_audio_only_after_capability_verification():
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        scheduler=ModelExecutionScheduler(),
        supports_audio=True,
    )
    provider._generate = AsyncMock(
        return_value=ProviderResult(
            provider="ollama",
            model="gemma4:e4b",
            content='{"speech_detected":false}',
            latency_ms=1.0,
        )
    )

    await provider.generate(
        ProviderRequest(
            purpose=ProviderPurpose.INTERACTIVE,
            prompt="observe",
            audio_base64="wav-data",
            audio_mime_type="audio/wav",
        )
    )

    request = provider._generate.await_args.args[0]
    assert request.audio_base64 == "wav-data"
    assert provider.capabilities.supports_audio is True
