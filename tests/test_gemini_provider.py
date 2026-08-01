from unittest.mock import AsyncMock

import pytest

from src.providers.gemini_provider import GeminiProvider


@pytest.mark.asyncio
async def test_gemini_provider_delegates_to_existing_service():
    service = AsyncMock()
    service.generate_text.return_value = "response"
    service.generate_embedding.return_value = [0.1, 0.2]
    service.moderate_content.return_value = {"is_safe": True}
    provider = GeminiProvider(service=service)

    assert await provider.generate_text("hello") == "response"
    assert await provider.generate_embedding("hello") == [0.1, 0.2]
    assert await provider.moderate_content(text="hello") == {"is_safe": True}
    assert provider.capabilities.provider == "gemini"
    assert provider.capabilities.is_local is False
