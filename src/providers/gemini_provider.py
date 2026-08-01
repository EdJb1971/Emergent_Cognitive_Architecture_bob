from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.providers.base import ProviderCapabilities
from src.services.llm_integration_service import LLMIntegrationService


class GeminiProvider:
    """Compatibility adapter that preserves the current Gemini runtime behavior."""

    def __init__(self, service: Optional[LLMIntegrationService] = None):
        self._service = service or LLMIntegrationService()
        self.capabilities = ProviderCapabilities(
            provider="gemini",
            model=settings.LLM_MODEL_NAME,
            is_local=False,
            supports_images=True,
            supports_audio=True,
            supports_embeddings=True,
            supports_structured_output=True,
        )

    async def generate_text(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        image_base64: Optional[str] = None,
        audio_base64: Optional[str] = None,
        image_mime_type: Optional[str] = "image/jpeg",
        audio_mime_type: Optional[str] = "audio/wav",
    ) -> str:
        return await self._service.generate_text(
            prompt=prompt,
            model_name=model_name or settings.LLM_MODEL_NAME,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            safety_settings=safety_settings,
            image_base64=image_base64,
            audio_base64=audio_base64,
            image_mime_type=image_mime_type,
            audio_mime_type=audio_mime_type,
        )

    async def generate_embedding(
        self, text: str, model_name: Optional[str] = None
    ) -> List[float]:
        return await self._service.generate_embedding(
            text=text,
            model_name=model_name or settings.EMBEDDING_MODEL_NAME,
        )

    async def moderate_content(
        self,
        text: Optional[str] = None,
        image_base64: Optional[str] = None,
        audio_base64: Optional[str] = None,
        model_name: Optional[str] = None,
        image_mime_type: Optional[str] = "image/jpeg",
        audio_mime_type: Optional[str] = "audio/wav",
    ) -> Dict[str, Any]:
        return await self._service.moderate_content(
            text=text,
            image_base64=image_base64,
            audio_base64=audio_base64,
            model_name=model_name or settings.LLM_MODEL_FOR_MODERATION,
            image_mime_type=image_mime_type,
            audio_mime_type=audio_mime_type,
        )
