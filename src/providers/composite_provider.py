from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.providers.base import (
    EmbeddingModelIdentity,
    EmbeddingProvider,
    LLMProvider,
    ProviderCapabilities,
)


class CompositeProvider:
    """Composes independently selected generation, embedding, and safety providers."""

    def __init__(
        self,
        generation_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        safety_provider: LLMProvider,
    ):
        self.generation_provider = generation_provider
        self.embedding_provider = embedding_provider
        self.safety_provider = safety_provider
        generation = generation_provider.capabilities
        embedding = embedding_provider.capabilities
        safety = safety_provider.capabilities
        self.capabilities = ProviderCapabilities(
            provider=f"{generation.provider}+{embedding.provider}+{safety.provider}",
            model=(
                f"generation={generation.model};"
                f"embedding={embedding.model};"
                f"safety={safety.model}"
            ),
            is_local=generation.is_local and embedding.is_local and safety.is_local,
            supports_images=generation.supports_images,
            supports_audio=generation.supports_audio,
            supports_embeddings=True,
            supports_structured_output=generation.supports_structured_output,
            supports_tools=generation.supports_tools,
        )

    @property
    def identity(self) -> EmbeddingModelIdentity:
        return self.embedding_provider.identity

    async def embed(self, text: str) -> List[float]:
        return await self.embedding_provider.embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return await self.embedding_provider.embed_batch(texts)

    async def verify(self) -> EmbeddingModelIdentity:
        return await self.embedding_provider.verify()

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
        response_json: bool = False,
    ) -> str:
        return await self.generation_provider.generate_text(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            safety_settings=safety_settings,
            image_base64=image_base64,
            audio_base64=audio_base64,
            image_mime_type=image_mime_type,
            audio_mime_type=audio_mime_type,
            response_json=response_json,
        )

    async def generate_embedding(
        self, text: str, model_name: Optional[str] = None
    ) -> List[float]:
        return await self.embedding_provider.embed(text)

    async def moderate_content(
        self,
        text: Optional[str] = None,
        image_base64: Optional[str] = None,
        audio_base64: Optional[str] = None,
        model_name: Optional[str] = None,
        image_mime_type: Optional[str] = "image/jpeg",
        audio_mime_type: Optional[str] = "audio/wav",
    ) -> Dict[str, Any]:
        return await self.safety_provider.moderate_content(
            text=text,
            image_base64=image_base64,
            audio_base64=audio_base64,
            model_name=model_name,
            image_mime_type=image_mime_type,
            audio_mime_type=audio_mime_type,
        )
