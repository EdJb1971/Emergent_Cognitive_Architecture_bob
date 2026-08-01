from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    model: str
    is_local: bool
    supports_images: bool = False
    supports_audio: bool = False
    supports_embeddings: bool = False
    supports_structured_output: bool = False
    supports_tools: bool = False


class LLMProvider(Protocol):
    capabilities: ProviderCapabilities

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
    ) -> str: ...

    async def generate_embedding(
        self, text: str, model_name: Optional[str] = None
    ) -> List[float]: ...

    async def moderate_content(
        self,
        text: Optional[str] = None,
        image_base64: Optional[str] = None,
        audio_base64: Optional[str] = None,
        model_name: Optional[str] = None,
        image_mime_type: Optional[str] = "image/jpeg",
        audio_mime_type: Optional[str] = "audio/wav",
    ) -> Dict[str, Any]: ...
