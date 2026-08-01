from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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


class ProviderPurpose(str, Enum):
    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    EMBEDDING = "embedding"
    MODERATION = "moderation"


@dataclass(frozen=True)
class ProviderRequest:
    purpose: ProviderPurpose
    prompt: str
    model: Optional[str] = None
    required_capabilities: tuple[str, ...] = ()
    structured_output_schema: Optional[Dict[str, Any]] = None
    response_json: bool = False
    privacy_classification: str = "local"
    context_budget: Optional[int] = None
    timeout_seconds: float = 90.0
    temperature: float = 0.7
    max_output_tokens: int = 2048
    stop_sequences: Optional[List[str]] = None
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None
    image_mime_type: Optional[str] = "image/jpeg"
    audio_mime_type: Optional[str] = "audio/wav"


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    model: str
    content: str
    latency_ms: float
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    parse_repaired: bool = False
    capability_evidence: Dict[str, bool] = field(default_factory=dict)


EMBEDDING_IDENTITY_METADATA_VERSION = 1


@dataclass(frozen=True)
class EmbeddingModelIdentity:
    provider: str
    model: str
    vector_dimension: Optional[int] = None

    def describe(self) -> str:
        dimension = self.vector_dimension if self.vector_dimension is not None else "unknown"
        return f"{self.provider}/{self.model}@{dimension}d"

    def as_collection_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "embedding_dimension": self.vector_dimension,
            "embedding_identity_version": EMBEDDING_IDENTITY_METADATA_VERSION,
        }


class EmbeddingIdentityMismatch(Exception):
    """Raised when stored vectors were produced by a different embedding model."""

    def __init__(self, collection_name: str, stored: str, active: str):
        self.collection_name = collection_name
        self.stored = stored
        self.active = active
        super().__init__(
            f"Collection '{collection_name}' holds vectors from {stored} but the active "
            f"embedding provider is {active}. Matching dimensions do not make these vector "
            f"spaces compatible; rebuild into a new collection instead of mixing them."
        )


def read_collection_identity(metadata: Optional[Dict[str, Any]]) -> Optional[EmbeddingModelIdentity]:
    if not metadata or not metadata.get("embedding_provider"):
        return None
    return EmbeddingModelIdentity(
        provider=str(metadata["embedding_provider"]),
        model=str(metadata.get("embedding_model", "")),
        vector_dimension=metadata.get("embedding_dimension"),
    )


def check_embedding_identity(
    collection_name: str,
    stored_metadata: Optional[Dict[str, Any]],
    active: EmbeddingModelIdentity,
) -> Optional[EmbeddingIdentityMismatch]:
    """Compare provider/model rather than dimension; distinct models often share a dimension."""
    stored = read_collection_identity(stored_metadata)
    if stored is None:
        return None
    if stored.provider == active.provider and stored.model == active.model:
        return None
    return EmbeddingIdentityMismatch(collection_name, stored.describe(), active.describe())


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
        response_json: bool = False,
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


class EmbeddingProvider(Protocol):
    capabilities: ProviderCapabilities
    identity: EmbeddingModelIdentity

    async def embed(self, text: str) -> List[float]: ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

    async def verify(self) -> EmbeddingModelIdentity: ...
