from __future__ import annotations

import asyncio
import logging
import time
from typing import List

import aiohttp

from src.core.exceptions import LLMServiceException
from src.providers.base import EmbeddingModelIdentity, ProviderCapabilities, ProviderPurpose
from src.providers.execution_scheduler import ModelExecutionScheduler

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    """Dedicated local embedding adapter; independent of the chat model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        scheduler: ModelExecutionScheduler,
        request_timeout_seconds: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.scheduler = scheduler
        self.request_timeout_seconds = request_timeout_seconds
        self._vector_dimension: int | None = None
        self.capabilities = ProviderCapabilities(
            provider="ollama",
            model=model,
            is_local=True,
            supports_embeddings=True,
        )

    @property
    def identity(self) -> EmbeddingModelIdentity:
        return EmbeddingModelIdentity(
            provider="ollama",
            model=self.model,
            vector_dimension=self._vector_dimension,
        )

    async def verify(self) -> EmbeddingModelIdentity:
        """Resolve the runtime vector dimension before any collection is stamped with it."""
        if self._vector_dimension is None:
            await self.embed("embedding identity probe")
        return self.identity

    async def embed(self, text: str) -> List[float]:
        if not text:
            raise LLMServiceException(detail="Text cannot be empty for embedding generation.", status_code=400)
        return (await self._embed_many([text]))[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if any(not text for text in texts):
            raise LLMServiceException(detail="Text cannot be empty for embedding generation.", status_code=400)
        return await self._embed_many(texts)

    async def _embed_many(self, texts: List[str]) -> List[List[float]]:
        async def operation() -> List[List[float]]:
            started = time.perf_counter()
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_seconds)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": texts},
                    ) as response:
                        response.raise_for_status()
                        body = await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                raise LLMServiceException(
                    detail=f"Ollama embedding request failed: {error}", status_code=503
                ) from error

            embeddings = body.get("embeddings", [])
            if len(embeddings) != len(texts) or not all(embeddings):
                raise LLMServiceException(
                    detail=f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs.",
                    status_code=502,
                )
            for embedding in embeddings:
                self._record_dimension(len(embedding))
            logger.info(
                "OLLAMA_EMBED: model=%s inputs=%d chars=%d latency=%.0fms",
                self.model,
                len(texts),
                sum(len(text) for text in texts),
                (time.perf_counter() - started) * 1000,
            )
            return embeddings

        return await self.scheduler.execute(ProviderPurpose.EMBEDDING, operation)

    def _record_dimension(self, dimension: int) -> None:
        if self._vector_dimension is None:
            self._vector_dimension = dimension
        elif dimension != self._vector_dimension:
            raise LLMServiceException(
                detail=(
                    f"Ollama embedding dimension changed from {self._vector_dimension} "
                    f"to {dimension} for model '{self.model}'."
                ),
                status_code=502,
            )