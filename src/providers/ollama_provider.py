from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp

from src.core.exceptions import LLMServiceException
from src.providers.base import ProviderCapabilities, ProviderPurpose, ProviderRequest, ProviderResult
from src.providers.execution_scheduler import ModelExecutionScheduler


class OllamaProvider:
    """Local text and embedding provider backed by Ollama's HTTP API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        scheduler: ModelExecutionScheduler,
        request_timeout_seconds: float = 90.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.scheduler = scheduler
        self.request_timeout_seconds = request_timeout_seconds
        self.capabilities = ProviderCapabilities(
            provider="ollama",
            model=model,
            is_local=True,
            supports_structured_output=True,
        )

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        if request.image_base64 or request.audio_base64:
            raise LLMServiceException(
                detail="The local text-only provider does not yet accept image or audio input.",
                status_code=501,
            )

        async def operation() -> ProviderResult:
            return await self._generate(request)

        return await self.scheduler.execute(request.purpose, operation)

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
        result = await self.generate(
            ProviderRequest(
                purpose=ProviderPurpose.INTERACTIVE,
                prompt=prompt,
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stop_sequences=stop_sequences,
                image_base64=image_base64,
                audio_base64=audio_base64,
                image_mime_type=image_mime_type,
                audio_mime_type=audio_mime_type,
            )
        )
        return result.content

    async def generate_embedding(
        self, text: str, model_name: Optional[str] = None
    ) -> List[float]:
        raise LLMServiceException(
            detail=(
                f"Ollama model '{self.model}' is not configured as an embedding provider. "
                "Configure a separate local embedding model before enabling full local routing."
            ),
            status_code=501,
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
        if image_base64 or audio_base64:
            raise LLMServiceException(
                detail="Local multimodal moderation is not implemented.", status_code=501
            )
        # Local text routing is enabled only for trusted/local development inputs in this slice.
        # A dedicated safety provider replaces this permissive bridge before broader use.
        return {"is_safe": True, "provider": "ollama", "enforcement": "not_implemented"}

    async def _generate(self, request: ProviderRequest) -> ProviderResult:
        started = time.perf_counter()
        options: Dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_output_tokens,
        }
        if request.stop_sequences:
            options["stop"] = request.stop_sequences
        if request.context_budget:
            options["num_ctx"] = request.context_budget

        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
            "options": options,
        }
        timeout = aiohttp.ClientTimeout(total=request.timeout_seconds or self.request_timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    body = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise LLMServiceException(detail=f"Ollama generation request failed: {error}", status_code=503) from error

        content = body.get("response", "")
        if not content.strip():
            raise LLMServiceException(detail="Ollama returned no generated content.", status_code=502)
        return ProviderResult(
            provider="ollama",
            model=body.get("model", self.model),
            content=content,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            finish_reason=body.get("done_reason"),
            usage={
                "prompt_tokens": body.get("prompt_eval_count", 0),
                "completion_tokens": body.get("eval_count", 0),
            },
            capability_evidence={"text": True, "embeddings": True},
        )
