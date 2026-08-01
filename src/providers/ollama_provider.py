from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from src.core.exceptions import LLMServiceException
from src.providers.base import ProviderCapabilities, ProviderPurpose, ProviderRequest, ProviderResult
from src.providers.execution_scheduler import ModelExecutionScheduler

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Local text and embedding provider backed by Ollama's HTTP API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        scheduler: ModelExecutionScheduler,
        request_timeout_seconds: float = 90.0,
        num_ctx: Optional[int] = None,
        thinking: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.scheduler = scheduler
        self.request_timeout_seconds = request_timeout_seconds
        self.num_ctx = num_ctx
        self.thinking = thinking
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
        response_json: bool = False,
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
                response_json=response_json,
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
        context_window = request.context_budget or self.num_ctx
        if context_window:
            options["num_ctx"] = context_window

        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
            "think": self.thinking,
            "options": options,
        }
        if request.structured_output_schema:
            payload["format"] = request.structured_output_schema
        elif request.response_json:
            payload["format"] = "json"
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
            thinking = (body.get("thinking") or "").strip()
            hint = (
                "The model spent its output budget on reasoning tokens; set OLLAMA_THINKING=false."
                if thinking
                else "A prompt longer than num_ctx is the usual cause."
            )
            raise LLMServiceException(
                detail=(
                    f"Ollama returned no generated content (done_reason={body.get('done_reason')}, "
                    f"prompt_tokens={body.get('prompt_eval_count')}, num_ctx={options.get('num_ctx')}, "
                    f"num_predict={options.get('num_predict')}, thinking_chars={len(thinking)}). {hint}"
                ),
                status_code=502,
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        server_ms = round(body.get("total_duration", 0) / 1e6, 2)
        # latency minus server-side work is time spent queued behind another request on the same model.
        logger.info(
            "OLLAMA_CALL: model=%s latency=%.0fms server=%.0fms queue=%.0fms load=%.0fms "
            "prompt_eval=%.0fms eval=%.0fms prompt_tokens=%s completion_tokens=%s",
            body.get("model", self.model),
            latency_ms,
            server_ms,
            max(latency_ms - server_ms, 0.0),
            body.get("load_duration", 0) / 1e6,
            body.get("prompt_eval_duration", 0) / 1e6,
            body.get("eval_duration", 0) / 1e6,
            body.get("prompt_eval_count", 0),
            body.get("eval_count", 0),
        )
        return ProviderResult(
            provider="ollama",
            model=body.get("model", self.model),
            content=content,
            latency_ms=latency_ms,
            finish_reason=body.get("done_reason"),
            usage={
                "prompt_tokens": body.get("prompt_eval_count", 0),
                "completion_tokens": body.get("eval_count", 0),
            },
            capability_evidence={"text": True, "embeddings": False, "json": bool(payload.get("format"))},
        )
