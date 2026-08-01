"""Grounded Gemini research adapter using Google Search citation annotations."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from src.models.research_models import (
    ResearchClaim,
    ResearchPacket,
    ResearchPacketStatus,
    ResearchRequest,
    ResearchSource,
)


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _kind(value: Any) -> str:
    raw = _field(value, "type", default="")
    raw = getattr(raw, "value", raw)
    return str(raw).casefold()


class GeminiGroundedResearchProvider:
    """Question-only provider that accepts only source-annotated grounded output."""

    provider_name = "gemini-grounded-search"

    def __init__(
        self,
        *,
        api_key: Optional[str],
        model_name: str,
        timeout_seconds: float = 30.0,
        client: Optional[Any] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Research timeout must be positive.")
        self.model_name = model_name.removeprefix("models/").strip()
        self.timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client
        if client is None and api_key and self.model_name:
            from google import genai

            self._client = genai.Client(api_key=api_key).aio

    def is_available(self) -> bool:
        return self._client is not None and bool(self.model_name)

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def research(self, request: ResearchRequest) -> ResearchPacket:
        if not self.is_available():
            raise RuntimeError("Gemini grounded research provider is unavailable.")
        interaction = await asyncio.wait_for(
            self._client.interactions.create(
                model=self.model_name,
                input=request.query,
                tools=[{"type": "google_search"}],
                timeout=self.timeout_seconds,
            ),
            timeout=self.timeout_seconds + 1.0,
        )
        return self._to_packet(interaction, request)

    def _to_packet(self, interaction: Any, request: ResearchRequest) -> ResearchPacket:
        answer_parts: list[str] = []
        sources: list[ResearchSource] = []
        claims: list[ResearchClaim] = []
        search_queries: list[str] = []
        source_id_by_url: dict[str, str] = {}

        for step in _field(interaction, "steps", default=[]) or []:
            step_kind = _kind(step)
            if "google_search_call" in step_kind:
                arguments = _field(step, "arguments", default={}) or {}
                for query in (_field(arguments, "queries", default=[]) or [])[:20]:
                    cleaned = str(query).strip()
                    if cleaned and cleaned not in search_queries:
                        search_queries.append(cleaned[:500])
                continue
            if "model_output" not in step_kind:
                continue
            for block in _field(step, "content", default=[]) or []:
                if "text" not in _kind(block):
                    continue
                text = str(_field(block, "text", default="") or "").strip()
                if not text:
                    continue
                answer_parts.append(text)
                for annotation in _field(block, "annotations", default=[]) or []:
                    if "url_citation" not in _kind(annotation):
                        continue
                    url = str(_field(annotation, "url", default="") or "").strip()[:2048]
                    title = str(_field(annotation, "title", default="") or "").strip()
                    try:
                        start = int(_field(annotation, "start_index", "startIndex"))
                        end = int(_field(annotation, "end_index", "endIndex"))
                    except (TypeError, ValueError):
                        continue
                    if not url or not 0 <= start < end <= len(text):
                        continue
                    source_id = source_id_by_url.get(url)
                    if source_id is None:
                        source_id = f"s{len(sources) + 1}"
                        source_id_by_url[url] = source_id
                        sources.append(
                            ResearchSource(
                                source_id=source_id,
                                title=(title or url)[:500],
                                url=url,
                            )
                        )
                    claims.append(
                        ResearchClaim(
                            text=text[start:end].strip(),
                            source_ids=[source_id],
                            confidence=0.85,
                            start_index=start,
                            end_index=end,
                        )
                    )

        answer = "\n\n".join(answer_parts).strip()
        verified = bool(answer and sources and claims)
        return ResearchPacket(
            request_id=request.request_id,
            decision_id=request.decision_id,
            query=request.query,
            status=(
                ResearchPacketStatus.COMPLETED
                if verified
                else ResearchPacketStatus.FAILED
            ),
            provider=self.provider_name,
            model=self.model_name,
            answer=answer or None,
            claims=claims,
            sources=sources,
            search_queries=search_queries,
            grounding_verified=verified,
            confidence=0.85 if verified else 0.0,
            caveats=(
                []
                if verified
                else ["Gemini response contained no usable source-annotated grounding."]
            ),
            context_policy=request.context_policy,
            context_chars=0,
        )
