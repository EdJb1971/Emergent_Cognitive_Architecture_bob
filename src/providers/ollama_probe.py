from __future__ import annotations

import asyncio
from typing import Any, Dict

import aiohttp


class OllamaProbe:
    """Read-only availability and installed-model probe for a local Ollama server."""

    def __init__(self, base_url: str, model_name: str, embedding_model: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.embedding_model = embedding_model

    async def probe(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "configured_model": self.model_name or None,
            "configured_embedding_model": self.embedding_model or None,
            "base_url": self.base_url,
            "available": False,
            "model_installed": False,
            "embedding_model_installed": False,
            "models": [],
        }
        try:
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    response.raise_for_status()
                    payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            status["message"] = f"Ollama unavailable: {error}"
            return status

        models = payload.get("models", [])
        model_names = [model.get("name", "") for model in models]
        status["available"] = True
        status["models"] = model_names
        status["model_installed"] = self.model_name in model_names
        status["embedding_model_installed"] = (
            not self.embedding_model or self.embedding_model in model_names
        )
        status["message"] = (
            "Configured chat and embedding models are installed."
            if status["model_installed"] and status["embedding_model_installed"]
            else "A configured Ollama model is not installed."
        )
        return status
