from __future__ import annotations

import asyncio
from typing import Any, Dict

import aiohttp


class OllamaProbe:
    """Read-only availability and installed-model probe for a local Ollama server."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        embedding_model: str = "",
        vision_model: str = "",
        audio_model: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.vision_model = vision_model
        self.audio_model = audio_model

    async def probe(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "configured_model": self.model_name or None,
            "configured_embedding_model": self.embedding_model or None,
            "configured_vision_model": self.vision_model or None,
            "configured_audio_model": self.audio_model or None,
            "base_url": self.base_url,
            "available": False,
            "model_installed": False,
            "embedding_model_installed": False,
            "vision_model_installed": False,
            "vision_model_supports_images": False,
            "vision_model_capabilities": [],
            "audio_model_installed": False,
            "audio_model_supports_audio": False,
            "audio_model_capabilities": [],
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
        status["vision_model_installed"] = (
            not self.vision_model or self.vision_model in model_names
        )
        status["audio_model_installed"] = (
            not self.audio_model or self.audio_model in model_names
        )
        if self.vision_model and status["vision_model_installed"]:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                    async with session.post(
                        f"{self.base_url}/api/show",
                        json={"model": self.vision_model},
                    ) as response:
                        response.raise_for_status()
                        model_info = await response.json()
                capabilities = [str(item).lower() for item in model_info.get("capabilities", [])]
                status["vision_model_capabilities"] = capabilities
                status["vision_model_supports_images"] = "vision" in capabilities
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                status["vision_probe_message"] = f"Vision capability probe failed: {error}"
        if self.audio_model and status["audio_model_installed"]:
            if self.audio_model == self.vision_model and status["vision_model_capabilities"]:
                capabilities = status["vision_model_capabilities"]
            else:
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                        async with session.post(
                            f"{self.base_url}/api/show",
                            json={"model": self.audio_model},
                        ) as response:
                            response.raise_for_status()
                            model_info = await response.json()
                    capabilities = [str(item).lower() for item in model_info.get("capabilities", [])]
                except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                    capabilities = []
                    status["audio_probe_message"] = f"Audio capability probe failed: {error}"
            status["audio_model_capabilities"] = capabilities
            status["audio_model_supports_audio"] = "audio" in capabilities
        status["message"] = (
            "Configured Ollama models are installed."
            if status["model_installed"] and status["embedding_model_installed"] and status["vision_model_installed"] and status["audio_model_installed"]
            else "A configured Ollama model is not installed."
        )
        return status
