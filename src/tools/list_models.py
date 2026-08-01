"""Lists models actually available to the configured providers.

Model identifiers are discovered, never assumed. A label seen in a blog post is
not evidence that an API accepts it, and picking the research model matters:
without web grounding, that node's answers are bounded by its training cutoff.

Usage:
    python -m src.tools.list_models
    python -m src.tools.list_models --provider gemini --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from src.core.config import settings


async def list_ollama_models() -> List[Dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags") as response:
            response.raise_for_status()
            payload = await response.json()

    models = []
    for entry in payload.get("models", []):
        details = entry.get("details", {})
        models.append(
            {
                "name": entry.get("name"),
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
                "family": details.get("family"),
                "size_gb": round(entry.get("size", 0) / 1_000_000_000, 2),
            }
        )
    return sorted(models, key=lambda m: m["name"] or "")


def list_gemini_models() -> List[Dict[str, Any]]:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set; cannot query the Gemini model list.")

    import google.generativeai as genai

    genai.configure(api_key=settings.GEMINI_API_KEY)
    models = []
    for model in genai.list_models():
        models.append(
            {
                "name": model.name,
                "display_name": getattr(model, "display_name", None),
                "input_token_limit": getattr(model, "input_token_limit", None),
                "output_token_limit": getattr(model, "output_token_limit", None),
                "methods": sorted(getattr(model, "supported_generation_methods", []) or []),
            }
        )
    return sorted(models, key=lambda m: m["name"])


def _print_ollama(models: List[Dict[str, Any]]) -> None:
    print(f"\nOllama ({settings.OLLAMA_BASE_URL}) — {len(models)} installed")
    for model in models:
        size = f"{model['parameter_size']}" if model["parameter_size"] else "?"
        print(f"  {model['name']:<32} {size:>8}  {model['quantization'] or '':<10} {model['size_gb']} GB")


def _print_gemini(models: List[Dict[str, Any]]) -> None:
    generation = [m for m in models if "generateContent" in m["methods"]]
    embedding = [m for m in models if "embedContent" in m["methods"]]
    print(f"\nGemini — {len(generation)} generation, {len(embedding)} embedding")
    print("\n  generateContent:")
    for model in generation:
        limits = f"in={model['input_token_limit']} out={model['output_token_limit']}"
        print(f"    {model['name']:<45} {limits}")
    print("\n  embedContent:")
    for model in embedding:
        print(f"    {model['name']:<45} in={model['input_token_limit']}")


async def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Discover models available to the configured providers.")
    parser.add_argument("--provider", choices=("gemini", "ollama", "all"), default="all")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")
    args = parser.parse_args(argv)

    result: Dict[str, Any] = {}
    exit_code = 0

    if args.provider in ("ollama", "all"):
        try:
            result["ollama"] = await list_ollama_models()
        except Exception as error:
            result["ollama_error"] = str(error)
            exit_code = 1

    if args.provider in ("gemini", "all"):
        try:
            result["gemini"] = await asyncio.to_thread(list_gemini_models)
        except Exception as error:
            result["gemini_error"] = str(error)
            exit_code = 1

    if args.json:
        print(json.dumps(result, indent=2))
        return exit_code

    if "ollama" in result:
        _print_ollama(result["ollama"])
    if "ollama_error" in result:
        print(f"\nOllama unavailable: {result['ollama_error']}")
    if "gemini" in result:
        _print_gemini(result["gemini"])
    if "gemini_error" in result:
        print(f"\nGemini unavailable: {result['gemini_error']}")

    print("\nSet OLLAMA_CHAT_MODEL / OLLAMA_EMBEDDING_MODEL / LLM_MODEL_NAME from names shown above.")
    return exit_code


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
