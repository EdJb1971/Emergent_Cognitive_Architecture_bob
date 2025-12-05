import logging
import asyncio
import random
import re
import google.generativeai as genai
from google.generativeai.types import BlockedPromptException, HarmCategory, HarmBlockThreshold
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.config import settings
from src.core.exceptions import LLMServiceException, ConfigurationError

logger = logging.getLogger(__name__)

# Maximum input tokens for Gemini 2.5 Flash Lite (1M context window, reserve for response)
MAX_INPUT_TOKENS = 800000  # Conservative limit to prevent 1M+ token explosions

# Embedding limits (Google's text-embedding-004 has ~36KB payload limit)
MAX_EMBEDDING_PAYLOAD_BYTES = 35000  # Conservative limit under 36KB
MAX_EMBEDDING_CHUNK_TOKENS = 2048   # Conservative token limit per chunk

# Define retry settings for LLM calls
LLM_RETRY_SETTINGS = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=4, max=10),
    "retry": retry_if_exception_type(LLMServiceException) # Retry on our custom LLM exception
}

class LLMIntegrationService:
    """
    A centralized wrapper service for interacting with Google Gemini models.
    Manages API keys, model versions, temperature settings, and handles retries/rate limits.
    Now supports multimodal input.
    """
    # Class-scoped semaphores to cap concurrency per model and globally
    _model_semaphores: Dict[str, asyncio.Semaphore] = {}
    _global_semaphore: Optional[asyncio.Semaphore] = None

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ConfigurationError(detail="GEMINI_API_KEY is not set in environment variables.")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Initialize global semaphore once
        if not LLMIntegrationService._global_semaphore:
            LLMIntegrationService._global_semaphore = asyncio.BoundedSemaphore(settings.LLM_GLOBAL_MAX_CONCURRENCY)
        logger.info("LLMIntegrationService initialized.")

    def _get_model_semaphore(self, model_name: str) -> asyncio.Semaphore:
        """Get or create a bounded semaphore for a given model name."""
        sem = LLMIntegrationService._model_semaphores.get(model_name)
        if sem is None:
            sem = asyncio.BoundedSemaphore(settings.LLM_MAX_CONCURRENCY_PER_MODEL)
            LLMIntegrationService._model_semaphores[model_name] = sem
        return sem

    def _chunk_text_for_embedding(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces that fit within embedding API limits.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Check if text is already small enough
        text_bytes = len(text.encode('utf-8'))
        if text_bytes <= MAX_EMBEDDING_PAYLOAD_BYTES:
            return [text]

        chunks = []
        current_chunk = ""
        current_bytes = 0

        # Split by sentences/paragraphs first for better semantic chunking
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        for sentence in sentences:
            sentence_bytes = len(sentence.encode('utf-8'))

            # If adding this sentence would exceed the limit, save current chunk
            if current_bytes + sentence_bytes > MAX_EMBEDDING_PAYLOAD_BYTES and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_bytes = sentence_bytes
            else:
                current_chunk += " " + sentence
                current_bytes += sentence_bytes + 1  # +1 for space

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If we still have very large chunks, split them by words
        final_chunks = []
        for chunk in chunks:
            chunk_bytes = len(chunk.encode('utf-8'))
            if chunk_bytes <= MAX_EMBEDDING_PAYLOAD_BYTES:
                final_chunks.append(chunk)
            else:
                # Split by words
                words = chunk.split()
                word_chunk = ""
                word_bytes = 0

                for word in words:
                    word_size = len(word.encode('utf-8')) + 1  # +1 for space

                    if word_bytes + word_size > MAX_EMBEDDING_PAYLOAD_BYTES and word_chunk:
                        final_chunks.append(word_chunk.strip())
                        word_chunk = word
                        word_bytes = len(word.encode('utf-8'))
                    else:
                        if word_chunk:
                            word_chunk += " " + word
                            word_bytes += word_size
                        else:
                            word_chunk = word
                            word_bytes = len(word.encode('utf-8'))

                if word_chunk.strip():
                    final_chunks.append(word_chunk.strip())

        return final_chunks

    def _extract_retry_delay_seconds(self, error: Exception) -> float:
        """Parse retry delay seconds from common Gemini 429 error formats, else default."""
        msg = str(error)
        # Pattern 1: "Please retry in 59.592074012s."
        m = re.search(r"Please retry in\s*([0-9.]+)s", msg)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        # Pattern 2: retry_delay {\n  seconds: 59\n}
        m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        # Fallback base
        return float(getattr(settings, "LLM_429_BASE_DELAY_SEC", 10.0))

    @retry(**LLM_RETRY_SETTINGS)
    async def generate_text(
        self,
        prompt: str,
        model_name: str = settings.LLM_MODEL_NAME,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        image_base64: Optional[str] = None,
        audio_base64: Optional[str] = None,
        # SEC-001: Added explicit mime_type arguments
        image_mime_type: Optional[str] = "image/jpeg", 
        audio_mime_type: Optional[str] = "audio/wav"
    ) -> str:
        """
        Generates text using a specified Google Gemini model, supporting multimodal input.

        Args:
            prompt (str): The input prompt for text generation.
            model_name (str): The name of the Gemini model to use.
            temperature (float): Controls the randomness of the output (0.0 to 1.0).
            max_output_tokens (int): Maximum number of tokens to generate.
            stop_sequences (Optional[List[str]]): Sequences at which to stop generation.
            safety_settings (Optional[List[Dict[str, Any]]]): Custom safety settings.
            image_base64 (Optional[str]): Base64 encoded image data.
            audio_base64 (Optional[str]): Base64 encoded audio data.
            image_mime_type (Optional[str]): MIME type for the image data (e.g., 'image/png').
            audio_mime_type (Optional[str]): MIME type for the audio data (e.g., 'audio/mpeg').

        Returns:
            str: The generated text.

        Raises:
            LLMServiceException: If there's an error during text generation or prompt is blocked.
        """
        if not prompt and not image_base64 and not audio_base64:
            raise LLMServiceException(detail="At least one of prompt, image_base64, or audio_base64 must be provided for generation.")

        try:
            model = genai.GenerativeModel(model_name)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "stop_sequences": stop_sequences if stop_sequences else []
            }
            # Default safety settings if not provided
            if safety_settings is None:
                safety_settings = [
                    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                ]

            # Construct content parts for multimodal input
            content_parts: List[Union[str, Dict[str, str]]] = []
            if prompt: content_parts.append(prompt)
            # SEC-001: Use provided mime_type
            if image_base64: content_parts.append({"mime_type": image_mime_type, "data": image_base64})
            if audio_base64: content_parts.append({"mime_type": audio_mime_type, "data": audio_base64})

            # SAFETY: Token count guard to prevent context explosion
            if prompt:
                # Conservative estimate: 1 token ~= 4 characters for English
                estimated_tokens = len(prompt) // 4
                if estimated_tokens > MAX_INPUT_TOKENS:
                    logger.error(
                        f"Prompt too large! Estimated {estimated_tokens} tokens (limit: {MAX_INPUT_TOKENS}). "
                        f"Prompt length: {len(prompt)} chars. First 500 chars: {prompt[:500]}"
                    )
                    raise LLMServiceException(
                        detail=f"Prompt exceeds token limit ({estimated_tokens} > {MAX_INPUT_TOKENS}). "
                               f"This indicates a context accumulation bug. Check memory consolidation and agent output sizes.",
                        status_code=400
                    )

            # Concurrency control: global + per-model semaphore
            model_sem = self._get_model_semaphore(model_name)
            async with LLMIntegrationService._global_semaphore:
                async with model_sem:
                    response = await model.generate_content_async(
                        content_parts,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )

            if response.candidates:
                # Concatenate all text parts from the response
                generated_text_parts = [part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')]
                return "".join(generated_text_parts)
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                logger.warning(f"Prompt blocked by safety settings: {block_reason}. Categories: {response.prompt_feedback.safety_ratings}")
                raise LLMServiceException(detail=f"Prompt blocked due to safety concerns: {block_reason}", status_code=400)
            else:
                logger.error(f"LLM text generation failed with no candidates and no block reason for prompt: {prompt[:100]}...")
                raise LLMServiceException(detail="LLM text generation failed: No content generated.")

        except BlockedPromptException as e:
            logger.warning(f"Prompt blocked by LLM: {e}")
            raise LLMServiceException(detail=f"Prompt blocked by LLM: {e}", status_code=400)
        except Exception as e:
            # Detect 429 rate limit and respect server-provided retry delay if present
            msg = str(e)
            if "429" in msg or "ResourceExhausted" in msg:
                delay = self._extract_retry_delay_seconds(e)
                jitter = random.uniform(0.0, getattr(settings, "LLM_429_JITTER_SEC", 0.5))
                total_sleep = max(0.0, delay + jitter)
                logger.error(
                    f"Rate limit encountered for model {model_name}. Sleeping for {total_sleep:.1f}s before retry. Raw error: {msg}"
                )
                try:
                    await asyncio.sleep(total_sleep)
                except Exception:
                    pass
            logger.exception(f"Error during LLM text generation for model {model_name}: {e}")
            raise LLMServiceException(detail=f"Failed to generate text: {e}")

    @retry(**LLM_RETRY_SETTINGS)
    async def generate_embedding(
        self,
        text: str,
        model_name: str = settings.EMBEDDING_MODEL_NAME
    ) -> List[float]:
        """
        Generates a vector embedding for the given text using a specified Google Gemini embedding model.
        Automatically chunks large texts to fit within API payload limits.

        Args:
            text (str): The input text to embed.
            model_name (str): The name of the embedding model to use.

        Returns:
            List[float]: The embedding vector as a list of floats.

        Raises:
            LLMServiceException: If there's an error during embedding generation.
        """
        if not text:
            raise LLMServiceException(detail="Text cannot be empty for embedding generation.")

        try:
            # Chunk text if it's too large
            text_chunks = self._chunk_text_for_embedding(text)

            if len(text_chunks) == 1:
                # Single chunk - use original logic
                model_sem = self._get_model_semaphore(model_name)
                async with LLMIntegrationService._global_semaphore:
                    async with model_sem:
                        response = await genai.embed_content_async(
                            model=model_name,
                            content=text
                        )
                if response and response.get("embedding"):
                    return response["embedding"]
                else:
                    logger.error(f"LLM embedding generation failed for text: {text[:100]}...")
                    raise LLMServiceException(detail="LLM embedding generation failed: No embedding generated.")
            else:
                # Multiple chunks - generate embeddings for each and average them
                logger.debug(f"Text too large for single embedding ({len(text.encode('utf-8'))} bytes), chunking into {len(text_chunks)} pieces")

                chunk_embeddings = []
                model_sem = self._get_model_semaphore(model_name)

                async with LLMIntegrationService._global_semaphore:
                    async with model_sem:
                        for i, chunk in enumerate(text_chunks):
                            try:
                                response = await genai.embed_content_async(
                                    model=model_name,
                                    content=chunk
                                )
                                if response and response.get("embedding"):
                                    chunk_embeddings.append(response["embedding"])
                                else:
                                    logger.warning(f"Failed to generate embedding for chunk {i+1}/{len(text_chunks)}")
                            except Exception as chunk_error:
                                logger.warning(f"Error generating embedding for chunk {i+1}/{len(text_chunks)}: {chunk_error}")
                                continue

                if not chunk_embeddings:
                    raise LLMServiceException(detail="Failed to generate embeddings for any text chunks.")

                # Average the embeddings
                embedding_dim = len(chunk_embeddings[0])
                averaged_embedding = []

                for dim in range(embedding_dim):
                    dim_sum = sum(chunk_emb[dim] for chunk_emb in chunk_embeddings)
                    averaged_embedding.append(dim_sum / len(chunk_embeddings))

                logger.debug(f"Successfully generated averaged embedding from {len(chunk_embeddings)} chunks")
                return averaged_embedding

        except Exception as e:
            msg = str(e)
            if "429" in msg or "ResourceExhausted" in msg:
                delay = self._extract_retry_delay_seconds(e)
                jitter = random.uniform(0.0, getattr(settings, "LLM_429_JITTER_SEC", 0.5))
                total_sleep = max(0.0, delay + jitter)
                logger.error(
                    f"Rate limit encountered for embedding model {model_name}. Sleeping for {total_sleep:.1f}s before retry. Raw error: {msg}"
                )
                try:
                    await asyncio.sleep(total_sleep)
                except Exception:
                    pass
            logger.exception(f"Error during LLM embedding generation for model {model_name}: {e}")
            raise LLMServiceException(detail=f"Failed to generate embedding: {e}")

    @retry(**LLM_RETRY_SETTINGS)
    async def moderate_content(
        self,
        text: Optional[str] = None,
        image_base64: Optional[str] = None,
        audio_base64: Optional[str] = None,
        model_name: str = settings.LLM_MODEL_FOR_MODERATION,
        # SEC-001 related: Add mime types for moderation as well
        image_mime_type: Optional[str] = "image/jpeg",
        audio_mime_type: Optional[str] = "audio/wav"
    ) -> Dict[str, Any]:
        """
        Moderates content using Google Gemini's safety settings, now supporting multimodal input.

        Args:
            text (Optional[str]): The input text to moderate.
            image_base64 (Optional[str]): Base64 encoded image data to moderate.
            audio_base64 (Optional[str]): Base64 encoded audio data to moderate.
            model_name (str): The name of the Gemini model to use for moderation.
            image_mime_type (Optional[str]): MIME type for the image data.
            audio_mime_type (Optional[str]): MIME type for the audio data.

        Returns:
            Dict[str, Any]: A dictionary containing moderation results, including safety ratings.
                            Returns {'is_safe': True} if no harm is detected above threshold.

        Raises:
            LLMServiceException: If there's an error during moderation.
        """
        if not text and not image_base64 and not audio_base64:
            return {"is_safe": True, "message": "No content provided for moderation."}

        try:
            model = genai.GenerativeModel(model_name)
            
            content_parts: List[Union[str, Dict[str, str]]] = []
            if text: content_parts.append(text)
            if image_base64: content_parts.append({"mime_type": image_mime_type, "data": image_base64})
            if audio_base64: content_parts.append({"mime_type": audio_mime_type, "data": audio_base64})

            # Use a simple prompt to trigger safety checks, as generate_content_async applies them by default.
            # If only multimodal content is provided, a minimal prompt might still be needed for some models.
            # For Gemini, passing content_parts directly is usually sufficient for moderation.
            # However, to ensure a response, a minimal prompt is safer.
            if not text and (image_base64 or audio_base64):
                content_parts.insert(0, "Analyze this content for safety.")

            response = await model.generate_content_async(
                content_parts,
                safety_settings=[
                    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                ]
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                logger.warning(f"Content blocked by safety settings: {block_reason}. Categories: {response.prompt_feedback.safety_ratings}")
                return {"is_safe": False, "block_reason": block_reason, "safety_ratings": [r.to_dict() for r in response.prompt_feedback.safety_ratings]}
            else:
                # If no block reason, it's considered safe enough based on thresholds
                return {"is_safe": True, "safety_ratings": [r.to_dict() for r in response.prompt_feedback.safety_ratings if response.prompt_feedback]}

        except BlockedPromptException as e:
            logger.warning(f"Content blocked by LLM during moderation: {e}")
            return {"is_safe": False, "block_reason": "Blocked by LLM safety system", "detail": str(e)}
        except Exception as e:
            logger.exception(f"Error during LLM content moderation for model {model_name}: {e}")
            raise LLMServiceException(detail=f"Failed to moderate content: {e}")
