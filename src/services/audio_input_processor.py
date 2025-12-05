import logging
import json
import re
from typing import Optional, List

from src.core.exceptions import LLMServiceException, APIException
from src.services.llm_integration_service import LLMIntegrationService
from src.models.multimodal_models import AudioAnalysis
from src.core.config import settings

logger = logging.getLogger(__name__)

class AudioInputProcessor:
    """
    A backend service responsible for receiving raw audio input (base64 encoded audio),
    performing speech-to-text conversion, and potentially identifying audio events or emotional cues.
    Outputs structured textual data that the Perception Agent can consume. Leverages LLMs for analysis.
    """
    MODEL_NAME = settings.LLM_MODEL_NAME # Using a multimodal model like gemini-2.0-flash-exp

    def __init__(self, llm_service: LLMIntegrationService):
        self.llm_service = llm_service
        logger.info("AudioInputProcessor initialized.")

    async def process_audio(self, audio_base64: str, audio_mime_type: Optional[str] = "audio/wav") -> AudioAnalysis:
        """
        Processes raw audio data (base64 encoded audio) and returns structured analysis.

        Args:
            audio_base64 (str): Base64 encoded audio data.
            audio_mime_type (Optional[str]): The MIME type of the audio (e.g., 'audio/mpeg').

        Returns:
            AudioAnalysis: Structured output containing audio analysis (e.g., transcription).

        Raises:
            APIException: If there's an error during processing.
        """
        if not audio_base64:
            raise APIException(detail="Audio data cannot be empty for audio processing.", status_code=400)

        prompt = f"""
        Transcribe the provided audio into text. Identify the language, estimate the number of speakers, and detect any significant audio events (e.g., laughter, music, silence).
        Provide your analysis in a JSON object with the following structure:
        {{
            "transcription": "The transcribed text from the audio.",
            "language": "Detected language (e.g., 'en', 'es'), or null if not detectable.",
            "speaker_count": "Estimated number of speakers, or null if not detectable.",
            "audio_events": ["event1", "event2", ...], "e.g., 'laughter', 'music', 'background_noise'"
        }}
        Ensure the output is a valid JSON string.
        """

        try:
            llm_response_str = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.1, # Lower temperature for accurate transcription and factual analysis
                max_output_tokens=1000,
                audio_base64=audio_base64,
                audio_mime_type=audio_mime_type # SEC-001: Pass mime type
            )
            
            # Attempt to parse as JSON directly; if it fails, try to salvage from markdown/code fences
            analysis_data = None
            parse_error: Optional[Exception] = None
            try:
                analysis_data = json.loads(llm_response_str)
            except Exception as e:
                parse_error = e
                cleaned = self._strip_markdown_fences(llm_response_str)
                # Try direct parse after stripping
                try:
                    analysis_data = json.loads(cleaned)
                except Exception:
                    # Try to extract the first JSON object heuristically
                    extracted = self._extract_first_json_object(cleaned)
                    if extracted:
                        analysis_data = json.loads(extracted)

            if analysis_data is None:
                # As a last resort, fallback to treating the response as plain transcription text
                logger.error(
                    f"AudioInputProcessor failed to parse LLM response as JSON: {parse_error}. Raw response: {llm_response_str[:200]}...",
                    exc_info=True,
                )
                fallback = {
                    "transcription": self._strip_markdown_fences(llm_response_str).strip().strip('`'),
                    "language": None,
                    "speaker_count": None,
                    "audio_events": [],
                }
                analysis_data = fallback

            audio_analysis = AudioAnalysis(**analysis_data)

            # SEC-003: Implement additional moderation step for the output of the LLM
            # Moderating textual fields of AudioAnalysis
            text_to_moderate: List[str] = []
            if audio_analysis.transcription:
                text_to_moderate.append(audio_analysis.transcription)
            if audio_analysis.language: # Language itself is less likely to be harmful, but including for completeness
                text_to_moderate.append(audio_analysis.language)
            if audio_analysis.audio_events:
                text_to_moderate.extend(audio_analysis.audio_events) # Extend with individual events

            combined_text_for_moderation = " ".join(text_to_moderate)

            if combined_text_for_moderation:
                moderation_result = await self.llm_service.moderate_content(text=combined_text_for_moderation)
                if not moderation_result.get("is_safe"):
                    logger.warning(f"AudioInputProcessor output blocked due to safety concerns: {moderation_result.get('block_reason', 'N/A')}")
                    raise APIException(
                        detail=f"Audio analysis output contains harmful content: {moderation_result.get('block_reason', 'N/A')}",
                        status_code=400
                    )

            logger.info(f"AudioInputProcessor successfully processed audio. Transcription: {audio_analysis.transcription[:50]}...")
            return audio_analysis
        except LLMServiceException as e:
            logger.error(f"AudioInputProcessor failed to get LLM response: {e.detail}", exc_info=True)
            raise APIException(
                detail=f"LLM interaction failed for audio processing: {e.detail}",
                status_code=e.status_code
            )
        except Exception as e:
            logger.exception(f"AudioInputProcessor encountered an unexpected error during processing.")
            raise APIException(
                detail=f"An unexpected error occurred during audio processing: {e}",
                status_code=500
            )

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove common Markdown code fences from the text (e.g., ```json ... ```)."""
        if not text:
            return text
        s = text.strip()
        # Remove starting fence like ```json or ```
        if s.startswith("```"):
            # drop the first line
            nl = s.find("\n")
            if nl != -1:
                s = s[nl+1:]
        # Remove ending fence
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        """Heuristically extract the first JSON object from text, handling nested braces."""
        if not text:
            return None
        # Find the first opening brace
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None
