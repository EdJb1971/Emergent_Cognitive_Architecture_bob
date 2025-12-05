import logging
import json
from typing import Optional, List

from src.core.exceptions import LLMServiceException, APIException
from src.services.llm_integration_service import LLMIntegrationService
from src.models.multimodal_models import VisualAnalysis
from src.core.config import settings

logger = logging.getLogger(__name__)

class VisualInputProcessor:
    """
    A backend service responsible for receiving raw visual input (base64 encoded image),
    performing initial processing (e.g., object detection, scene analysis, OCR),
    and generating structured textual descriptions or features that the Perception Agent can consume.
    Leverages LLMs for analysis.
    """
    MODEL_NAME = settings.LLM_MODEL_NAME # Using a multimodal model like gemini-2.0-flash-exp

    def __init__(self, llm_service: LLMIntegrationService):
        self.llm_service = llm_service
        logger.info("VisualInputProcessor initialized.")

    async def process_visual(self, image_base64: str, image_mime_type: Optional[str] = "image/jpeg") -> VisualAnalysis:
        """
        Processes raw visual data (base64 encoded image) and returns structured analysis.

        Args:
            image_base64 (str): Base64 encoded image data.
            image_mime_type (Optional[str]): The MIME type of the image (e.g., 'image/png').

        Returns:
            VisualAnalysis: Structured output containing visual analysis.

        Raises:
            APIException: If there's an error during processing.
        """
        if not image_base64:
            raise APIException(detail="Image data cannot be empty for visual processing.", status_code=400)

        prompt = f"""
        Analyze the provided image and generate a detailed textual description. Identify key objects, describe the overall scene, and extract any visible text using OCR.
        Provide your analysis in a JSON object with the following structure:
        {{
            "description": "A detailed textual description of the image content.",
            "objects_detected": ["object1", "object2", ...],
            "scene_description": "Description of the overall scene or environment.",
            "ocr_text": "Any text extracted from the image, or null if none."
        }}
        Ensure the output is a valid JSON string.
        """

        try:
            llm_response_str = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.2, # Lower temperature for factual description
                max_output_tokens=1000,
                image_base64=image_base64,
                image_mime_type=image_mime_type # SEC-001: Pass mime type
            )
            
            analysis_data = json.loads(llm_response_str)
            visual_analysis = VisualAnalysis(**analysis_data)

            # SEC-002: Implement additional moderation step for the output of the LLM
            # Moderating textual fields of VisualAnalysis
            text_to_moderate: List[str] = []
            if visual_analysis.description:
                text_to_moderate.append(visual_analysis.description)
            if visual_analysis.scene_description:
                text_to_moderate.append(visual_analysis.scene_description)
            if visual_analysis.ocr_text:
                text_to_moderate.append(visual_analysis.ocr_text)
            
            # Join all textual parts for a single moderation call, or iterate if granular feedback is needed
            combined_text_for_moderation = " ".join(text_to_moderate)

            if combined_text_for_moderation:
                moderation_result = await self.llm_service.moderate_content(text=combined_text_for_moderation)
                if not moderation_result.get("is_safe"):
                    logger.warning(f"VisualInputProcessor output blocked due to safety concerns: {moderation_result.get('block_reason', 'N/A')}")
                    raise APIException(
                        detail=f"Visual analysis output contains harmful content: {moderation_result.get('block_reason', 'N/A')}",
                        status_code=400
                    )

            logger.info(f"VisualInputProcessor successfully processed image. Objects: {visual_analysis.objects_detected[:3]}")
            return visual_analysis
        except LLMServiceException as e:
            logger.error(f"VisualInputProcessor failed to get LLM response: {e.detail}", exc_info=True)
            raise APIException(
                detail=f"LLM interaction failed for visual processing: {e.detail}",
                status_code=e.status_code
            )
        except json.JSONDecodeError as e:
            logger.error(f"VisualInputProcessor failed to parse LLM response as JSON: {e}. Raw response: {llm_response_str[:200]}...", exc_info=True)
            raise APIException(
                detail=f"Failed to parse LLM response for visual analysis. Invalid JSON format. Error: {e}",
                status_code=500
            )
        except Exception as e:
            logger.exception(f"VisualInputProcessor encountered an unexpected error during processing.")
            raise APIException(
                detail=f"An unexpected error occurred during visual processing: {e}",
                status_code=500
            )
