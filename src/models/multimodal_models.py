from pydantic import BaseModel, Field
from typing import List, Optional

class VisualAnalysis(BaseModel):
    """
    Structured analysis of visual input.
    """
    description: str = Field(..., description="A textual description of the visual content.")
    objects_detected: List[str] = Field(default_factory=list, description="List of objects identified in the image.")
    scene_description: str = Field(..., description="Description of the overall scene or environment.")
    ocr_text: Optional[str] = Field(None, description="Any text extracted from the image via OCR.")

class AudioAnalysis(BaseModel):
    """
    Structured analysis of audio input.
    """
    transcription: str = Field(..., description="Speech-to-text transcription of the audio.")
    language: Optional[str] = Field(None, description="Detected language of the speech.")
    speaker_count: Optional[int] = Field(None, description="Number of speakers detected in the audio.")
    audio_events: List[str] = Field(default_factory=list, description="List of significant audio events or sounds detected (e.g., 'laughter', 'music', 'silence').")
