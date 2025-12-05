"""
Token counting utility for memory management and LLM interactions.
Provides accurate token counts for Gemini and fallback counting methods.
"""
import logging
from typing import Dict, Optional, Union
from functools import lru_cache
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings
from src.core.exceptions import APIException

logger = logging.getLogger(__name__)

class TokenCounter:
    """
    Utility class for counting tokens in text, with support for Gemini's tokenizer
    and fallback methods. Includes caching for performance and retry logic for API calls.
    """
    
    def __init__(self):
        """Initialize the token counter with configured API key and fallback settings."""
        self._model = None
        self._use_fallback = False
        self._cache: Dict[str, int] = {}
        try:
            # Prefer GEMINI_API_KEY from settings; fall back to env var if necessary
            api_key = getattr(settings, "GEMINI_API_KEY", None)
            genai.configure(api_key=api_key)
            # Initialize a model for token counting
            self._model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            # Test the tokenizer
            self.count_tokens("test")
            logger.info("TokenCounter initialized with Gemini tokenizer")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini tokenizer, using fallback: {e}")
            self._use_fallback = True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    @lru_cache(maxsize=10000)  # Cache recent results
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using Gemini's tokenizer or fallback method.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Number of tokens in the text
            
        Raises:
            APIException: If token counting fails after retries
        """
        if not text:
            return 0
            
        try:
            if self._use_fallback:
                return self._count_tokens_fallback(text)
                
            # Use Gemini's model.count_tokens() method
            if not self._model:
                return self._count_tokens_fallback(text)
            
            result = self._model.count_tokens(text)
            return result.total_tokens
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}", exc_info=True)
            # Fall back to approximate counting for this call
            return self._count_tokens_fallback(text)

    def _count_tokens_fallback(self, text: str) -> int:
        """
        Fallback token counting method when Gemini's tokenizer is unavailable.
        Uses a conservative approximation based on word and character counts.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        # Conservative estimate: assume 1.3 tokens per word and add
        # extra tokens for punctuation/spacing
        words = text.split()
        char_count = len(text)
        word_count = len(words)
        
        # Base estimate on word count
        estimated_tokens = int(word_count * 1.3)
        
        # Add extra tokens for heavy punctuation/special chars
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        estimated_tokens += int(special_chars * 0.5)
        
        # Ensure we don't underestimate
        return max(estimated_tokens, int(char_count / 4))

    def estimate_tokens_needed(self, text_length: int) -> int:
        """
        Estimate tokens needed for a text of given length without counting.
        Useful for quick capacity planning.
        
        Args:
            text_length: Length of text in characters
            
        Returns:
            int: Estimated token count needed
        """
        # Conservative estimate: assume 4 chars per token on average
        return int(text_length / 3.5) + 1

    async def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """
        Count tokens for multiple texts efficiently.
        
        Args:
            texts: List of texts to count tokens for
            
        Returns:
            list[int]: List of token counts corresponding to input texts
        """
        return [self.count_tokens(text) for text in texts]

    def get_token_budget(self, reserve_ratio: float = 0.2) -> tuple[int, int]:
        """
        Get recommended token budget and reserve for STM based on model limits.
        
        Args:
            reserve_ratio: Ratio of model's max tokens to reserve (default 0.2)
            
        Returns:
            tuple[int, int]: (token_budget, token_reserve)
        """
        if self._use_fallback:
            # Conservative defaults if we can't check model
            return 25_000, 5_000
            
        try:
            # Gemini's limit is 250k
            MODEL_MAX_TOKENS = 250_000
            reserve = int(MODEL_MAX_TOKENS * reserve_ratio)
            budget = int(MODEL_MAX_TOKENS * 0.15)  # 15% for STM by default
            return budget, reserve
        except Exception as e:
            logger.warning(f"Error getting token budget, using defaults: {e}")
            return 25_000, 5_000  # Safe defaults