"""
Token counting utility for memory management and LLM interactions.

Counting is local and provider-neutral. A remote tokenizer would put a network
round-trip in the hot path of every memory operation and tie STM budgets to one
vendor; token budgets are heuristics, so a local estimate is the better trade.
"""
import logging
from typing import Dict, Optional, Union
from functools import lru_cache

from src.core.config import settings
from src.core.exceptions import APIException

logger = logging.getLogger(__name__)

class TokenCounter:
    """
    Utility class for counting tokens in text using a local estimator,
    with caching for performance.
    """
    
    def __init__(self):
        """Initialize the token counter."""
        self._cache: Dict[str, int] = {}
        logger.info("TokenCounter initialized with local estimator.")

    @lru_cache(maxsize=10000)  # Cache recent results
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using a local approximation.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Number of tokens in the text
        """
        if not text:
            return 0
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
        Get recommended token budget and reserve for STM.

        Derived from configuration rather than a hardcoded vendor context limit,
        so the budget follows whichever provider is active.

        Args:
            reserve_ratio: Ratio of the budget to hold back (default 0.2)
            
        Returns:
            tuple[int, int]: (token_budget, token_reserve)
        """
        budget = getattr(settings, "STM_TOKEN_BUDGET", 25_000)
        return budget, int(budget * reserve_ratio)