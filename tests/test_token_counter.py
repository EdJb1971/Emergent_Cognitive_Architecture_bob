"""Tests for the token counter utility."""
import pytest
from unittest.mock import Mock, patch
import google.generativeai as genai

from src.utils.token_counter import TokenCounter
from src.core.exceptions import APIException

@pytest.fixture
def token_counter():
    """Create a TokenCounter instance for testing."""
    with patch('google.generativeai.configure'):
        counter = TokenCounter()
        return counter

def test_count_tokens_empty():
    """Test counting tokens for empty text."""
    counter = TokenCounter()
    assert counter.count_tokens("") == 0

def test_count_tokens_simple():
    """Test counting tokens for simple text."""
    counter = TokenCounter()
    text = "Hello world"
    # Even with fallback, should be reasonable
    count = counter.count_tokens(text)
    assert count > 0
    assert count <= 5  # Reasonable upper bound for two words

def test_fallback_counting():
    """Test the fallback counting method directly."""
    counter = TokenCounter()
    text = "This is a test of the fallback counter!"
    count = counter._count_tokens_fallback(text)
    # Should count ~8-10 tokens (7 words + punctuation)
    assert 7 <= count <= 12

def test_count_tokens_batch():
    """Test batch token counting."""
    counter = TokenCounter()
    texts = ["Hello world", "This is another test", ""]
    counts = counter.count_tokens_batch(texts)
    assert len(counts) == 3
    assert counts[0] > 0
    assert counts[1] > counts[0]  # Longer text = more tokens
    assert counts[2] == 0  # Empty text

def test_token_budget():
    """Test getting token budget recommendations."""
    counter = TokenCounter()
    budget, reserve = counter.get_token_budget(0.2)
    assert budget > 0
    assert reserve > 0
    assert budget > reserve  # Budget should be larger than reserve
    
@pytest.mark.asyncio
async def test_count_tokens_batch_async():
    """Test async batch token counting."""
    counter = TokenCounter()
    texts = ["Hello", "World", "Test"]
    counts = await counter.count_tokens_batch(texts)
    assert len(counts) == 3
    assert all(count > 0 for count in counts)

def test_estimate_tokens_needed():
    """Test token estimation for capacity planning."""
    counter = TokenCounter()
    # Test various text lengths
    assert counter.estimate_tokens_needed(0) == 1  # Minimum 1 token
    assert counter.estimate_tokens_needed(100) >= 25  # ~3.5 chars per token + safety
    
    # Longer text should need more tokens
    short = counter.estimate_tokens_needed(100)
    long = counter.estimate_tokens_needed(1000)
    assert long > short

def test_caching():
    """Test that token counting results are cached."""
    counter = TokenCounter()
    text = "This is a test of the cache"
    
    # First call should cache
    count1 = counter.count_tokens(text)
    # Second call should hit cache
    count2 = counter.count_tokens(text)
    
    assert count1 == count2

@pytest.mark.asyncio
async def test_retry_behavior():
    """Test retry behavior on API failures."""
    counter = TokenCounter()
    
    # Mock the Gemini API to fail twice then succeed
    fail_count = 0
    def mock_count(*args, **kwargs):
        nonlocal fail_count
        if fail_count < 2:
            fail_count += 1
            raise Exception("API Error")
        return Mock(total_tokens=5)
    
    with patch('google.generativeai.count_tokens', side_effect=mock_count):
        # Should succeed after retries
        count = counter.count_tokens("test retry")
        assert count > 0
        # Should have tried 3 times
        assert fail_count == 2