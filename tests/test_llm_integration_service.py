import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.services.llm_integration_service import LLMIntegrationService
from src.core.exceptions import LLMServiceException, ConfigurationError
from src.core.config import settings
from google.generativeai.types import BlockedPromptException, HarmCategory, HarmBlockThreshold

@pytest.fixture(autouse=True)
def mock_settings():
    with patch('src.core.config.settings') as mock_config:
        mock_config.GEMINI_API_KEY = "test_api_key"
        mock_config.LLM_MODEL_NAME = "test-model"
        mock_config.EMBEDDING_MODEL_NAME = "test-embedding-model"
        mock_config.LLM_MODEL_FOR_MODERATION = "test-moderation-model"
        yield mock_config

@pytest.fixture
def llm_service():
    return LLMIntegrationService()

@pytest.mark.asyncio
async def test_llm_service_init_no_api_key():
    with patch('src.core.config.settings.GEMINI_API_KEY', ""):
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is not set"):
            LLMIntegrationService()

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_generate_text_success(mock_generative_model, llm_service):
    mock_response = AsyncMock()
    mock_response.candidates = [AsyncMock()]
    mock_response.candidates[0].content.parts = [AsyncMock()]
    mock_response.candidates[0].content.parts[0].text = "Generated response"
    mock_generative_model.return_value.generate_content_async.return_value = mock_response

    prompt = "Hello, world!"
    result = await llm_service.generate_text(prompt)
    assert result == "Generated response"
    mock_generative_model.assert_called_once_with(settings.LLM_MODEL_NAME)
    mock_generative_model.return_value.generate_content_async.assert_called_once()

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_generate_text_blocked_prompt(mock_generative_model, llm_service):
    mock_response = AsyncMock()
    mock_response.candidates = []
    mock_response.prompt_feedback = AsyncMock()
    mock_response.prompt_feedback.block_reason = AsyncMock(name="SAFETY")
    mock_response.prompt_feedback.safety_ratings = []
    mock_generative_model.return_value.generate_content_async.return_value = mock_response

    prompt = "Harmful content"
    with pytest.raises(LLMServiceException, match="Prompt blocked due to safety concerns") as exc_info:
        await llm_service.generate_text(prompt)
    assert exc_info.value.status_code == 400

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_generate_text_no_candidates(mock_generative_model, llm_service):
    mock_response = AsyncMock()
    mock_response.candidates = []
    mock_response.prompt_feedback = None
    mock_generative_model.return_value.generate_content_async.return_value = mock_response

    prompt = "Empty response"
    with pytest.raises(LLMServiceException, match="No content generated."):
        await llm_service.generate_text(prompt)

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_generate_text_api_error(mock_generative_model, llm_service):
    mock_generative_model.return_value.generate_content_async.side_effect = Exception("API error")

    prompt = "Error test"
    with pytest.raises(LLMServiceException, match="Failed to generate text"):
        await llm_service.generate_text(prompt)

@pytest.mark.asyncio
@patch('google.generativeai.embed_content_async')
async def test_generate_embedding_success(mock_embed_content_async, llm_service):
    mock_response = AsyncMock()
    mock_response.embedding = [0.1, 0.2, 0.3]
    mock_embed_content_async.return_value = mock_response

    text = "Text to embed"
    result = await llm_service.generate_embedding(text)
    assert result == [0.1, 0.2, 0.3]
    mock_embed_content_async.assert_called_once_with(model=settings.EMBEDDING_MODEL_NAME, content=text)

@pytest.mark.asyncio
@patch('google.generativeai.embed_content_async')
async def test_generate_embedding_empty_text(mock_embed_content_async, llm_service):
    with pytest.raises(LLMServiceException, match="Text cannot be empty for embedding generation."):
        await llm_service.generate_embedding("")

@pytest.mark.asyncio
@patch('google.generativeai.embed_content_async')
async def test_generate_embedding_api_error(mock_embed_content_async, llm_service):
    mock_embed_content_async.side_effect = Exception("Embedding API error")

    text = "Error embedding"
    with pytest.raises(LLMServiceException, match="Failed to generate embedding"):
        await llm_service.generate_embedding(text)

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_moderate_content_safe(mock_generative_model, llm_service):
    mock_response = AsyncMock()
    mock_response.prompt_feedback = AsyncMock()
    mock_response.prompt_feedback.block_reason = None
    mock_response.prompt_feedback.safety_ratings = []
    mock_generative_model.return_value.generate_content_async.return_value = mock_response

    text = "Safe content"
    result = await llm_service.moderate_content(text)
    assert result["is_safe"] is True

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_moderate_content_unsafe(mock_generative_model, llm_service):
    mock_response = AsyncMock()
    mock_response.prompt_feedback = AsyncMock()
    mock_response.prompt_feedback.block_reason = AsyncMock(name="HARASSMENT")
    mock_response.prompt_feedback.safety_ratings = [AsyncMock(category=HarmCategory.HARM_CATEGORY_HARASSMENT, probability=4, is_blocked=True, to_dict=lambda: {'category': 'HARASSMENT', 'probability': 'HIGH', 'is_blocked': True})]
    mock_generative_model.return_value.generate_content_async.return_value = mock_response

    text = "Unsafe content"
    result = await llm_service.moderate_content(text)
    assert result["is_safe"] is False
    assert result["block_reason"] == "HARASSMENT"

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_moderate_content_blocked_prompt_exception(mock_generative_model, llm_service):
    mock_generative_model.return_value.generate_content_async.side_effect = BlockedPromptException("Blocked by LLM")

    text = "Blocked content"
    result = await llm_service.moderate_content(text)
    assert result["is_safe"] is False
    assert result["block_reason"] == "Blocked by LLM safety system"

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_moderate_content_api_error(mock_generative_model, llm_service):
    mock_generative_model.return_value.generate_content_async.side_effect = Exception("Moderation API error")

    text = "Error moderation"
    with pytest.raises(LLMServiceException, match="Failed to moderate content"):
        await llm_service.moderate_content(text)

@pytest.mark.asyncio
async def test_generate_text_empty_prompt(llm_service):
    with pytest.raises(LLMServiceException, match="Prompt cannot be empty"):
        await llm_service.generate_text("")

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_generate_text_retry_mechanism(mock_generative_model, llm_service):
    mock_generative_model.return_value.generate_content_async.side_effect = [
        LLMServiceException(detail="Transient error"),
        LLMServiceException(detail="Another transient error"),
        AsyncMock(candidates=[AsyncMock(content=AsyncMock(parts=[AsyncMock(text="Success after retry")])])])
    ]

    prompt = "Retry test"
    result = await llm_service.generate_text(prompt)
    assert result == "Success after retry"
    assert mock_generative_model.return_value.generate_content_async.call_count == 3

@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
async def test_generate_text_retry_failure(mock_generative_model, llm_service):
    mock_generative_model.return_value.generate_content_async.side_effect = LLMServiceException(detail="Persistent error")

    prompt = "Retry failure test"
    with pytest.raises(LLMServiceException, match="Persistent error"):
        await llm_service.generate_text(prompt)
    assert mock_generative_model.return_value.generate_content_async.call_count == 3
