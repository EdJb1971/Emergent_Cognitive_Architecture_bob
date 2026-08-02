from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ConfigurationError
from src.providers.base import ProviderCapabilities
from src.providers.factory import (
    build_audio_provider,
    build_synthesis_provider,
    build_visual_provider,
    enforce_local_only,
)


def _provider(is_local: bool, name: str = "p"):
    provider = MagicMock()
    provider.capabilities = ProviderCapabilities(provider=name, model="m", is_local=is_local)
    return provider


def test_synthesis_defaults_to_the_agent_provider():
    generation = _provider(is_local=True, name="ollama")
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.SYNTHESIS_PROVIDER = ""
        assert build_synthesis_provider(MagicMock(), generation) is generation


def test_synthesis_can_be_moved_to_gemini_independently():
    with patch("src.providers.factory.settings") as mock_settings, \
         patch("src.providers.factory.GeminiProvider") as gemini:
        mock_settings.SYNTHESIS_PROVIDER = "gemini"
        mock_settings.LLM_MODEL_FOR_RESPONSE_GENERATION = "models/gemini-3.5-flash-lite"
        build_synthesis_provider(MagicMock(), _provider(is_local=True))
    gemini.assert_called_once_with(model="models/gemini-3.5-flash-lite")


def test_unknown_synthesis_provider_fails_fast():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.SYNTHESIS_PROVIDER = "wishful"
        with pytest.raises(ConfigurationError):
            build_synthesis_provider(MagicMock(), _provider(is_local=True))


def test_local_only_mode_rejects_a_cloud_role():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.LOCAL_ONLY_MODE = True
        with pytest.raises(ConfigurationError, match="synthesis=gemini"):
            enforce_local_only(
                cognitive=_provider(is_local=True, name="ollama"),
                synthesis=_provider(is_local=False, name="gemini"),
            )


def test_local_only_mode_accepts_an_all_local_configuration():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.LOCAL_ONLY_MODE = True
        enforce_local_only(
            cognitive=_provider(is_local=True, name="ollama"),
            synthesis=_provider(is_local=True, name="ollama"),
        )


def test_cloud_roles_are_allowed_when_local_only_is_off():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.LOCAL_ONLY_MODE = False
        enforce_local_only(synthesis=_provider(is_local=False, name="gemini"))


def test_visual_provider_is_local_and_capability_gated():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.VISUAL_INPUT_ENABLED = True
        mock_settings.VISUAL_PROVIDER = "ollama"
        mock_settings.OLLAMA_VISION_MODEL = "gemma4:e4b"
        mock_settings.OLLAMA_CHAT_MODEL = "text-model"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_NUM_CTX = 8192
        provider = build_visual_provider(MagicMock(), capability_verified=True)

    assert provider.capabilities.provider == "ollama"
    assert provider.capabilities.is_local is True
    assert provider.capabilities.supports_images is True
    assert provider.capabilities.model == "gemma4:e4b"


def test_visual_provider_rejects_cloud_configuration():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.VISUAL_INPUT_ENABLED = True
        mock_settings.VISUAL_PROVIDER = "gemini"
        with pytest.raises(ConfigurationError, match="cloud visual fallback is prohibited"):
            build_visual_provider(MagicMock(), capability_verified=True)


def test_unconfigured_optional_visual_role_degrades_without_breaking_startup():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.VISUAL_INPUT_ENABLED = True
        mock_settings.VISUAL_PROVIDER = "ollama"
        mock_settings.OLLAMA_VISION_MODEL = ""
        mock_settings.OLLAMA_CHAT_MODEL = ""
        assert build_visual_provider(MagicMock(), capability_verified=False) is None


def test_audio_provider_is_local_and_capability_gated():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.AUDIO_INPUT_ENABLED = True
        mock_settings.AUDIO_PROVIDER = "ollama"
        mock_settings.OLLAMA_AUDIO_MODEL = "gemma4:e4b"
        mock_settings.OLLAMA_CHAT_MODEL = "text-model"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_NUM_CTX = 32768
        provider = build_audio_provider(MagicMock(), capability_verified=True)

    assert provider.capabilities.provider == "ollama"
    assert provider.capabilities.is_local is True
    assert provider.capabilities.supports_audio is True
    assert provider.capabilities.model == "gemma4:e4b"
    assert provider.num_ctx == 8192


def test_audio_provider_rejects_cloud_configuration():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.AUDIO_INPUT_ENABLED = True
        mock_settings.AUDIO_PROVIDER = "gemini"
        with pytest.raises(ConfigurationError, match="cloud audio fallback is prohibited"):
            build_audio_provider(MagicMock(), capability_verified=True)


def test_unconfigured_optional_audio_role_degrades_without_breaking_startup():
    with patch("src.providers.factory.settings") as mock_settings:
        mock_settings.AUDIO_INPUT_ENABLED = True
        mock_settings.AUDIO_PROVIDER = "ollama"
        mock_settings.OLLAMA_AUDIO_MODEL = ""
        mock_settings.OLLAMA_CHAT_MODEL = ""
        assert build_audio_provider(MagicMock(), capability_verified=False) is None
