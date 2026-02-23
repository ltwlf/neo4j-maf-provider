"""
Tests for AzureAISettings.

Tests environment variable loading, computed inference endpoint,
and is_configured property.
"""

from __future__ import annotations

import pytest

from agent_framework_neo4j import AzureAISettings


class TestAzureAISettings:
    """Test AzureAISettings environment variable loading."""

    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should load from AZURE_AI_* environment variables."""
        monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://myproject.ai.azure.com/api/projects/123")
        monkeypatch.setenv("AZURE_AI_EMBEDDING_NAME", "text-embedding-3-large")

        settings = AzureAISettings()
        assert settings.project_endpoint == "https://myproject.ai.azure.com/api/projects/123"
        assert settings.embedding_model == "text-embedding-3-large"

    def test_default_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Embedding model should default to text-embedding-ada-002."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_AI_EMBEDDING_NAME", raising=False)

        settings = AzureAISettings()
        assert settings.embedding_model == "text-embedding-ada-002"

    def test_project_endpoint_defaults_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Project endpoint should default to None when not set."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)

        settings = AzureAISettings()
        assert settings.project_endpoint is None


class TestInferenceEndpoint:
    """Test the computed inference_endpoint property."""

    def test_derives_models_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should convert project endpoint to models endpoint."""
        monkeypatch.setenv(
            "AZURE_AI_PROJECT_ENDPOINT",
            "https://myhost.ai.azure.com/api/projects/my-project-id",
        )
        settings = AzureAISettings()
        assert settings.inference_endpoint == "https://myhost.ai.azure.com/models"

    def test_passthrough_when_no_projects_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should pass through endpoint when it doesn't contain /api/projects/."""
        monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://custom-endpoint.example.com")
        settings = AzureAISettings()
        assert settings.inference_endpoint == "https://custom-endpoint.example.com"

    def test_none_when_no_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return None when project_endpoint is not set."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        settings = AzureAISettings()
        assert settings.inference_endpoint is None


class TestIsConfigured:
    """Test the is_configured property."""

    def test_configured_when_endpoint_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://myhost.ai.azure.com/api/projects/123")
        settings = AzureAISettings()
        assert settings.is_configured is True

    def test_not_configured_when_no_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        settings = AzureAISettings()
        assert settings.is_configured is False
