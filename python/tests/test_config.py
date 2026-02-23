"""
Tests for ProviderConfig.

Tests type-safe accessors, additional validation edge cases,
and get_connection.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from neo4j_graphrag.embeddings import Embedder

from agent_framework_neo4j._config import DEFAULT_CONTEXT_PROMPT, ProviderConfig


def _mock_embedder() -> Embedder:
    """Create a mock embedder that satisfies the Embedder interface."""
    return MagicMock(spec=Embedder)


class TestProviderConfigDefaults:
    """Test default values."""

    def test_default_context_prompt(self) -> None:
        config = ProviderConfig(index_name="idx", index_type="fulltext")
        assert config.context_prompt == DEFAULT_CONTEXT_PROMPT

    def test_default_top_k(self) -> None:
        config = ProviderConfig(index_name="idx", index_type="fulltext")
        assert config.top_k == 5

    def test_default_message_history_count(self) -> None:
        config = ProviderConfig(index_name="idx", index_type="fulltext")
        assert config.message_history_count == 10


class TestProviderConfigValidation:
    """Test validation edge cases."""

    def test_message_history_count_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="message_history_count must be at least 1"):
            ProviderConfig(index_name="idx", index_type="fulltext", message_history_count=0)

    def test_message_history_count_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="message_history_count must be at least 1"):
            ProviderConfig(index_name="idx", index_type="fulltext", message_history_count=-1)

    def test_vector_requires_embedder(self) -> None:
        with pytest.raises(ValueError, match="embedder is required"):
            ProviderConfig(index_name="idx", index_type="vector")

    def test_hybrid_requires_embedder(self) -> None:
        with pytest.raises(ValueError, match="embedder is required"):
            ProviderConfig(
                index_name="idx",
                index_type="hybrid",
                fulltext_index_name="ft_idx",
            )

    def test_hybrid_requires_fulltext_index(self) -> None:
        with pytest.raises(ValueError, match="fulltext_index_name is required"):
            ProviderConfig(
                index_name="idx",
                index_type="hybrid",
                embedder=_mock_embedder(),
            )

    def test_valid_hybrid_config(self) -> None:
        config = ProviderConfig(
            index_name="vec_idx",
            index_type="hybrid",
            fulltext_index_name="ft_idx",
            embedder=_mock_embedder(),
        )
        assert config.index_type == "hybrid"

    def test_fulltext_does_not_require_embedder(self) -> None:
        config = ProviderConfig(index_name="idx", index_type="fulltext")
        assert config.embedder is None


class TestGetConnection:
    """Test get_connection accessor."""

    def test_returns_tuple_when_all_set(self) -> None:
        config = ProviderConfig(
            index_name="idx",
            index_type="fulltext",
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )
        uri, username, password = config.get_connection()
        assert uri == "bolt://localhost:7687"
        assert username == "neo4j"
        assert password == "password"

    def test_raises_when_uri_missing(self) -> None:
        config = ProviderConfig(
            index_name="idx",
            index_type="fulltext",
            username="neo4j",
            password="password",
        )
        with pytest.raises(ValueError, match="Neo4j connection requires"):
            config.get_connection()

    def test_raises_when_username_missing(self) -> None:
        config = ProviderConfig(
            index_name="idx",
            index_type="fulltext",
            uri="bolt://localhost:7687",
            password="password",
        )
        with pytest.raises(ValueError, match="Neo4j connection requires"):
            config.get_connection()

    def test_raises_when_password_missing(self) -> None:
        config = ProviderConfig(
            index_name="idx",
            index_type="fulltext",
            uri="bolt://localhost:7687",
            username="neo4j",
        )
        with pytest.raises(ValueError, match="Neo4j connection requires"):
            config.get_connection()

    def test_raises_when_all_missing(self) -> None:
        config = ProviderConfig(index_name="idx", index_type="fulltext")
        with pytest.raises(ValueError, match="Neo4j connection requires"):
            config.get_connection()


class TestGetRetrievalQuery:
    """Test get_retrieval_query accessor."""

    def test_returns_query_when_set(self) -> None:
        config = ProviderConfig(
            index_name="idx",
            index_type="fulltext",
            retrieval_query="RETURN node.text AS text, score",
        )
        assert config.get_retrieval_query() == "RETURN node.text AS text, score"

    def test_raises_when_not_set(self) -> None:
        config = ProviderConfig(index_name="idx", index_type="fulltext")
        with pytest.raises(ValueError, match="retrieval_query not set"):
            config.get_retrieval_query()


class TestGetFulltextIndexName:
    """Test get_fulltext_index_name accessor."""

    def test_returns_name_when_set(self) -> None:
        config = ProviderConfig(
            index_name="idx",
            index_type="fulltext",
            fulltext_index_name="ft_idx",
        )
        assert config.get_fulltext_index_name() == "ft_idx"

    def test_raises_when_not_set(self) -> None:
        config = ProviderConfig(index_name="idx", index_type="fulltext")
        with pytest.raises(ValueError, match="fulltext_index_name not set"):
            config.get_fulltext_index_name()


class TestGetEmbedder:
    """Test get_embedder accessor."""

    def test_returns_embedder_when_set(self) -> None:
        embedder = _mock_embedder()
        config = ProviderConfig(
            index_name="idx",
            index_type="vector",
            embedder=embedder,
        )
        assert config.get_embedder() is embedder

    def test_raises_when_not_set(self) -> None:
        config = ProviderConfig(index_name="idx", index_type="fulltext")
        with pytest.raises(ValueError, match="embedder not set"):
            config.get_embedder()
