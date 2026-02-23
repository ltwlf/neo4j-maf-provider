"""
Tests for result formatting functions.

Tests _format_cypher_result, _format_retriever_result, and _format_field
from the provider module.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

from agent_framework_neo4j import Neo4jContextProvider
from agent_framework_neo4j._provider import _format_cypher_result


def _make_record(data: dict[str, Any]) -> MagicMock:
    """Create a mock neo4j.Record that supports dict() and pop()."""
    record = MagicMock(spec=neo4j.Record)
    # Make dict(record) return a copy of data
    record.__iter__ = MagicMock(return_value=iter(data.keys()))
    record.keys = MagicMock(return_value=list(data.keys()))
    record.values = MagicMock(return_value=list(data.values()))
    record.items = MagicMock(return_value=list(data.items()))

    # neo4j.Record supports dict(record) via __iter__ + __getitem__
    # Simpler: just make it return a dict directly
    def _as_dict() -> dict[str, Any]:
        return dict(data)

    # Patch __iter__ so dict(record) works
    record.__iter__ = lambda self: iter(data)
    record.__getitem__ = lambda self, key: data[key]
    return record


def _make_provider() -> Neo4jContextProvider:
    """Create a minimal provider for testing instance methods."""
    return Neo4jContextProvider(index_name="test_index", index_type="fulltext")


# ---------------------------------------------------------------------------
# _format_cypher_result (module-level function)
# ---------------------------------------------------------------------------

class TestFormatCypherResult:
    """Test the module-level _format_cypher_result function."""

    def test_extracts_text_as_content(self) -> None:
        record = _make_record({"text": "Engine report", "score": 0.95, "company": "Acme"})
        item = _format_cypher_result(record)
        assert item.content == "Engine report"
        assert item.metadata is not None
        assert item.metadata["score"] == 0.95
        assert item.metadata["company"] == "Acme"
        assert "text" not in item.metadata

    def test_fallback_to_first_string_field(self) -> None:
        record = _make_record({"score": 0.9, "title": "My Document", "count": 42})
        item = _format_cypher_result(record)
        assert item.content == "My Document"

    def test_fallback_to_str_record(self) -> None:
        record = _make_record({"score": 0.9, "count": 42})
        item = _format_cypher_result(record)
        # Should fall back to str(record) since no string fields
        assert isinstance(item.content, str)

    def test_empty_metadata_is_none(self) -> None:
        record = _make_record({"text": "Only text"})
        item = _format_cypher_result(record)
        assert item.content == "Only text"
        # After popping 'text', remaining dict is empty → metadata should be None-ish or empty
        assert item.metadata is None or item.metadata == {}


# ---------------------------------------------------------------------------
# _format_retriever_result (instance method)
# ---------------------------------------------------------------------------

class TestFormatRetrieverResult:
    """Test the _format_retriever_result instance method."""

    def test_formats_items_with_score(self) -> None:
        provider = _make_provider()
        result = RetrieverResult(items=[
            RetrieverResultItem(content="Result one", metadata={"score": 0.95}),
        ])
        formatted = provider._format_retriever_result(result)
        assert len(formatted) == 1
        assert "[Score: 0.950]" in formatted[0]
        assert "Result one" in formatted[0]

    def test_formats_multiple_items(self) -> None:
        provider = _make_provider()
        result = RetrieverResult(items=[
            RetrieverResultItem(content="First", metadata={"score": 0.9}),
            RetrieverResultItem(content="Second", metadata={"score": 0.8}),
        ])
        formatted = provider._format_retriever_result(result)
        assert len(formatted) == 2

    def test_includes_metadata_fields(self) -> None:
        provider = _make_provider()
        result = RetrieverResult(items=[
            RetrieverResultItem(
                content="Report text",
                metadata={"score": 0.9, "company": "Acme", "source": "sec_filing"},
            ),
        ])
        formatted = provider._format_retriever_result(result)
        assert "[company: Acme]" in formatted[0]
        assert "[source: sec_filing]" in formatted[0]

    def test_skips_none_metadata_values(self) -> None:
        provider = _make_provider()
        result = RetrieverResult(items=[
            RetrieverResultItem(
                content="Text", metadata={"score": 0.9, "optional_field": None},
            ),
        ])
        formatted = provider._format_retriever_result(result)
        assert "optional_field" not in formatted[0]

    def test_handles_none_score(self) -> None:
        provider = _make_provider()
        result = RetrieverResult(items=[
            RetrieverResultItem(content="Text", metadata={"score": None}),
        ])
        formatted = provider._format_retriever_result(result)
        assert "Score" not in formatted[0]
        assert "Text" in formatted[0]

    def test_handles_no_metadata(self) -> None:
        provider = _make_provider()
        result = RetrieverResult(items=[
            RetrieverResultItem(content="Just content", metadata=None),
        ])
        formatted = provider._format_retriever_result(result)
        assert len(formatted) == 1
        assert "Just content" in formatted[0]

    def test_empty_items_returns_empty_list(self) -> None:
        provider = _make_provider()
        result = RetrieverResult(items=[])
        formatted = provider._format_retriever_result(result)
        assert formatted == []


# ---------------------------------------------------------------------------
# _format_field (instance method)
# ---------------------------------------------------------------------------

class TestFormatField:
    """Test the _format_field instance method."""

    def test_string_value(self) -> None:
        provider = _make_provider()
        assert provider._format_field("company", "Acme") == "[company: Acme]"

    def test_numeric_value(self) -> None:
        provider = _make_provider()
        assert provider._format_field("count", 42) == "[count: 42]"

    def test_list_value(self) -> None:
        provider = _make_provider()
        result = provider._format_field("risks", ["market", "credit", "operational"])
        assert result == "[risks: market, credit, operational]"

    def test_empty_list(self) -> None:
        provider = _make_provider()
        result = provider._format_field("risks", [])
        assert result == ""

    def test_boolean_value(self) -> None:
        provider = _make_provider()
        result = provider._format_field("active", True)
        assert result == "[active: True]"

    def test_float_value(self) -> None:
        provider = _make_provider()
        result = provider._format_field("weight", 3.14)
        assert result == "[weight: 3.14]"
