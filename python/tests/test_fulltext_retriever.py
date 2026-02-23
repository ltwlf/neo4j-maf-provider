"""
Tests for FulltextRetriever.

Tests keyword extraction, validation models, search result construction,
and record formatting.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j_graphrag.exceptions import RetrieverInitializationError
from neo4j_graphrag.types import RawSearchResult

from agent_framework_neo4j._fulltext import (
    FulltextRetriever,
    FulltextRetrieverModel,
    FulltextSearchModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_driver() -> MagicMock:
    """Create a mock Neo4j driver that passes Neo4jDriverModel validation."""
    driver = MagicMock(spec=neo4j.Driver)
    # Neo4jDriverModel validates that driver has execute_query
    driver.execute_query = MagicMock()
    return driver


def _make_record(data: dict[str, Any]) -> MagicMock:
    """Create a mock neo4j.Record with get() support."""
    record = MagicMock(spec=neo4j.Record)
    record.get = lambda key, default=None: data.get(key, default)
    return record


# ---------------------------------------------------------------------------
# FulltextRetrieverModel validation
# ---------------------------------------------------------------------------

class TestFulltextRetrieverModel:
    """Test Pydantic validation model for FulltextRetriever config."""

    def test_valid_config(self) -> None:
        driver = _make_mock_driver()
        from neo4j_graphrag.types import Neo4jDriverModel
        model = FulltextRetrieverModel(
            driver_model=Neo4jDriverModel(driver=driver),
            index_name="my_index",
        )
        assert model.index_name == "my_index"
        assert model.filter_stop_words is True
        assert model.retrieval_query is None

    def test_empty_index_name_rejected(self) -> None:
        driver = _make_mock_driver()
        from neo4j_graphrag.types import Neo4jDriverModel
        with pytest.raises(ValueError, match="index_name cannot be empty"):
            FulltextRetrieverModel(
                driver_model=Neo4jDriverModel(driver=driver),
                index_name="",
            )

    def test_whitespace_index_name_rejected(self) -> None:
        driver = _make_mock_driver()
        from neo4j_graphrag.types import Neo4jDriverModel
        with pytest.raises(ValueError, match="index_name cannot be empty"):
            FulltextRetrieverModel(
                driver_model=Neo4jDriverModel(driver=driver),
                index_name="   ",
            )


# ---------------------------------------------------------------------------
# FulltextSearchModel validation
# ---------------------------------------------------------------------------

class TestFulltextSearchModel:
    """Test Pydantic validation model for search parameters."""

    def test_valid_search(self) -> None:
        model = FulltextSearchModel(query_text="engine vibration", top_k=10)
        assert model.query_text == "engine vibration"
        assert model.top_k == 10

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            FulltextSearchModel(query_text="")

    def test_whitespace_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            FulltextSearchModel(query_text="   ")

    def test_top_k_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            FulltextSearchModel(query_text="test", top_k=0)

    def test_top_k_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            FulltextSearchModel(query_text="test", top_k=-1)

    def test_default_top_k(self) -> None:
        model = FulltextSearchModel(query_text="test")
        assert model.top_k == 5


# ---------------------------------------------------------------------------
# FulltextRetriever initialisation
# ---------------------------------------------------------------------------

class TestFulltextRetrieverInit:
    """Test FulltextRetriever constructor and validation."""

    def test_valid_init(self) -> None:
        driver = _make_mock_driver()
        retriever = FulltextRetriever(driver, index_name="search_idx")
        assert retriever.index_name == "search_idx"
        assert retriever.filter_stop_words is True
        assert retriever.retrieval_query is None

    def test_custom_retrieval_query(self) -> None:
        driver = _make_mock_driver()
        query = "RETURN node.text AS text, score"
        retriever = FulltextRetriever(driver, index_name="idx", retrieval_query=query)
        assert retriever.retrieval_query == query

    def test_empty_index_name_raises(self) -> None:
        driver = _make_mock_driver()
        with pytest.raises(RetrieverInitializationError):
            FulltextRetriever(driver, index_name="")

    def test_disable_stop_words(self) -> None:
        driver = _make_mock_driver()
        retriever = FulltextRetriever(driver, index_name="idx", filter_stop_words=False)
        assert retriever.filter_stop_words is False


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    """Test stop-word filtering in _extract_keywords."""

    def _retriever(self) -> FulltextRetriever:
        return FulltextRetriever(_make_mock_driver(), index_name="idx")

    def test_removes_stop_words(self) -> None:
        r = self._retriever()
        result = r._extract_keywords("What maintenance issues involve engine vibration?")
        assert "maintenance" in result
        assert "engine" in result
        assert "vibration" in result
        assert "what" not in result
        assert "involve" not in result

    def test_preserves_domain_terms(self) -> None:
        r = self._retriever()
        result = r._extract_keywords("turbine blade inspection report")
        assert "turbine" in result
        assert "blade" in result
        assert "inspection" in result
        assert "report" in result

    def test_all_stop_words_returns_empty(self) -> None:
        r = self._retriever()
        result = r._extract_keywords("what is the")
        assert result == ""

    def test_single_char_words_removed(self) -> None:
        r = self._retriever()
        # single-character words (len <= 1) should be stripped
        result = r._extract_keywords("a b c engine")
        assert "engine" in result
        # 'a' is a stop word and single char, 'b' and 'c' are single char
        assert "b" not in result
        assert "c" not in result

    def test_case_insensitive(self) -> None:
        r = self._retriever()
        result = r._extract_keywords("WHAT Engine VIBRATION")
        assert "engine" in result
        assert "vibration" in result
        assert "what" not in result

    def test_empty_input(self) -> None:
        r = self._retriever()
        assert r._extract_keywords("") == ""

    def test_punctuation_handling(self) -> None:
        r = self._retriever()
        result = r._extract_keywords("engine-vibration fault!")
        assert "engine" in result
        assert "vibration" in result
        assert "fault" in result


# ---------------------------------------------------------------------------
# default_record_formatter
# ---------------------------------------------------------------------------

class TestDefaultRecordFormatter:
    """Test the default record → RetrieverResultItem formatter."""

    def _retriever(self) -> FulltextRetriever:
        return FulltextRetriever(_make_mock_driver(), index_name="idx")

    def test_text_field(self) -> None:
        r = self._retriever()
        record = _make_record({"text": "Engine report content", "score": 0.95})
        item = r.default_record_formatter(record)
        assert item.content == "Engine report content"
        assert item.metadata["score"] == 0.95

    def test_content_field_fallback(self) -> None:
        r = self._retriever()
        record = _make_record({"content": "Fallback content", "score": 0.8})
        item = r.default_record_formatter(record)
        assert item.content == "Fallback content"

    def test_node_field_fallback(self) -> None:
        r = self._retriever()
        record = _make_record({"node": {"id": 123}, "score": 0.7})
        item = r.default_record_formatter(record)
        assert "123" in item.content


# ---------------------------------------------------------------------------
# get_search_results
# ---------------------------------------------------------------------------

class TestGetSearchResults:
    """Test fulltext search execution with mocked Neo4j driver."""

    def test_basic_search(self) -> None:
        driver = _make_mock_driver()
        mock_records = [_make_record({"node": "n1", "score": 0.9})]
        driver.execute_query.return_value = (mock_records, None, None)

        retriever = FulltextRetriever(driver, index_name="search_idx", filter_stop_words=False)
        result = retriever.get_search_results(query_text="engine vibration", top_k=3)

        assert isinstance(result, RawSearchResult)
        assert len(result.records) == 1
        # Verify query parameters
        call_args = driver.execute_query.call_args
        params = call_args[0][1]
        assert params["index_name"] == "search_idx"
        assert params["query"] == "engine vibration"
        assert params["top_k"] == 3

    def test_stop_word_filtering_applied(self) -> None:
        driver = _make_mock_driver()
        driver.execute_query.return_value = ([], None, None)

        retriever = FulltextRetriever(driver, index_name="idx", filter_stop_words=True)
        retriever.get_search_results(query_text="What is the engine status?", top_k=5)

        call_args = driver.execute_query.call_args
        params = call_args[0][1]
        # Stop words removed, only "engine" and "status" remain
        assert "what" not in params["query"]
        assert "engine" in params["query"]
        assert "status" in params["query"]

    def test_all_stop_words_returns_empty(self) -> None:
        driver = _make_mock_driver()

        retriever = FulltextRetriever(driver, index_name="idx", filter_stop_words=True)
        result = retriever.get_search_results(query_text="what is the", top_k=5)

        assert result.records == []
        driver.execute_query.assert_not_called()

    def test_retrieval_query_included_in_cypher(self) -> None:
        driver = _make_mock_driver()
        driver.execute_query.return_value = ([], None, None)

        custom_query = "MATCH (node)-[:RELATES]->(other) RETURN node.text AS text, score"
        retriever = FulltextRetriever(
            driver, index_name="idx", retrieval_query=custom_query, filter_stop_words=False,
        )
        retriever.get_search_results(query_text="engine", top_k=5)

        cypher = driver.execute_query.call_args[0][0]
        assert "RELATES" in cypher
        # Retrieval query mode applies LIMIT twice
        assert cypher.count("LIMIT $top_k") == 2

    def test_without_retrieval_query_has_single_limit(self) -> None:
        driver = _make_mock_driver()
        driver.execute_query.return_value = ([], None, None)

        retriever = FulltextRetriever(driver, index_name="idx", filter_stop_words=False)
        retriever.get_search_results(query_text="engine", top_k=5)

        cypher = driver.execute_query.call_args[0][0]
        assert cypher.count("LIMIT $top_k") == 1
        assert "RETURN node, score" in cypher

    def test_custom_query_params_merged(self) -> None:
        driver = _make_mock_driver()
        driver.execute_query.return_value = ([], None, None)

        retriever = FulltextRetriever(driver, index_name="idx", filter_stop_words=False)
        retriever.get_search_results(
            query_text="engine", top_k=5, query_params={"extra_param": "value"},
        )

        params = driver.execute_query.call_args[0][1]
        assert params["extra_param"] == "value"

    def test_custom_params_dont_override_builtins(self) -> None:
        driver = _make_mock_driver()
        driver.execute_query.return_value = ([], None, None)

        retriever = FulltextRetriever(driver, index_name="idx", filter_stop_words=False)
        retriever.get_search_results(
            query_text="engine", top_k=5, query_params={"index_name": "malicious"},
        )

        params = driver.execute_query.call_args[0][1]
        # Built-in params should not be overridden
        assert params["index_name"] == "idx"

    def test_read_routing_used(self) -> None:
        driver = _make_mock_driver()
        driver.execute_query.return_value = ([], None, None)

        retriever = FulltextRetriever(driver, index_name="idx", filter_stop_words=False)
        retriever.get_search_results(query_text="engine", top_k=5)

        call_kwargs = driver.execute_query.call_args[1]
        assert call_kwargs["routing_"] == neo4j.RoutingControl.READ

    def test_metadata_includes_query_text(self) -> None:
        driver = _make_mock_driver()
        driver.execute_query.return_value = ([], None, None)

        retriever = FulltextRetriever(driver, index_name="idx", filter_stop_words=True)
        result = retriever.get_search_results(query_text="What is the engine status?", top_k=5)

        assert result.metadata["query_text"] == "What is the engine status?"
        assert "engine" in result.metadata["search_text"]
