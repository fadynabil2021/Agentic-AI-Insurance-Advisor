"""
Unit tests for the deterministic routing functions.
These tests require NO LLM and NO external services — pure Python.
Run: pytest tests/test_routing.py -v
"""
import pytest
from agent.routing import route_after_validation, route_after_scoring
from agent.state import AgentState, QueryType, ConfidenceLevel


# ─── route_after_validation ──────────────────────────────────────────────────

class TestRouteAfterValidation:
    def test_unsupported_query_routes_to_unsupported(self):
        state = {
            "error": {"type": "UNSUPPORTED_QUERY", "message": "outside domain"},
            "missing_fields": [],
        }
        assert route_after_validation(state) == "unsupported"

    def test_parse_error_routes_to_unsupported(self):
        state = {
            "error": {"type": "PARSE_ERROR", "message": "failed to parse"},
            "missing_fields": [],
        }
        assert route_after_validation(state) == "unsupported"

    def test_missing_fields_routes_to_incomplete(self):
        state = {
            "error": None,
            "missing_fields": ["industry", "region"],
        }
        assert route_after_validation(state) == "incomplete"

    def test_all_fields_present_routes_to_complete(self):
        state = {
            "error": None,
            "missing_fields": [],
        }
        assert route_after_validation(state) == "complete"

    def test_no_missing_fields_key_routes_to_complete(self):
        """missing_fields=None should be treated as no missing fields."""
        state = {"error": None}
        assert route_after_validation(state) == "complete"

    def test_empty_error_dict_does_not_trigger_unsupported(self):
        """An empty error dict should not match any error type."""
        state = {
            "error": {},
            "missing_fields": [],
        }
        assert route_after_validation(state) == "complete"

    def test_other_error_type_does_not_trigger_unsupported(self):
        """Only UNSUPPORTED_QUERY and PARSE_ERROR should route to unsupported."""
        state = {
            "error": {"type": "EMPTY_RETRIEVAL", "message": "no results"},
            "missing_fields": [],
        }
        assert route_after_validation(state) == "complete"


# ─── route_after_scoring ─────────────────────────────────────────────────────

class TestRouteAfterScoring:
    def test_recommend_on_happy_path(self):
        state = {
            "retry_count": 0,
            "retrieval_results": [{"type": "package", "name": "Standard"}],
            "scoring_results": [{"package": {"name": "Standard"}, "score": 80}],
            "query_type": QueryType.RECOMMEND,
        }
        assert route_after_scoring(state) == "recommend"

    def test_compare_when_query_type_is_compare(self):
        state = {
            "retry_count": 0,
            "retrieval_results": [{"type": "package", "name": "Standard"}],
            "scoring_results": [{"package": {"name": "Standard"}, "score": 80}],
            "query_type": QueryType.COMPARE,
        }
        assert route_after_scoring(state) == "compare"

    def test_retry_when_no_packages_and_under_limit(self):
        state = {
            "retry_count": 0,
            "retrieval_results": [{"type": "rule", "id": "rule_1"}],  # no packages
            "scoring_results": [],
            "query_type": QueryType.RECOMMEND,
        }
        assert route_after_scoring(state) == "retry"

    def test_fallback_when_no_packages_and_at_retry_limit(self):
        state = {
            "retry_count": 2,
            "retrieval_results": [{"type": "rule", "id": "rule_1"}],  # no packages
            "scoring_results": [],
            "query_type": QueryType.RECOMMEND,
        }
        assert route_after_scoring(state) == "fallback"

    def test_fallback_when_scoring_empty_non_explain(self):
        state = {
            "retry_count": 0,
            "retrieval_results": [{"type": "package", "name": "Standard"}],
            "scoring_results": [],
            "query_type": QueryType.RECOMMEND,
        }
        assert route_after_scoring(state) == "fallback"

    def test_explain_query_routes_to_recommend_even_without_scoring(self):
        state = {
            "retry_count": 0,
            "retrieval_results": [{"type": "package", "name": "Standard"}],
            "scoring_results": [],
            "query_type": QueryType.EXPLAIN,
        }
        assert route_after_scoring(state) == "recommend"

    def test_cheapest_follows_recommend_path(self):
        state = {
            "retry_count": 0,
            "retrieval_results": [{"type": "package", "name": "Basic"}],
            "scoring_results": [{"package": {"name": "Basic"}, "score": 90}],
            "query_type": QueryType.CHEAPEST,
        }
        assert route_after_scoring(state) == "recommend"

    def test_empty_retrieval_results_triggers_retry(self):
        state = {
            "retry_count": 0,
            "retrieval_results": [],
            "scoring_results": [],
            "query_type": QueryType.RECOMMEND,
        }
        assert route_after_scoring(state) == "retry"

    def test_none_retrieval_results_triggers_retry(self):
        state = {
            "retry_count": 0,
            "retrieval_results": None,
            "scoring_results": [],
            "query_type": QueryType.RECOMMEND,
        }
        assert route_after_scoring(state) == "retry"
