"""
Unit tests for the deterministic validator node.
Runs without LLM.
"""
import pytest
from agent.nodes.validator import validator_node
from agent.state import QueryType

def test_validator_recommend_all_fields_present():
    state = {
        "query_type": QueryType.RECOMMEND,
        "extracted_entities": {"industry": "healthcare", "region": "riyadh"},
        "execution_trace": [],
    }
    result = validator_node(state, None)
    assert result["requires_clarification"] is False
    assert len(result["missing_fields"]) == 0

def test_validator_recommend_missing_fields():
    state = {
        "query_type": QueryType.RECOMMEND,
        "extracted_entities": {"industry": "healthcare"}, # missing region
        "execution_trace": [],
    }
    result = validator_node(state, None)
    assert result["requires_clarification"] is True
    assert "region" in result["missing_fields"]
    assert "Which region are you based in?" in result["clarification_question"]

def test_validator_unsupported_query():
    state = {
        "query_type": QueryType.UNSUPPORTED,
        "extracted_entities": {},
        "execution_trace": [],
    }
    result = validator_node(state, None)
    assert "error" in result
    assert result["error"]["type"] == "UNSUPPORTED_QUERY"

def test_validator_parse_error_passthrough():
    state = {
        "error": {"type": "PARSE_ERROR"},
        "execution_trace": [],
    }
    result = validator_node(state, None)
    # the error should just be passed through, state should not be altered much
    assert result.get("error") and result["error"]["type"] == "PARSE_ERROR"

def test_validator_explain_no_fields_required():
    state = {
        "query_type": QueryType.EXPLAIN,
        "extracted_entities": {},
        "execution_trace": [],
    }
    result = validator_node(state, None)
    assert result["requires_clarification"] is False
    assert len(result["missing_fields"]) == 0
