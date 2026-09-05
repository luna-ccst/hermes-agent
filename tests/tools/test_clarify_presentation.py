"""Exact live clarification lookup for serialized external presentations."""
import pytest
from tools import clarify_gateway as cg


@pytest.mark.parametrize("resolve_first", [False, True])
def test_bound_reply_selects_exact_live_request_not_oldest(resolve_first):
    key = "presentation-core"
    first = cg.register("presentation-core-a", key, "A?", None)
    second = cg.register("presentation-core-b", key, "B?", None)
    try:
        if resolve_first:
            assert cg.resolve_gateway_clarify(first.clarify_id, "A")
        assert cg.attempt_text_response_for_session(
            key, "B", request_id=second.clarify_id) == cg.TEXT_RESOLVED
        assert second.response == "B"
        assert first.response == ("A" if resolve_first else None)
    finally:
        cg.clear_session(key)
