"""Regression coverage for host timeout diagnostics and lean digest budgets."""
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("waited,since,reason", [(600.0, 0.01, "total ceiling"), (180.0, 120.0, "idle timeout")])
def test_forwarder_reports_actual_timeout(monkeypatch, caplog, waited, since, reason):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "budget-report"
    agent._cached_system_prompt = "sys"
    agent._conversation_root_id = lambda: None
    agent._emit_warning = MagicMock()
    agent._touch_activity = MagicMock()
    agent.context_compressor = MagicMock()
    monkeypatch.setattr("agent.portal_tags.get_conversation_context", lambda: object())
    monkeypatch.setattr("agent.conversation_compression.resolve_context_compression_timeouts", lambda: (120, 600))

    def timeout(**kwargs):
        kwargs["on_timeout"](120, waited, since)
        return kwargs["messages"], "sys"

    monkeypatch.setattr("agent.conversation_compression.run_compress_context_with_progress_timeout", timeout)
    messages = [{"role": "user", "content": "retain me"}]
    assert agent._compress_context(messages, "sys")[0] is messages
    warning = agent._emit_warning.call_args.args[0]
    assert reason in warning
    assert f"{waited:.1f}s" in warning
    assert f"{since:.1f}s" in warning
    assert "No messages were dropped" in warning
    assert "no output" not in warning
    cooldown = agent.context_compressor.record_timeout_failure.call_args.args[0]
    assert reason in cooldown
    assert reason in caplog.text


def test_lean_stops_scheduling_after_host_cancellation(monkeypatch):
    from types import SimpleNamespace
    from agent.context_compressor import ContextCompressor
    from agent.conversation_compression import CompressionCommitFence

    compressor = object.__new__(ContextCompressor)
    fence = CompressionCommitFence()
    compressor._compression_cancelled_check = lambda: fence.is_cancelled
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        fence.try_cancel_before_commit()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))])

    monkeypatch.setattr("agent.context_compressor._serialize_turns_for_digest", lambda *args: "x" * 150000)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", provider)
    text = compressor._build_chunk_digests([])
    assert len(calls) == 1
    assert "cancelled" in text
    assert "session_search" in text


def test_lean_budget_exhaustion_keeps_summary_user_text_and_recovery(monkeypatch):
    from types import SimpleNamespace
    from agent.context_compressor import ContextCompressor

    compressor = object.__new__(ContextCompressor)
    remaining = [10.0]
    compressor._session_id = "budget-exhausted"
    compressor._compression_remaining_seconds = lambda: remaining[0]
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        remaining[0] = 0.0
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="first digest"))])

    monkeypatch.setattr("agent.context_compressor._serialize_turns_for_digest", lambda *args: "x" * 150000)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", provider)
    summary = compressor._augment_summary_lean("Valid main summary", [{"role": "user", "content": "Keep this exact requirement."}])
    assert len(calls) == 1
    assert 0 < calls[0]["timeout"] < 10
    assert "budget" in summary
    assert "Valid main summary" in summary
    assert "Keep this exact requirement." in summary
    assert "## Context Recovery" in summary
    assert "session_search" in summary


@pytest.mark.parametrize("budget,expected_calls", [(0.0, 0), (0.5, 0), (None, 1)])
def test_lean_budget_boundary_and_disabled_watchdog(monkeypatch, budget, expected_calls):
    from types import SimpleNamespace
    from agent.context_compressor import ContextCompressor

    compressor = object.__new__(ContextCompressor)
    compressor._compression_remaining_seconds = lambda: budget
    compressor._session_id = "budget-boundary"
    provider = MagicMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))]))
    monkeypatch.setattr("agent.auxiliary_client.call_llm", provider)
    summary = compressor._augment_summary_lean("main summary", [{"role": "user", "content": "verbatim requirement"}])
    assert provider.call_count == expected_calls
    assert "main summary" in summary and "verbatim requirement" in summary
    assert "session_search" in summary
    if expected_calls == 0:
        assert "budget exhausted" in summary
    else:
        assert provider.call_args.kwargs["timeout"] is None


def test_sync_host_passes_remaining_budget_through_real_compression(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from run_agent import AIAgent
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("budget-sync", source="cli")
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                    quiet_mode=True, session_db=db, session_id="budget-sync", skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = "sys"
    agent._compression_feasibility_checked = True
    compressor = agent.context_compressor
    observed = []
    monkeypatch.setattr("agent.conversation_compression.resolve_context_compression_timeouts", lambda: (10, 20))
    monkeypatch.setattr("agent.context_compressor._serialize_turns_for_digest", lambda *args: "x" * 150000)

    def provider(**kwargs):
        observed.append(kwargs["timeout"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))])

    def compress(messages, **kwargs):
        compressor._augment_summary_lean("summary", messages)
        return messages  # no-op isolates plumbing from session rotation

    monkeypatch.setattr("agent.auxiliary_client.call_llm", provider)
    monkeypatch.setattr(compressor, "compress", compress)
    original = [{"role": "user", "content": "keep"}]
    try:
        assert agent._compress_context(original, "sys", approx_tokens=120000)[0] is original
        assert len(observed) == 3
        assert all(value is not None and 0 < value < 20 for value in observed)
        assert getattr(compressor, "_compression_remaining_seconds", None) is None
        assert getattr(compressor, "_compression_cancelled_check", None) is None
    finally:
        agent._end_session_on_close = False
        agent.close()
        db.close()


@pytest.mark.parametrize("cause", ["budget", "host", "hard"])
def test_inflight_digest_stops_via_existing_protected_provider_seam(monkeypatch, cause):
    import threading
    from types import SimpleNamespace
    from agent.context_compressor import ContextCompressor
    from agent.auxiliary_client import (
        AuxiliaryExplicitCancellation, aux_interrupt_protection,
        _run_protected_sync_provider_call,
    )

    remaining = [20.0]
    cancelled = threading.Event()
    hard = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    provider_done = threading.Event()
    owner_done = threading.Event()
    calls = []
    results = []
    compressor = object.__new__(ContextCompressor)
    compressor._compression_cancelled_check = cancelled.is_set
    compressor._compression_remaining_seconds = lambda: remaining[0]
    monkeypatch.setattr("agent.context_compressor._serialize_turns_for_digest", lambda *args: "x" * 150000)

    def provider(kwargs):
        calls.append(kwargs)
        entered.set()
        try:
            assert release.wait(5)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="late digest"))])
        finally:
            provider_done.set()

    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: _run_protected_sync_provider_call(provider, kw))

    def work():
        try:
            with aux_interrupt_protection(cancel_event=hard):
                results.append(compressor._augment_summary_lean("main summary", [{"role": "user", "content": "verbatim user"}]))
        except BaseException as exc:
            results.append(exc)
        finally:
            owner_done.set()

    thread = threading.Thread(target=work)
    thread.start()
    try:
        assert entered.wait(2)
        if cause == "budget":
            remaining[0] = 0.0
        elif cause == "host":
            cancelled.set()
        else:
            hard.set()
        assert owner_done.wait(2), "digest owner outlived cancellation/budget"
        assert not release.is_set()
        assert len(calls) == 1
        if cause == "hard":
            assert isinstance(results[0], AuxiliaryExplicitCancellation)
        else:
            assert isinstance(results[0], str)
            assert "main summary" in results[0]
            assert "verbatim user" in results[0]
            assert "session_search" in results[0]
    finally:
        release.set()
        thread.join(5)
        assert provider_done.wait(2)


@pytest.mark.parametrize("cancel_seam", ["digest", "provenance", "static_fallback"])
def test_real_sync_timeout_during_digest_preserves_context_and_cooldown(monkeypatch, tmp_path, caplog, cancel_seam):
    import copy
    import threading
    from types import SimpleNamespace
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from agent import conversation_compression as cc

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("cancel-digest", source="cli")
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                    quiet_mode=True, session_db=db, session_id="cancel-digest", skip_context_files=True, skip_memory=True)
    agent._end_session_on_close = False
    agent._cached_system_prompt = "sys"
    agent._compression_feasibility_checked = True
    compressor = agent.context_compressor
    compressor.tail_mode = "lean"
    original = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}: " + "detail " * 1000} for i in range(60)]
    baseline = copy.deepcopy(original)
    for msg in original:
        db.append_message("cancel-digest", msg["role"], msg["content"])
    stored = db.get_messages("cancel-digest")
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    calls = []
    real_compress = cc.compress_context

    def tracked(*args, **kwargs):
        try:
            return real_compress(*args, **kwargs)
        finally:
            done.set()

    def pause_for_host():
        from agent.auxiliary_client import _notify_aux_progress
        _notify_aux_progress()
        entered.set()
        assert release.wait(10)

    def provider(**kwargs):
        calls.append(kwargs)
        if "main_runtime" not in kwargs and cancel_seam != "provenance":
            pause_for_host()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="## Active Task\nContinue the requested work.\n## Completed Actions\nReviewed details."))])

    if cancel_seam == "provenance":
        validate = compressor._validate_summary_user_provenance

        def paused_validate(*args):
            pause_for_host()
            return validate(*args)

        monkeypatch.setattr(compressor, "_validate_summary_user_provenance", paused_validate)
    elif cancel_seam == "static_fallback":
        monkeypatch.setattr(compressor, "_generate_summary", lambda *a, **kw: None)

    monkeypatch.setattr(cc, "compress_context", tracked)
    # The regression is silence DURING digest generation, not slow setup.
    # Event-gate the idle observation until that exact seam is reached.
    original_since_progress = cc.CompressionCommitFence.seconds_since_progress
    monkeypatch.setattr(cc.CompressionCommitFence, "seconds_since_progress",
                        lambda fence: original_since_progress(fence) if entered.is_set() else 0.0)
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda: (0.2, 20))
    monkeypatch.setattr("agent.context_compressor.call_llm", provider)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", provider)
    monkeypatch.setattr("agent.context_compressor._serialize_turns_for_digest", lambda *args: "x" * 150000)
    try:
        result, prompt = agent._compress_context(original, "sys", approx_tokens=120000)
        assert entered.is_set(), "must reach an actual lean digest through compress()"
        assert result is original and prompt == "sys"
        assert original == baseline
        assert db.get_messages("cancel-digest") == stored
        cooldown_error = compressor._last_summary_error
        cooldown_until = compressor._summary_failure_cooldown_until
        cooldown_row = db.get_compression_failure_cooldown_row("cancel-digest")
        assert "idle timeout" in cooldown_error
        release.set()
        assert done.wait(3)
        assert len(calls) == {"digest": 2, "provenance": 4, "static_fallback": 1}[cancel_seam]
        assert compressor._summary_failure_cooldown_until == cooldown_until
        assert compressor._consecutive_timeout_failures == 1
        assert db.get_compression_failure_cooldown_row("cancel-digest") == cooldown_row
        assert original == baseline
        assert db.get_messages("cancel-digest") == stored
        assert agent.session_id == "cancel-digest"
        assert compressor._previous_summary is None
        assert compressor._last_summary_error == cooldown_error
        assert '"failure_class":"commit_fence_cancelled"' in caplog.text
    finally:
        release.set()
        assert done.wait(5)
        agent.close()
        db.close()
