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



@pytest.mark.parametrize("cancel_seam", ["augmentation", "provenance"])
def test_real_sync_timeout_during_summary_finalization_preserves_context_and_cooldown(monkeypatch, tmp_path, caplog, cancel_seam):
    import copy
    import threading
    from types import SimpleNamespace
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from agent import conversation_compression as cc

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("cancel-finalization", source="cli")
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                    quiet_mode=True, session_db=db, session_id="cancel-finalization", skip_context_files=True, skip_memory=True)
    agent._end_session_on_close = False
    agent._cached_system_prompt = "sys"
    agent._compression_feasibility_checked = True
    compressor = agent.context_compressor
    compressor.tail_mode = "lean"
    original = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}: " + "detail " * 1000} for i in range(60)]
    baseline = copy.deepcopy(original)
    for msg in original:
        db.append_message("cancel-finalization", msg["role"], msg["content"])
    stored = db.get_messages("cancel-finalization")
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
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="## Active Task\nContinue the requested work.\n## Completed Actions\nReviewed details."))])

    if cancel_seam == "provenance":
        validate = compressor._validate_summary_user_provenance

        def paused_validate(*args):
            pause_for_host()
            return validate(*args)

        monkeypatch.setattr(compressor, "_validate_summary_user_provenance", paused_validate)
    else:
        augment = compressor._augment_summary_lean

        def paused_augment(*args):
            pause_for_host()
            return augment(*args)

        monkeypatch.setattr(compressor, "_augment_summary_lean", paused_augment)

    monkeypatch.setattr(cc, "compress_context", tracked)
    # The regression is silence DURING summary finalization, not slow setup.
    # Event-gate the idle observation until that exact seam is reached.
    original_since_progress = cc.CompressionCommitFence.seconds_since_progress
    monkeypatch.setattr(cc.CompressionCommitFence, "seconds_since_progress",
                        lambda fence: original_since_progress(fence) if entered.is_set() else 0.0)
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda: (0.2, 20))
    monkeypatch.setattr("agent.context_compressor.call_llm", provider)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", provider)
    try:
        result, prompt = agent._compress_context(original, "sys", approx_tokens=120000)
        assert entered.is_set(), "must reach summary finalization through compress()"
        assert result is original and prompt == "sys"
        assert original == baseline
        assert db.get_messages("cancel-finalization") == stored
        cooldown_error = compressor._last_summary_error
        cooldown_until = compressor._summary_failure_cooldown_until
        cooldown_row = db.get_compression_failure_cooldown_row("cancel-finalization")
        assert "idle timeout" in cooldown_error
        release.set()
        assert done.wait(3)
        assert len(calls) == 1
        assert compressor._summary_failure_cooldown_until == cooldown_until
        assert compressor._consecutive_timeout_failures == 1
        assert db.get_compression_failure_cooldown_row("cancel-finalization") == cooldown_row
        assert original == baseline
        assert db.get_messages("cancel-finalization") == stored
        assert agent.session_id == "cancel-finalization"
        assert compressor._previous_summary is None
        assert compressor._last_summary_error == cooldown_error
    finally:
        release.set()
        assert done.wait(5)
        agent.close()
        db.close()
