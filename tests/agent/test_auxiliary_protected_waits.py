"""Protected auxiliary budgets include retry backoff and semaphore queues."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor


@pytest.mark.parametrize("cause", ["budget", "hard"])
def test_real_retry_loop_observes_digest_cutoff(monkeypatch, cause):
    clock = [0.0]
    attempts = []

    def sleep(seconds):
        clock[0] += seconds

    def create(**kwargs):
        attempts.append(clock[0])
        raise httpx.RemoteProtocolError("incomplete chunked read")

    client = SimpleNamespace(
        base_url="https://example.test/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(aux, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda *a: ("custom", "test", client.base_url, "test", "chat_completions"))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (client, "test"))
    monkeypatch.setattr(aux, "_get_task_extra_body", lambda task: {})
    monkeypatch.setattr(aux, "_acquire_sync_aux_semaphore", lambda task: None)
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 2)
    compressor = object.__new__(ContextCompressor)
    compressor._session_id = "retry-budget"
    compressor._compression_remaining_seconds = lambda: 1.2 - clock[0] if cause == "budget" else 20.0
    monkeypatch.setattr("agent.context_compressor._serialize_turns_for_digest", lambda *a: "x" * 150000)

    with aux.aux_interrupt_protection(cancel_check=lambda: cause == "hard" and clock[0] >= 1.08):
        if cause == "hard":
            with pytest.raises(aux.AuxiliaryExplicitCancellation):
                compressor._augment_summary_lean("main summary", [{"role": "user", "content": "verbatim requirement"}])
        else:
            result = compressor._augment_summary_lean("main summary", [{"role": "user", "content": "verbatim requirement"}])
            assert "main summary" in result
            assert "verbatim requirement" in result
            assert "budget exhausted" in result
            assert "session_search" in result
    assert clock[0] < 1.2, "retry backoff exhausted the host's commit reserve"
    assert len(attempts) == 2, "no provider retry or sibling may start after cutoff"


@pytest.mark.parametrize("cause", ["budget", "hard"])
def test_protected_semaphore_queue_observes_digest_cutoff(monkeypatch, cause):
    clock = [0.0]
    releases = []

    class BusySemaphore:
        def acquire(self, timeout=None):
            # Without bounded polling the permit becomes available too late.
            clock[0] += 3.0 if timeout is None else timeout
            return clock[0] >= 3.0

        def release(self):
            releases.append(True)

    provider = MagicMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="late"))]))
    monkeypatch.setattr(aux, "_acquire_sync_aux_semaphore", lambda task: BusySemaphore())
    monkeypatch.setattr(aux, "_call_llm_impl", provider)
    compressor = object.__new__(ContextCompressor)
    compressor._session_id = "queue-budget"
    compressor._compression_remaining_seconds = lambda: 1.2 - clock[0] if cause == "budget" else 20.0
    with aux.aux_interrupt_protection(cancel_check=lambda: cause == "hard" and clock[0] >= 1.08):
        if cause == "hard":
            with pytest.raises(aux.AuxiliaryExplicitCancellation):
                compressor._augment_summary_lean("main summary", [{"role": "user", "content": "verbatim requirement"}])
        else:
            result = compressor._augment_summary_lean("main summary", [{"role": "user", "content": "verbatim requirement"}])
            assert "main summary" in result and "verbatim requirement" in result
            assert "budget exhausted" in result and "session_search" in result
    assert clock[0] < 1.2
    provider.assert_not_called()
    assert releases == [], "never release a permit this request did not acquire"
