"""Protected auxiliary budgets include retry backoff and semaphore queues."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agent import auxiliary_client as aux


def test_real_retry_loop_observes_owner_cancellation(monkeypatch):
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
    with aux.aux_interrupt_protection(cancel_check=lambda: clock[0] >= 1.08):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            aux.call_llm(task="compression", messages=[{"role": "user", "content": "summarize"}])
    assert clock[0] < 1.2, "retry backoff exhausted the host's commit reserve"
    assert len(attempts) == 2, "no provider retry or sibling may start after cutoff"


def test_protected_semaphore_queue_observes_owner_cancellation(monkeypatch):
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
    with aux.aux_interrupt_protection(cancel_check=lambda: clock[0] >= 1.08):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            aux.call_llm(task="compression", messages=[{"role": "user", "content": "summarize"}])
    assert clock[0] < 1.2
    provider.assert_not_called()
    assert releases == [], "never release a permit this request did not acquire"
