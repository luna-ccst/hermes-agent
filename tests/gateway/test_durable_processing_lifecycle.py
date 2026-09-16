"""Real Base/Runner lifecycle seams; no network, model, or live state."""
import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageDisposition, MessageEvent, ProcessingOutcome
from gateway.run import GatewayRunner, _dequeue_pending_event
from gateway.session import SessionSource, build_session_key


class ReceiptAdapter(BasePlatformAdapter):
    @property
    def authorization_is_upstream(self):
        return True

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=False), Platform.WEBHOOK)
        self.receipts = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def on_processing_complete(self, event, outcome):
        self.receipts.append((event.message_id, outcome))


def event(identifier="delivery", *, internal=False):
    result = MessageEvent(text="synthetic prompt", message_id=identifier,
        source=SessionSource(platform=Platform.WEBHOOK, chat_id="thread", user_id="valid-user", chat_type="dm"),
        internal=internal, allow_gateway_control=False)
    # setattr also exercises the baseline, before the field is added.
    result.requires_processing_completion = True
    return result


def runner_for(adapter):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.WEBHOOK: adapter}
    runner._draining = False
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._running_agents = {}
    runner.session_store = None
    runner._startup_restore_queue = []
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    return runner


@pytest_asyncio.fixture
async def busy_control():
    adapter = ReceiptAdapter()
    message = event("control")
    message.allow_gateway_control = True
    key = build_session_key(message.source)
    owner = asyncio.create_task(asyncio.Event().wait())
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = owner
    try:
        yield adapter, message, owner
    finally:
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "/stop"])
@pytest.mark.parametrize("delivery", ["failed", "raised", "cancelled"])
async def test_control_receipt_reflects_delivery(busy_control, command, delivery):
    adapter, message, owner = busy_control
    message.text = command
    async def handler(message):
        return "response"
    async def send(*args, **kwargs):
        if delivery == "raised":
            raise RuntimeError("synthetic send failure")
        if delivery == "cancelled":
            raise asyncio.CancelledError()
        return SendResult(success=False, error="synthetic send failure")
    adapter.set_message_handler(handler)
    adapter.send = send
    if delivery == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await adapter.handle_message(message)
    else:
        await adapter.handle_message(message)
    assert adapter.receipts == [(message.message_id, ProcessingOutcome.FAILURE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "/stop", None])
async def test_startup_cancel_has_exactly_one_receipt(busy_control, command):
    adapter, message, owner = busy_control
    runner = runner_for(adapter)
    message.text = command or "ordinary prompt"
    entered = asyncio.Event()
    async def handler(message):
        entered.set()
        await asyncio.Event().wait()
    adapter.set_message_handler(handler)
    task = asyncio.create_task(runner._dispatch_startup_restore_event(adapter, message))
    if command:
        await entered.wait()
    else:
        await asyncio.sleep(0)  # durable ownership wait, before Base completion
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.receipts == [(message.message_id, ProcessingOutcome.FAILURE)]
    assert not owner.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("result", ["success", "silent", "rejected", "deferred", "send_failed", "send_raised", "send_cancelled", "handler_cancelled", "handler_raised"])
async def test_clarification_bypass_receipt(busy_control, replay, result):
    from tools import clarify_gateway
    adapter, message, owner = busy_control
    key = build_session_key(message.source)
    clarify_gateway.register("receipt-test", key, "Which?", ["one", "two"])
    message.text = "one"
    async def handler(message):
        if result == "handler_cancelled":
            raise asyncio.CancelledError()
        if result == "handler_raised":
            raise RuntimeError("synthetic handler failure")
        if result == "rejected":
            return MessageDisposition.REJECTED
        if result == "deferred":
            return MessageDisposition.DEFERRED
        assert clarify_gateway.resolve_text_response_for_session(key, message.text)
        return None if result == "silent" else "accepted"
    async def send(*args, **kwargs):
        if result == "send_raised":
            raise RuntimeError("synthetic send failure")
        if result == "send_cancelled":
            raise asyncio.CancelledError()
        return SendResult(success=result != "send_failed", error="synthetic failure")
    adapter.set_message_handler(handler)
    adapter.send = send
    try:
        dispatch = (runner_for(adapter)._dispatch_startup_restore_event(adapter, message)
                    if replay else adapter.handle_message(message))
        if result.endswith("cancelled"):
            with pytest.raises(asyncio.CancelledError):
                await dispatch
        else:
            await dispatch
        expected = (ProcessingOutcome.SUCCESS if result in {"success", "silent"} else
                    ProcessingOutcome.CANCELLED if result == "rejected" else ProcessingOutcome.FAILURE)
        assert adapter.receipts == ([] if result == "deferred" else [(message.message_id, expected)])
        assert not owner.done()
        assert not adapter._durable_waiters
    finally:
        clarify_gateway.clear_session(key)


async def drain(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*tuple(adapter._background_tasks))


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_durable_followup_cannot_be_consumed_by_in_band_runtime_drain(monkeypatch, internal):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = ReceiptAdapter()
    runner = runner_for(adapter)
    first = event("first", internal=internal)
    second = event("second")
    key = build_session_key(first.source)
    # Real auth lookup must honor the registered upstream-authorizing adapter.
    assert runner._is_user_authorized(second.source)
    started, release = asyncio.Event(), asyncio.Event()
    handled, consumed = [], []

    async def model_boundary(message):
        handled.append(message.message_id)
        if message is first:
            started.set()
            await release.wait()
            # This is the real dequeue helper used by _run_agent's recursive
            # followup. Such recursion has no second Base wrapper/receipt.
            pending = _dequeue_pending_event(adapter, key)
            if pending:
                consumed.append(pending.message_id)
        return None

    adapter.set_message_handler(model_boundary)
    await adapter.handle_message(first)
    await started.wait()
    submitted = asyncio.create_task(adapter.handle_message(second))
    await asyncio.sleep(0)  # run dispatch to its first blocking boundary
    release.set()
    try:
        await submitted
        await drain(adapter)
        assert consumed == [], "durable delivery was stolen by the in-band drain"
        assert handled == ["first", "second"]
        assert adapter.receipts == [("first", ProcessingOutcome.SUCCESS), ("second", ProcessingOutcome.SUCCESS)]
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_busy_rejection_completes_once_without_touching_active_run():
    adapter = ReceiptAdapter()
    runner = runner_for(adapter)
    runner._is_user_authorized = lambda source: False
    denied = event()
    denied.requires_processing_completion = False
    key = build_session_key(denied.source)
    release = asyncio.Event()
    owner = asyncio.create_task(release.wait())
    guard = asyncio.Event()
    adapter._active_sessions[key] = guard
    adapter._session_tasks[key] = owner
    adapter.set_message_handler(AsyncMock(side_effect=AssertionError("denied reached handler")))
    try:
        await adapter.handle_message(denied)
        assert adapter.receipts == [(denied.message_id, ProcessingOutcome.CANCELLED)]
        assert not guard.is_set()
        assert adapter._pending_messages == {}
        adapter._message_handler.assert_not_awaited()
        assert not owner.done()
    finally:
        release.set()
        await owner


@pytest.mark.asyncio
async def test_startup_handoff_has_no_receipt_until_actual_replay(monkeypatch):
    adapter = ReceiptAdapter()
    runner = runner_for(adapter)
    runner._startup_restore_in_progress = True
    runner.adapters = {}
    message = event()
    adapter.set_message_handler(runner._handle_message)
    await adapter.handle_message(message)
    await drain(adapter)
    assert runner._startup_restore_queue == [message]
    assert adapter.receipts == []
    from tests.gateway.test_incomplete_gateway_turns import _make_runner
    import gateway.run as gateway_run
    # Registration happens after startup intake. Replay executes the real cold
    # handler and real upstream authorization; only the model/API is stubbed.
    replay = _make_runner(adapter)
    del replay._is_user_authorized
    replay.adapters = {Platform.WEBHOOK: adapter}
    replay._startup_restore_queue = runner._startup_restore_queue
    replay._startup_restore_in_progress = True
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *args, **kwargs: 100)
    adapter.set_message_handler(replay._handle_message)
    assert replay._is_user_authorized(message.source)
    assert await replay._drain_startup_restore_queue() == 1
    await drain(adapter)
    replay._run_agent.assert_awaited_once()
    assert adapter.receipts == [(message.message_id, ProcessingOutcome.SUCCESS)]


@pytest.mark.asyncio
async def test_durable_cold_rejection_is_terminal_not_success():
    adapter = ReceiptAdapter()
    runner = runner_for(adapter)
    runner._is_user_authorized_for_source = lambda source: False
    runner._get_unauthorized_dm_behavior = lambda *args, **kwargs: "ignore"
    runner._scale_to_zero_note_real_inbound = lambda: None
    adapter.set_message_handler(runner._handle_message)
    message = event()
    await adapter.handle_message(message)
    await drain(adapter)
    assert adapter.receipts == [(message.message_id, ProcessingOutcome.CANCELLED)]


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected", [False, True])
@pytest.mark.parametrize("command", ["/stop", "/status"])
async def test_durable_stop_has_receipt_and_denied_stop_preserves_owner(rejected, command):
    adapter = ReceiptAdapter()
    message = event("stop")
    message.text = command
    message.allow_gateway_control = True
    key = build_session_key(message.source)
    release = asyncio.Event()
    owner = asyncio.create_task(release.wait())
    guard = asyncio.Event()
    adapter._active_sessions[key] = guard
    adapter._session_tasks[key] = owner
    async def control(message):
        return MessageDisposition.REJECTED if rejected else "stopped"
    adapter.set_message_handler(control)
    try:
        await adapter.handle_message(message)
        expected = ProcessingOutcome.CANCELLED if rejected else ProcessingOutcome.SUCCESS
        assert adapter.receipts == [(message.message_id, expected)]
        if rejected:
            assert adapter._active_sessions[key] is guard
            assert not guard.is_set()
            assert not owner.done()
        elif command == "/stop":
            assert owner.cancelled()
        else:
            assert not owner.done()
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.asyncio
async def test_unconfigured_durable_delivery_receives_failure():
    adapter = ReceiptAdapter()
    message = event()
    await adapter.handle_message(message)
    assert adapter.receipts == [(message.message_id, ProcessingOutcome.FAILURE)]


@pytest.mark.asyncio
async def test_cancelling_waiting_delivery_never_cancels_existing_owner():
    adapter = ReceiptAdapter()
    first, second = event("first"), event("second")
    entered, release = asyncio.Event(), asyncio.Event()
    handled = []
    async def model(message):
        handled.append(message.message_id)
        entered.set()
        await release.wait()
    adapter.set_message_handler(model)
    await adapter.handle_message(first)
    await entered.wait()
    owner = adapter._session_tasks[build_session_key(first.source)]
    waiting = asyncio.create_task(adapter.handle_message(second))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not owner.done()
    release.set()
    await drain(adapter)
    assert handled == ["first"]
    assert adapter.receipts == [("first", ProcessingOutcome.SUCCESS)]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel", "expected_cancel", "cleanup"])
async def test_background_outcome_is_reported_exactly_once(failure):
    adapter = ReceiptAdapter()
    message = event()
    async def model(message):
        if failure == "error":
            raise RuntimeError("synthetic handler failure")
        if failure in {"cancel", "expected_cancel"}:
            if failure == "expected_cancel":
                adapter._expected_cancelled_tasks.add(asyncio.current_task())
            raise asyncio.CancelledError()
    adapter.set_message_handler(model)
    if failure == "cleanup":
        calls = 0
        async def flush(key):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic post-receipt flush failure")
        adapter._flush_text_debounce_now = flush
    await adapter.handle_message(message)
    await asyncio.gather(*tuple(adapter._background_tasks), return_exceptions=True)
    expected = (ProcessingOutcome.SUCCESS if failure == "cleanup" else
                ProcessingOutcome.CANCELLED if failure == "expected_cancel" else ProcessingOutcome.FAILURE)
    assert adapter.receipts == [(message.message_id, expected)]


@pytest.mark.asyncio
async def test_startup_replay_busy_session_does_not_block_unrelated_session():
    adapter = ReceiptAdapter()
    runner = runner_for(adapter)
    one, two = event("one"), event("two")
    two.source.chat_id = "other-session"
    key = build_session_key(one.source)
    release = asyncio.Event()
    owner = asyncio.create_task(release.wait())
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = owner
    handled = []
    async def model(message):
        handled.append(message.message_id)
    adapter.set_message_handler(model)
    runner._startup_restore_queue = [one, two]
    dispatch = asyncio.create_task(runner._drain_startup_restore_queue())
    try:
        await asyncio.sleep(0)
        assert dispatch.done(), "one busy session held the global startup drain"
        assert dispatch.result() == 2
        for _ in range(5):
            await asyncio.sleep(0)
        assert handled == ["two"]
        release.set()
        await owner
        await drain(adapter)
        assert handled == ["two", "one"]
        assert sorted(adapter.receipts) == [("one", ProcessingOutcome.SUCCESS), ("two", ProcessingOutcome.SUCCESS)]
    finally:
        release.set()
        await owner
        await dispatch
        await drain(adapter)


@pytest.mark.asyncio
async def test_stop_cancels_durable_waiting_turn_before_releasing_owner():
    adapter = ReceiptAdapter()
    first, second, stop = event("first"), event("second"), event("stop")
    stop.text, stop.allow_gateway_control = "/stop", True
    entered, release = asyncio.Event(), asyncio.Event()
    handled = []
    async def handler(message):
        handled.append(message.message_id)
        if message is first:
            entered.set()
            await release.wait()
        if message is stop:
            return "stopped"
    adapter.set_message_handler(handler)
    await adapter.handle_message(first)
    await entered.wait()
    waiting = asyncio.create_task(adapter.handle_message(second))
    await asyncio.sleep(0)
    try:
        await adapter.handle_message(stop)
        await asyncio.gather(waiting, return_exceptions=True)
        await drain(adapter)
        assert handled == ["first", "stop"]
        assert ("second", ProcessingOutcome.CANCELLED) in adapter.receipts
    finally:
        release.set()
        await adapter.cancel_background_tasks()
