"""Real waiters and trusted gateway correlation; no commands executed."""
import pytest
from gateway.run import GatewayRunner
from gateway.platforms.event import MessageEvent
from tools import approval, clarify_gateway as cg
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['/approve', '/approve session', '/approve always', '/deny'])
async def test_bound_approval_never_resolves_replacement(text):
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda source: 'bound-test'
    first = _ApprovalEntry({'command': 'not executed'})
    second = _ApprovalEntry({'command': 'not executed either'})
    event = MessageEvent(text=text, trusted_prompt_reply={
        'kind': 'approval', 'session_key': 'bound-test',
        'request_id': first.data['request_id'],
    })
    with approval._lock:
        approval._gateway_queues['bound-test'] = [second]
    try:
        await runner._handle_bound_prompt_reply(event, 'bound-test')
        assert second.result is None
    finally:
        with approval._lock:
            approval._gateway_queues.pop('bound-test', None)


@pytest.mark.asyncio
async def test_bound_clarify_does_not_resolve_replacement():
    runner = object.__new__(GatewayRunner)
    entry = cg.register('new-question', 'bound-test', 'New?', None)
    try:
        await runner._handle_bound_prompt_reply(MessageEvent(
            text='old answer', trusted_prompt_reply={
                'kind': 'clarify', 'session_key': 'bound-test', 'request_id': 'old-question',
            }), 'bound-test')
        assert not entry.event.is_set()
    finally:
        cg.clear_session('bound-test')


@pytest.mark.parametrize('request_id', ['', 'stale'])
def test_explicit_request_never_falls_back_to_fifo_or_all(request_id):
    first = _ApprovalEntry({'command': 'A'})
    second = _ApprovalEntry({'command': 'B'})
    with approval._lock:
        approval._gateway_queues['optional-id'] = [first, second]
    try:
        assert approval.resolve_gateway_approval(
            'optional-id', 'once', resolve_all=True, request_id=request_id,
        ) == 0
        assert first.result is second.result is None
        assert approval.resolve_gateway_approval('optional-id', 'once') == 1
        assert first.result == 'once' and second.result is None
    finally:
        with approval._lock:
            approval._gateway_queues.pop('optional-id', None)


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['/approve all', '/approve always now', '/restart', '/deny all'])
async def test_bound_input_cannot_execute_general_slash_commands(text):
    runner = object.__new__(GatewayRunner)
    event = MessageEvent(text=text, trusted_prompt_reply={
        'kind': 'approval', 'session_key': 'malicious', 'request_id': 'valid-id',
    })
    assert await runner._handle_bound_prompt_reply(event, 'malicious') == ''


def test_clarify_replacement_during_text_coercion_cannot_consume_new_prompt(monkeypatch):
    original = cg.register('race-A', 'race-session', 'A?', None)
    original_coerce = cg._coerce_text_response_detailed

    def replace_during_coercion(entry, text):
        cg.clear_session('race-session')
        cg.register('race-B', 'race-session', 'B?', None)
        return original_coerce(entry, text)

    monkeypatch.setattr(cg, '_coerce_text_response_detailed', replace_during_coercion)
    try:
        assert cg.attempt_text_response_for_session(
            'race-session', 'answer A', request_id=original.clarify_id,
        ) == cg.TEXT_NO_PENDING
        pending = cg.get_pending_for_session('race-session', include_choice_prompts=True)
        assert pending.clarify_id == 'race-B' and not pending.event.is_set()
    finally:
        cg.clear_session('race-session')


def test_cancelled_approval_cannot_be_reapproved():
    entry = _ApprovalEntry({'command': 'never executed'})
    entry.result = 'deny'
    entry.event.set()
    with approval._lock:
        approval._gateway_queues['cancelled-test'] = [entry]
    try:
        assert approval.resolve_gateway_approval(
            'cancelled-test', 'once', request_id=entry.data['request_id'],
        ) == 0
        assert entry.result == 'deny'
    finally:
        with approval._lock:
            approval._gateway_queues.pop('cancelled-test', None)


@pytest.mark.asyncio
@pytest.mark.parametrize('authorized', [False, True])
@pytest.mark.parametrize('kind,text', [('approval', '/approve'), ('clarify', 'answer')])
async def test_real_gateway_authorizes_bound_reply(monkeypatch, authorized, kind, text):
    from gateway.session import SessionSource
    from gateway.config import Platform
    from hermes_cli import lifecycle
    monkeypatch.setattr(lifecycle, 'invoke_hook', lambda *a, **kw: [])
    runner = object.__new__(GatewayRunner)
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized_for_source = lambda source: authorized
    runner._get_unauthorized_dm_behavior = lambda *a, **kw: 'ignore'
    runner._session_key_for_source = lambda source: 'integration-test'
    runner.adapters = {}
    runner._pending_approvals = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='test', user_id='actor')
    entry = _ApprovalEntry({'command': 'not executed'})
    question = cg.register('integration-question', 'integration-test', 'Question?', None)
    with approval._lock:
        approval._gateway_queues['integration-test'] = [entry]
    event = MessageEvent(text=text, source=source, trusted_prompt_reply={
        'kind': kind, 'session_key': 'integration-test',
        'request_id': entry.data['request_id'] if kind == 'approval' else question.clarify_id,
    })
    try:
        await runner._handle_message(event)
        assert (entry.result == 'once') == (authorized and kind == 'approval')
        assert (question.response == 'answer') == (authorized and kind == 'clarify')
    finally:
        cg.clear_session('integration-test')
        with approval._lock:
            approval._gateway_queues.pop('integration-test', None)
