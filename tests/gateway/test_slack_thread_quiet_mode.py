"""Slack trailing mode flags and durable per-thread display policy."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.thread_quiet_mode import (
    QUIET_MODE_EVENT_KEY,
    QUIET_MODE_SESSION_KEY,
    resolve_thread_display_policy,
    resolve_thread_quiet_mode,
    strip_trailing_mode_flag,
)


@pytest.mark.parametrize(
    ("raw", "clean", "quiet_requested"),
    [
        ("Investigate the failed deploy ~", "Investigate the failed deploy", False),
        ("Investigate this\n\n~   ", "Investigate this", False),
        ("Return to quiet ~~", "Return to quiet", True),
        ("Return to quiet\n\n~~   ", "Return to quiet", True),
    ],
)
def test_trailing_mode_flag_is_removed(raw, clean, quiet_requested):
    assert strip_trailing_mode_flag(raw) == (clean, quiet_requested)


@pytest.mark.parametrize(
    "text",
    [
        "Explain what —q means here",
        "Investigate the failed deploy —q",
        "Investigate the failed deploy --q",
        "Use ~ in the middle of a sentence",
        "A code sample: `~` then continue",
        "Ordinary message",
    ],
)
def test_non_trailing_mode_text_is_preserved(text):
    assert strip_trailing_mode_flag(text) == (text, None)


def test_quiet_policy_hides_tool_chain_but_keeps_progress_summaries():
    policy = resolve_thread_display_policy(quiet_mode=True)

    assert policy.streaming is True
    assert policy.tool_progress is False
    assert policy.live_status is False
    assert policy.tool_log is True
    assert policy.interim_assistant_messages is True
    assert policy.thinking_progress is False
    assert policy.native_task_cards is False
    assert policy.long_running_notifications is True
    assert policy.tool_progress_mode_override is None
    assert policy.thinking_progress_mode_override is None


def test_verbose_policy_enables_every_progress_surface():
    policy = resolve_thread_display_policy(quiet_mode=False)

    assert all(
        (
            policy.streaming,
            policy.tool_progress,
            policy.live_status,
            policy.tool_log,
            policy.interim_assistant_messages,
            policy.thinking_progress,
            policy.native_task_cards,
            policy.long_running_notifications,
        )
    )
    assert policy.tool_progress_mode_override == "all"
    assert policy.thinking_progress_mode_override == "raw"


@pytest.mark.asyncio
async def test_slack_defaults_to_quiet_mode_without_a_saved_preference():
    store = AsyncMock()
    entry = SimpleNamespace(session_key="slack:T:C:thread", metadata={})
    event = SimpleNamespace(metadata={})
    source = SimpleNamespace(platform=Platform.SLACK)

    assert await resolve_thread_quiet_mode(store, entry, event, source) is True
    store.set_session_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_slack_verbose_request_persists_and_applies_immediately():
    store = AsyncMock()
    store.set_session_metadata.return_value = True
    entry = SimpleNamespace(session_key="slack:T:C:thread", metadata={})
    event = SimpleNamespace(metadata={QUIET_MODE_EVENT_KEY: False})
    source = SimpleNamespace(platform=Platform.SLACK)

    enabled = await resolve_thread_quiet_mode(store, entry, event, source)

    assert enabled is False
    store.set_session_metadata.assert_awaited_once_with(
        entry.session_key, QUIET_MODE_SESSION_KEY, False
    )


@pytest.mark.asyncio
async def test_slack_quiet_request_persists_and_applies_immediately():
    store = AsyncMock()
    store.set_session_metadata.return_value = True
    entry = SimpleNamespace(session_key="slack:T:C:thread", metadata={})
    event = SimpleNamespace(metadata={QUIET_MODE_EVENT_KEY: True})
    source = SimpleNamespace(platform=Platform.SLACK)

    enabled = await resolve_thread_quiet_mode(store, entry, event, source)

    assert enabled is True
    store.set_session_metadata.assert_awaited_once_with(
        entry.session_key, QUIET_MODE_SESSION_KEY, True
    )


@pytest.mark.parametrize("quiet_mode", [True, False])
@pytest.mark.asyncio
async def test_persisted_slack_mode_applies_without_repeating_flag(quiet_mode):
    store = AsyncMock()
    entry = SimpleNamespace(
        session_key="slack:T:C:thread",
        metadata={QUIET_MODE_SESSION_KEY: quiet_mode},
    )
    event = SimpleNamespace(metadata={})
    source = SimpleNamespace(platform=Platform.SLACK)

    assert await resolve_thread_quiet_mode(store, entry, event, source) is quiet_mode
    store.set_session_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_mode_event_metadata_does_not_affect_other_platforms():
    store = AsyncMock()
    entry = SimpleNamespace(session_key="telegram:chat", metadata={})
    event = SimpleNamespace(metadata={QUIET_MODE_EVENT_KEY: True})
    source = SimpleNamespace(platform=Platform.TELEGRAM)

    assert await resolve_thread_quiet_mode(store, entry, event, source) is False
    store.set_session_metadata.assert_not_awaited()
