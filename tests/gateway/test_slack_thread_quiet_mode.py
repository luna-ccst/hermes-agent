"""Slack trailing quiet flag and durable per-thread display policy."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.thread_quiet_mode import (
    QUIET_MODE_EVENT_KEY,
    QUIET_MODE_SESSION_KEY,
    resolve_thread_quiet_mode,
    strip_trailing_quiet_flag,
)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Investigate the failed deploy ~", "Investigate the failed deploy"),
        ("Investigate this\n\n~   ", "Investigate this"),
    ],
)
def test_trailing_quiet_flag_is_removed(raw, clean):
    assert strip_trailing_quiet_flag(raw) == (clean, True)


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
def test_non_trailing_quiet_text_is_preserved(text):
    assert strip_trailing_quiet_flag(text) == (text, False)


@pytest.mark.asyncio
async def test_slack_quiet_request_persists_on_session_and_applies_immediately():
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


@pytest.mark.asyncio
async def test_persisted_slack_quiet_mode_applies_without_repeating_flag():
    store = AsyncMock()
    entry = SimpleNamespace(
        session_key="slack:T:C:thread",
        metadata={QUIET_MODE_SESSION_KEY: True},
    )
    event = SimpleNamespace(metadata={})
    source = SimpleNamespace(platform=Platform.SLACK)

    assert await resolve_thread_quiet_mode(store, entry, event, source) is True
    store.set_session_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_quiet_event_metadata_does_not_affect_other_platforms():
    store = AsyncMock()
    entry = SimpleNamespace(session_key="telegram:chat", metadata={})
    event = SimpleNamespace(metadata={QUIET_MODE_EVENT_KEY: True})
    source = SimpleNamespace(platform=Platform.TELEGRAM)

    assert await resolve_thread_quiet_mode(store, entry, event, source) is False
    store.set_session_metadata.assert_not_awaited()
