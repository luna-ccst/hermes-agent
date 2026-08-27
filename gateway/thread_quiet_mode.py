"""Per-thread quiet-mode helpers for Slack gateway turns."""

from __future__ import annotations

import re
from typing import Any

from gateway.config import Platform

QUIET_MODE_EVENT_KEY = "slack_thread_quiet_requested"
QUIET_MODE_ACTIVE_EVENT_KEY = "slack_thread_quiet_active"
QUIET_MODE_SESSION_KEY = "slack_thread_quiet_mode"

# The marker must be the final token so ordinary prose containing ``~`` is not
# altered.
_TRAILING_QUIET_FLAG = re.compile(r"~\s*$")


def strip_trailing_quiet_flag(text: str) -> tuple[str, bool]:
    """Remove a trailing Slack quiet flag and report whether it was present."""
    if not isinstance(text, str):
        return text, False
    match = _TRAILING_QUIET_FLAG.search(text)
    if match is None:
        return text, False
    return text[: match.start()].rstrip(), True


async def resolve_thread_quiet_mode(
    async_session_store: Any,
    session_entry: Any,
    event: Any,
    source: Any,
) -> bool:
    """Persist a Slack quiet request and resolve this thread's current mode."""
    if getattr(source, "platform", None) != Platform.SLACK:
        return False

    event_metadata = getattr(event, "metadata", None)
    requested = bool(
        isinstance(event_metadata, dict)
        and event_metadata.get(QUIET_MODE_EVENT_KEY)
    )
    if requested:
        await async_session_store.set_session_metadata(
            session_entry.session_key,
            QUIET_MODE_SESSION_KEY,
            True,
        )
        # The store normally mutates the same SessionEntry instance. Keep this
        # assignment as a small defense for test doubles and alternate stores.
        session_entry.metadata[QUIET_MODE_SESSION_KEY] = True

    return requested or bool(
        getattr(session_entry, "metadata", {}).get(QUIET_MODE_SESSION_KEY)
    )
