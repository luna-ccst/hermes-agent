"""Per-thread quiet/verbose display helpers for Slack gateway turns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from gateway.config import Platform

QUIET_MODE_EVENT_KEY = "slack_thread_quiet_requested"
QUIET_MODE_ACTIVE_EVENT_KEY = "slack_thread_quiet_active"
QUIET_MODE_SESSION_KEY = "slack_thread_quiet_mode"

# The marker must be the final token so ordinary prose containing ``~`` is not
# altered. A single ``~`` requests verbose mode; ``~~`` restores quiet mode.
_TRAILING_MODE_FLAG = re.compile(r"(?P<marker>~~|~)\s*$")


@dataclass(frozen=True)
class ThreadDisplayPolicy:
    """User-visible progress surfaces for a Slack thread turn."""

    streaming: bool
    tool_progress: bool
    live_status: bool
    tool_log: bool
    interim_assistant_messages: bool
    thinking_progress: bool
    native_task_cards: bool
    long_running_notifications: bool
    tool_progress_mode_override: str | None
    thinking_progress_mode_override: str | None


def resolve_thread_display_policy(quiet_mode: bool) -> ThreadDisplayPolicy:
    """Keep summaries visible in quiet mode while hiding the raw work chain."""
    verbose = not quiet_mode
    return ThreadDisplayPolicy(
        streaming=True,
        tool_progress=verbose,
        live_status=verbose,
        tool_log=True,
        interim_assistant_messages=True,
        thinking_progress=verbose,
        native_task_cards=verbose,
        long_running_notifications=True,
        tool_progress_mode_override="all" if verbose else None,
        thinking_progress_mode_override="raw" if verbose else None,
    )


def strip_trailing_mode_flag(text: str) -> tuple[str, bool | None]:
    """Strip a trailing Slack mode flag and return its requested quiet state."""
    if not isinstance(text, str):
        return text, None
    match = _TRAILING_MODE_FLAG.search(text)
    if match is None:
        return text, None
    quiet_requested = match.group("marker") == "~~"
    return text[: match.start()].rstrip(), quiet_requested


async def resolve_thread_quiet_mode(
    async_session_store: Any,
    session_entry: Any,
    event: Any,
    source: Any,
) -> bool:
    """Persist a Slack mode request and resolve this thread's quiet state."""
    if getattr(source, "platform", None) != Platform.SLACK:
        return False

    event_metadata = getattr(event, "metadata", None)
    request_present = bool(
        isinstance(event_metadata, dict)
        and QUIET_MODE_EVENT_KEY in event_metadata
    )
    if request_present and isinstance(event_metadata, dict):
        quiet_requested = bool(event_metadata[QUIET_MODE_EVENT_KEY])
        await async_session_store.set_session_metadata(
            session_entry.session_key,
            QUIET_MODE_SESSION_KEY,
            quiet_requested,
        )
        # The store normally mutates the same SessionEntry instance. Keep this
        # assignment as a small defense for test doubles and alternate stores.
        session_entry.metadata[QUIET_MODE_SESSION_KEY] = quiet_requested
        return quiet_requested

    return bool(
        getattr(session_entry, "metadata", {}).get(QUIET_MODE_SESSION_KEY, True)
    )
