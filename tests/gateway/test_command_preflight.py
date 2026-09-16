import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, Platform, SessionSource
from gateway.platforms.event import MessageDisposition, MessageEvent
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


def _event(command="/status", *, profile=None):
    return MessageEvent(
        text=command,
        source=SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
            profile=profile,
        ),
        allow_gateway_control=True,
    )


def _runner(*, allowed):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        multiplex_profiles=True,
        platforms={
            Platform.WEBHOOK: PlatformConfig(
                enabled=True,
                extra={
                    "allow_admin_from": ["admin-user"],
                    "user_allowed_commands": allowed,
                },
            )
        },
    )
    runner._startup_restore_in_progress = False
    runner._startup_restore_release_event = asyncio.Event()
    runner._startup_restore_release_event.set()
    runner._is_user_authorized_for_source = lambda source: True
    return runner


class _Adapter(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, *args, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return None


@pytest.mark.asyncio
async def test_adapter_command_preflight_fails_closed_when_gateway_did_not_install_it():
    adapter = _Adapter(PlatformConfig(enabled=True), Platform.WEBHOOK)

    with pytest.raises(RuntimeError, match="command preflight"):
        await adapter.preflight_gateway_command(_event())


@pytest.mark.asyncio
async def test_profile_command_preflight_stamps_profile_and_applies_command_policy(
    tmp_path, monkeypatch
):
    import hermes_cli.profiles

    monkeypatch.setattr(hermes_cli.profiles, "get_profile_dir", lambda _: tmp_path)
    runner = _runner(allowed=["usage"])
    seen = []

    def authorized(source):
        seen.append((source.profile, Path(get_hermes_home())))
        return True

    runner._is_user_authorized_for_source = authorized
    result = await runner._make_profile_command_preflight("fitness")(_event())

    assert "⛔ /status is admin-only here" in result
    assert seen == [("fitness", tmp_path)]


@pytest.mark.asyncio
async def test_default_profile_command_preflight_preserves_transport_auth_and_routed_scope(
    tmp_path,
):
    runner = _runner(allowed=["status"])
    runner._profile_name_for_source = lambda source: "fitness"
    runner._resolve_profile_home_for_source = lambda source: tmp_path
    seen = []

    def authorized(source):
        seen.append(
            (
                source.profile,
                Path(source._authorization_profile_home),
                Path(get_hermes_home()),
            )
        )
        return True

    runner._is_user_authorized_for_source = authorized
    event = _event()
    result = await runner._make_default_profile_command_preflight()(event)

    assert result is None
    assert event.source.profile == "fitness"
    assert seen == [("fitness", Path(get_hermes_home()), tmp_path)]


@pytest.mark.asyncio
async def test_command_preflight_waits_for_startup_restore_release_without_queueing():
    runner = _runner(allowed=["status"])
    runner._set_startup_restore_in_progress(True)
    event = _event()

    task = asyncio.create_task(runner._gateway_command_preflight(event))
    await asyncio.sleep(0)

    assert not task.done()
    assert getattr(runner, "_startup_restore_queue", []) == []

    runner._set_startup_restore_in_progress(False)
    assert await asyncio.wait_for(task, 0.2) is None


@pytest.mark.asyncio
async def test_busy_status_is_permission_checked_before_status_handler(monkeypatch):
    from gateway.platforms.base import build_session_key
    from hermes_cli import lifecycle

    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *args, **kwargs: [])
    runner = _runner(allowed=["usage"])
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._peek_session_state = lambda key: None
    runner._handle_status_command = AsyncMock(return_value="secret status")
    source = _event().source
    key = build_session_key(source)
    agent = MagicMock()
    agent.get_activity_summary.return_value = {"seconds_since_activity": 0}
    runner._running_agents = {key: agent}
    runner._running_agents_ts = {key: 0}
    runner._is_session_running = lambda candidate: candidate == key

    result = await runner._handle_message(_event())

    assert "⛔ /status is admin-only here" in result
    runner._handle_status_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_preflight_restores_home_config_and_secrets_across_profiles(tmp_path, monkeypatch):
    from agent.secret_scope import get_secret, is_multiplex_active, set_multiplex_active
    from hermes_cli.config import load_config_readonly

    root = tmp_path / '.hermes'
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    homes = {}
    for name in ('a', 'b'):
        home = root / 'profiles' / name
        home.mkdir(parents=True)
        (home / '.env').write_text(f'SLACK_BOT_TOKEN=test-{name}\n')
        (home / 'config.yaml').write_text(f'model:\n  default: model-{name}\n')
        homes[name] = home

    runner = _runner(allowed=['status'])
    seen = []

    def authorized(source):
        seen.append((source.profile, Path(get_hermes_home()),
                     load_config_readonly()['model']['default'], get_secret('SLACK_BOT_TOKEN')))
        return True

    runner._is_user_authorized_for_source = authorized
    handlers = {name: runner._make_profile_command_preflight(name) for name in homes}
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for name in ('a', 'b', 'a'):
            assert await handlers[name](_event()) is None
            assert Path(get_hermes_home()) == root
    finally:
        set_multiplex_active(previous)
    assert seen == [(name, homes[name], f'model-{name}', f'test-{name}') for name in ('a', 'b', 'a')]
