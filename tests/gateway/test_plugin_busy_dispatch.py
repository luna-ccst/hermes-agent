"""Real busy gateway path preserves plugin arguments and the running agent."""
import time
from unittest.mock import MagicMock

import pytest

from hermes_cli import plugins
from gateway.session import build_session_key
from tests.gateway.test_gateway_command_dispatch_minimal import _make_event, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("async_handler", [False, True])
@pytest.mark.parametrize("policy,denied", [("dispatch", None), ("reject", None), ("dispatch", "Forbidden")])
async def test_busy_plugin_dispatch(monkeypatch, async_handler, policy, denied):
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)
    context = plugins.PluginContext(plugins.PluginManifest(name="busy-test", source="user"), manager)
    calls = []

    def sync_handler(args):
        calls.append(args)
        return "plugin output"

    async def async_fn(args):
        return sync_handler(args)

    context.register_command("busy-test", async_fn if async_handler else sync_handler, busy_policy=policy)
    runner, adapter = _make_runner()
    event = _make_event("/busy_test  some args  ")
    event.internal = False
    key = build_session_key(event.source)
    agent = MagicMock()
    agent.get_activity_summary.return_value = {"seconds_since_activity": 0}
    runner._running_agents[key] = agent
    runner._running_agents_ts[key] = time.time()
    checked = []

    def check_access(source, command):
        checked.append((source, command))
        return denied

    runner._check_slash_access = check_access
    result = await runner._handle_message(event)
    assert checked == [(event.source, "busy-test")]
    if denied:
        assert result == denied
        assert calls == []
    elif policy == "reject":
        assert "can't run mid-turn" in result
        assert calls == []
    else:
        assert result == "plugin output"
        assert calls == ["some args"]
    agent.interrupt.assert_not_called()
    assert runner._running_agents[key] is agent
    assert adapter._pending_messages == {}
