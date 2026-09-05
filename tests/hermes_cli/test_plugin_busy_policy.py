"""Plugin commands opt in to non-interrupting busy dispatch."""
import pytest

from hermes_cli import plugins
from hermes_cli.commands import resolve_command


@pytest.fixture
def context(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)
    return plugins.PluginContext(
        plugins.PluginManifest(name="busy-test", source="user"), manager
    )


def test_plugin_command_resolves_with_safe_default(context):
    context.register_command("busy-test", lambda args: args, "Test", "[args]")
    command = resolve_command("/BUSY_TEST")
    assert command is not None
    assert command.name == "busy-test"
    assert command.description == "Test"
    assert command.args_hint == "[args]"
    assert command.busy_policy == "reject"


@pytest.mark.parametrize("policy", ["reject", "dispatch"])
def test_explicit_busy_policy(context, policy):
    context.register_command("busy-test", str, busy_policy=policy)
    assert resolve_command("busy-test").busy_policy == policy


@pytest.mark.parametrize("policy", ["interrupt_then_dispatch", "unknown", None])
def test_unsafe_busy_policy_rejected(context, policy):
    with pytest.raises(ValueError, match="busy_policy"):
        context.register_command("busy-test", str, busy_policy=policy)
    assert resolve_command("busy-test") is None


@pytest.mark.parametrize("policy", ["reject", "dispatch"])
def test_adapter_guard_bypasses_without_interrupt(context, policy):
    from hermes_cli.commands import should_bypass_active_session, is_interrupt_then_dispatch
    context.register_command("busy-test", str, busy_policy=policy)
    assert should_bypass_active_session("busy_test")
    assert not is_interrupt_then_dispatch("busy_test")


def test_builtin_collision_and_replacement(context):
    original = resolve_command("status")
    assert context.register_command("status", str, busy_policy="dispatch") is None
    assert resolve_command("status") is original
    context.register_command("busy-test", str)
    context.register_command("busy-test", str, busy_policy="dispatch")
    assert resolve_command("busy-test").busy_policy == "dispatch"


def test_discovery_from_temporary_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin_dir = tmp_path / "plugins" / "busy-test"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: busy-test\nversion: 1.0.0\n")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_command('busy-test', str, busy_policy='dispatch')\n"
    )
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [busy-test]\n")
    command = resolve_command("busy-test")
    assert command is not None
    assert command.busy_policy == "dispatch"
    assert plugins.get_plugin_command_handler("busy-test")("hello") == "hello"


def test_discovery_failure_does_not_break_builtin_or_unknown(monkeypatch):
    def fail():
        raise RuntimeError("discovery unavailable")
    monkeypatch.setattr(plugins, "get_plugin_commands", fail)
    assert resolve_command("status").name == "status"
    assert resolve_command("unknown-command") is None


def test_slack_bang_command(context):
    from plugins.platforms.slack.adapter import _rewrite_known_bang_command
    from hermes_cli.commands import should_bypass_active_session, is_interrupt_then_dispatch
    context.register_command("ram", str, busy_policy="dispatch")
    assert _rewrite_known_bang_command("!ram") == "/ram"
    assert _rewrite_known_bang_command("!ram details") == "/ram details"
    assert should_bypass_active_session("ram")
    assert not is_interrupt_then_dispatch("ram")
