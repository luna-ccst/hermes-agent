"""Origin defaults are durable initial choices, never per-turn routing."""
import pytest
import yaml

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.session import SessionSource, SessionStore


def source(platform="telegram", chat_id="chat"):
    return SessionSource(platform=Platform(platform), chat_id=chat_id,
                         user_id="user", chat_type="dm")


def test_yaml_platform_default_is_persisted_at_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "legacy-model"},
        "gateway": {"new_session_models": {
            "default": "general-model", "telegram": "origin-model"}},
    }))
    config = load_gateway_config()
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source())
    assert entry.model_override == {"model": "origin-model"}
    assert SessionStore(tmp_path / "sessions", config).get_model_override(
        entry.session_key) == {"model": "origin-model"}


def test_reset_pins_latest_default_not_previous_explicit_choice(tmp_path):
    config = GatewayConfig(new_session_models={"default": "general-model"})
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source())
    store.set_model_override(entry.session_key, {"model": "explicit-model"})
    config.new_session_models["default"] = "updated-model"
    reset = store.reset_session(entry.session_key)
    assert reset.session_id != entry.session_id
    assert reset.model_override == {"model": "updated-model"}
    assert SessionStore(tmp_path / "sessions", config).get_model_override(
        entry.session_key) == {"model": "updated-model"}


@pytest.mark.parametrize("key", ["chat", "thread", "parent"])
def test_channel_choice_wins_over_seeded_platform_default(tmp_path, key):
    config = GatewayConfig.from_dict({
        "new_session_models": {"default": "general-model"},
        "platforms": {"telegram": {"channel_overrides": {
            key: {"model": "channel-model", "provider": "channel-provider"}}}},
    })
    origin = source()
    origin.thread_id = "thread"
    origin.parent_chat_id = "parent"
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(origin)
    assert entry.model_override == {
        "model": "channel-model", "provider": "channel-provider"}


@pytest.mark.parametrize("write_json", [True, False])
def test_defaults_and_explicit_choices_survive_reuse_and_restart(tmp_path, write_json):
    config = GatewayConfig(new_session_models={"default": "initial-model"},
                           write_sessions_json=write_json)
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source())
    config.new_session_models["default"] = "changed-model"
    assert store.get_or_create_session(source()).model_override == {"model": "initial-model"}
    assert store.get_or_create_session(source(chat_id="other")).model_override == {
        "model": "changed-model"}
    store.set_model_override(entry.session_key, {"model": "explicit-model"})
    restarted = SessionStore(tmp_path / "sessions", config)
    resumed = restarted.get_or_create_session(source())
    assert resumed.session_id == entry.session_id
    assert resumed.model_override == {"model": "explicit-model"}


def test_legacy_existing_session_is_not_reseeded(tmp_path):
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source())
    assert entry.model_override is None
    config.new_session_models = {"default": "new-model"}
    restarted = SessionStore(tmp_path / "sessions", config)
    resumed = restarted.get_or_create_session(source())
    assert resumed.session_id == entry.session_id
    assert resumed.model_override is None


@pytest.mark.parametrize("reset_kind", ["force", "suspended"])
def test_automatic_and_forced_new_sessions_take_current_default(tmp_path, reset_kind):
    config = GatewayConfig(new_session_models={"default": "initial-model"})
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source())
    if reset_kind == "suspended":
        store.suspend_session(entry.session_key)
    config.new_session_models["default"] = "updated-model"
    reset = store.get_or_create_session(source(), force_new=reset_kind == "force")
    assert reset.session_id != entry.session_id
    assert reset.model_override == {"model": "updated-model"}


def test_plugin_origin_uses_platform_key_not_delivery_destination(tmp_path, monkeypatch):
    from gateway.platform_registry import platform_registry

    monkeypatch.setattr(platform_registry, "is_registered", lambda name: name == "linear")
    config = GatewayConfig.from_dict({"new_session_models": {
        "default": "gpt-5.6-sol", "linear": "gpt-6-astra"}})
    store = SessionStore(tmp_path / "sessions", config)
    assert store.get_or_create_session(source("linear")).model_override == {"model": "gpt-6-astra"}
    assert store.get_or_create_session(source("slack")).model_override == {"model": "gpt-5.6-sol"}


def test_legacy_reset_without_origin_uses_recorded_platform(tmp_path):
    config = GatewayConfig(new_session_models={
        "default": "general-model", "telegram": "origin-model"})
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source())
    entry.origin = None
    reset = store.reset_session(entry.session_key)
    assert reset is not None
    assert reset.model_override == {"model": "origin-model"}


@pytest.mark.parametrize("raw", [None, [], "model", {"telegram": None},
                                      {"telegram": 42}, {"telegram": "  "}])
def test_invalid_defaults_keep_legacy_behavior(tmp_path, raw):
    config = GatewayConfig.from_dict({"gateway": {"new_session_models": raw}})
    assert config.new_session_models == {}
    assert SessionStore(tmp_path / "sessions", config).get_or_create_session(source()).model_override is None


def test_config_round_trip_and_nested_form():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["new_session_models"] == {}
    config = GatewayConfig.from_dict({"gateway": {"new_session_models": {
        "default": "general-model", "plugin-origin": "origin-model"}}})
    assert GatewayConfig.from_dict(config.to_dict()).new_session_models == config.new_session_models


def test_new_field_does_not_shift_positional_config_arguments():
    triggers = ["/new"]
    config = GatewayConfig({}, triggers)
    assert config.reset_triggers is triggers
    assert config.new_session_models == {}


def test_persisted_default_rehydrates_and_explicit_runtime_choice_wins(tmp_path):
    from gateway.run import GatewayRunner

    config = GatewayConfig(new_session_models={"default": "seeded-model"})
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source())
    runner = object.__new__(GatewayRunner)
    runner.session_store = SessionStore(tmp_path / "sessions", config)
    runner._session_model_overrides = {}
    runner._rehydrate_session_model_override(entry.session_key)
    assert runner._apply_session_model_override(entry.session_key, "legacy-model", {})[0] == "seeded-model"
    runner._session_model_overrides[entry.session_key] = {"model": "explicit-model"}
    runner._rehydrate_session_model_override(entry.session_key)
    assert runner._apply_session_model_override(entry.session_key, "legacy-model", {})[0] == "explicit-model"


def test_provider_only_channel_keeps_runtime_bundled_model(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    config = GatewayConfig.from_dict({
        "new_session_models": {"default": "general-model"},
        "platforms": {"telegram": {"channel_overrides": {
            "chat": {"provider": "channel-provider"}}}},
    })
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source())
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = store
    runner.config = config
    runner._session_model_overrides = {}
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "global-provider", "api_key": "test-only"})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", lambda provider: {
        "provider": provider, "api_key": "test-only", "model": "bundled-model"})
    monkeypatch.setattr(gateway_run, "_credential_pool_for_provider", lambda provider: None)
    model, runtime = runner._resolve_session_agent_runtime(
        source=source(), session_key=entry.session_key,
        user_config={"model": {"default": "legacy-model"}})
    assert (model, runtime["provider"]) == ("bundled-model", "channel-provider")


@pytest.mark.parametrize("origin,expected", [("telegram", "origin-model"), ("slack", "general-model")])
def test_real_config_store_to_runtime_resolution(tmp_path, monkeypatch, origin, expected):
    import gateway.run as gateway_run
    from hermes_cli.config import set_config_value

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    set_config_value("model.default", "legacy-model")
    set_config_value("gateway.new_session_models.default", "general-model")
    set_config_value("gateway.new_session_models.telegram", "origin-model")
    config = load_gateway_config()
    store = SessionStore(tmp_path / "sessions", config)
    event_source = source(origin)
    entry = store.get_or_create_session(event_source)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = SessionStore(tmp_path / "sessions", config)
    runner.config = config
    runner._session_model_overrides = {}
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "openai-codex", "api_key": "test-only",
        "model": "provider-bundled-model"})
    user_config = {"model": {"default": "legacy-model"}}
    model, runtime = runner._resolve_session_agent_runtime(
        source=event_source, session_key=entry.session_key, user_config=user_config)
    assert (model, runtime["provider"]) == (expected, "openai-codex")
    # Explicit /model's existing persistence/runtime path replaces the seed.
    runner.session_store.set_model_override(entry.session_key, {"model": "explicit-model"})
    runner._session_model_overrides[entry.session_key] = {"model": "explicit-model"}
    model, _ = runner._resolve_session_agent_runtime(
        source=event_source, session_key=entry.session_key, user_config=user_config)
    assert model == "explicit-model"
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["model"]["default"] == "legacy-model"
