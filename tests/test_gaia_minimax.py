"""Verify Gaia config reuse, multi-turn context, and credential boundaries."""

import io
import json
import os
import urllib.error
from unittest.mock import patch
from uuid import UUID

import pytest

from well_seismic.gaia_minimax import (
    GaiaMiniMaxError,
    GaiaMiniMaxSettings,
    chat_with_gaia_minimax,
    is_model_identity_question,
)


def config_document():
    return {
        "provider_settings": {"default_provider_id": "existing/MiniMax-M2.7"},
        "provider": [{"id": "existing/MiniMax-M2.7", "provider_source_id": "existing", "type": "minimax_token_plan", "model": "MiniMax-M2.7", "enable": True}],
        "provider_sources": [{"id": "existing", "type": "minimax_token_plan", "enable": True, "key": ["unit-test-only-key"]}],
    }


def test_load_existing_source_without_exposing_key(tmp_path):
    config_path = tmp_path / "cmd_config.json"
    config_path.write_text(json.dumps(config_document()), encoding="utf-8")
    with patch.dict(os.environ, {"WELL_SEISMIC_GAIA_CONFIG_PATH": str(config_path)}, clear=True):
        settings = GaiaMiniMaxSettings.from_environment()
    assert settings.model == "MiniMax-M2.7"
    assert settings.provider_id == "existing/MiniMax-M2.7"
    assert settings.api_key == "unit-test-only-key"
    assert settings.api_key not in repr(settings)
    assert settings.api_key not in json.dumps(settings.public_status())
    assert str(config_path) not in json.dumps(settings.public_status())


@pytest.mark.parametrize("reference", ["$HUIYAN_TEST_KEY", "${HUIYAN_TEST_KEY}"])
@pytest.mark.parametrize("as_list", [True, False])
def test_existing_environment_key_reference_is_resolved_only_on_backend(tmp_path, reference, as_list):
    document = config_document()
    document["provider_sources"][0]["key"] = [reference] if as_list else reference
    config_path = tmp_path / "cmd_config.json"
    original = json.dumps(document)
    config_path.write_text(original, encoding="utf-8")
    secret = "environment-unit-test-key"
    with patch.dict(os.environ, {
        "WELL_SEISMIC_GAIA_CONFIG_PATH": str(config_path), "HUIYAN_TEST_KEY": secret,
    }, clear=True):
        settings = GaiaMiniMaxSettings.from_environment()
        with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = model_response("Hello")
            chat_with_gaia_minimax(settings, question="Hello", context={}, conversation_id=None)
        request = opener.return_value.open.call_args.args[0]
    assert request.get_header("Authorization") == f"Bearer {secret}"
    assert reference not in request.get_header("Authorization")
    assert secret not in request.data.decode()
    assert secret not in repr(settings)
    assert secret not in json.dumps(settings.public_status())
    assert config_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("value", [None, "", "   ", "private\r\nheader", "redacted", "$NESTED_TEST_KEY"])
def test_unusable_environment_key_fails_closed_without_exposing_value(tmp_path, value):
    document = config_document()
    document["provider_sources"][0]["key"] = ["${HUIYAN_TEST_KEY}"]
    config_path = tmp_path / "private-config.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    environment = {"WELL_SEISMIC_GAIA_CONFIG_PATH": str(config_path)}
    if value is not None:
        environment["HUIYAN_TEST_KEY"] = value
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(GaiaMiniMaxError) as raised:
            GaiaMiniMaxSettings.from_environment()
    public_error = str(raised.value)
    assert public_error == "助手模型配置无效，请检查后端配置和已启用的模型。"
    assert str(config_path) not in public_error
    assert "HUIYAN_TEST_KEY" not in public_error


@pytest.mark.parametrize("reference", ["$", "${}", "${HUIYAN_TEST_KEY", "$HUIYAN TEST KEY"])
def test_malformed_environment_reference_is_not_used_as_a_credential(tmp_path, reference):
    document = config_document()
    document["provider_sources"][0]["key"] = [reference]
    config_path = tmp_path / "cmd_config.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    with patch.dict(os.environ, {"WELL_SEISMIC_GAIA_CONFIG_PATH": str(config_path)}, clear=True):
        with pytest.raises(GaiaMiniMaxError):
            GaiaMiniMaxSettings.from_environment()


@pytest.mark.parametrize("disabled_section", ["provider", "provider_sources"])
def test_disabled_provider_or_source_is_not_used(tmp_path, disabled_section):
    document = config_document()
    document[disabled_section][0]["enable"] = False
    config_path = tmp_path / "cmd_config.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    with patch.dict(os.environ, {"WELL_SEISMIC_GAIA_CONFIG_PATH": str(config_path)}, clear=True):
        with pytest.raises(GaiaMiniMaxError):
            GaiaMiniMaxSettings.from_environment()


def model_response(text, model="MiniMax-M2.7"):
    return io.BytesIO(json.dumps({
        "type": "message", "model": model,
        "content": [{"type": "thinking", "thinking": "internal"}, {"type": "text", "text": text}],
    }).encode())


def test_deepseek_inherits_source_protocol_base_and_credential(tmp_path):
    document = config_document()
    document["provider"].append({
        "id": "DeepSeek V4 Flash", "provider_source_id": "deepseek",
        "model": "deepseek/deepseek-v4-flash",
        "custom_extra_body": {"reasoning": {"enabled": False}, "stream": True},
    })
    document["provider_sources"] += [
        {"type": "unrelated_source"},
        {"id": "unrelated", "type": "unrelated_source"},
        {"id": "unrelated", "type": "unrelated_source"},
        {
            "id": "deepseek", "type": "openai_chat_completion",
            "api_base": "https://proxy.example/v1/", "key": ["unit-source-key"],
            "custom_headers": {"Authorization": "$HUIYAN_TEST_AUTH"},
        },
    ]
    configuration = tmp_path / "cmd_config.json"
    configuration.write_text(json.dumps(document), encoding="utf-8")
    authorization = "UnitAuth source-authorization"
    environment = {
        "WELL_SEISMIC_GAIA_CONFIG_PATH": str(configuration),
        "WELL_SEISMIC_GAIA_PROVIDER_ID": "DeepSeek V4 Flash",
        "HUIYAN_TEST_AUTH": authorization,
    }
    with patch.dict(os.environ, environment, clear=True):
        settings = GaiaMiniMaxSettings.from_environment()
    assert settings.protocol == "openai"
    assert settings.endpoint == "https://proxy.example/v1/chat/completions"
    assert settings.public_status()["display_label"] == "DeepSeek"
    assert authorization not in repr(settings)
    assert authorization not in json.dumps(settings.public_status())
    response = io.BytesIO(json.dumps({
        "model": "deepseek-v4-flash",
        "choices": [{
            "message": {"role": "assistant", "content": "可以查看测井曲线。"},
            "finish_reason": "stop",
        }],
    }).encode())
    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.return_value = response
        answer, _, actual = chat_with_gaia_minimax(
            settings, question="检查曲线", context={}, conversation_id=None,
        )
    request = opener.return_value.open.call_args.args[0]
    payload = json.loads(request.data)
    assert request.get_header("Authorization") == authorization
    assert request.get_header("X-api-key") is None
    assert payload["reasoning"] == {"enabled": False}
    assert "stream" not in payload
    assert answer == "可以查看测井曲线。" and actual == "deepseek-v4-flash"

    document["provider_sources"].append(document["provider_sources"][-1])
    configuration.write_text(json.dumps(document), encoding="utf-8")
    with patch.dict(os.environ, environment, clear=True), pytest.raises(GaiaMiniMaxError):
        GaiaMiniMaxSettings.from_environment()


@pytest.mark.parametrize("env, model, endpoint", [
    ({"DEEPSEEK_API_KEY": "unit-origin-key"}, "deepseek-chat", "https://api.deepseek.com/v1/chat/completions"),
    ({
        "WELL_SEISMIC_LLM_API_KEY": "unit-proxy-key",
        "WELL_SEISMIC_LLM_BASE_URL": "https://proxy.example/v4/chat/completions",
        "WELL_SEISMIC_LLM_MODEL": "custom-model",
    }, "custom-model", "https://proxy.example/v4/chat/completions"),
])
def test_openai_environment_fallback(env, model, endpoint):
    with patch.dict(os.environ, env, clear=True):
        settings = GaiaMiniMaxSettings.from_environment()
    assert settings.model == model
    assert settings.endpoint == endpoint
    assert settings.api_key not in repr(settings)


@pytest.mark.parametrize("base_url", [
    "ftp://example.com/v1", "https://user:pass@example.com/v1",
    "http://example.com/v1", "https://example.com/v1?secret=key",
])
def test_openai_fallback_invalid_endpoint_is_rejected(base_url):
    with patch.dict(os.environ, {
        "WELL_SEISMIC_LLM_API_KEY": "unit-key", "WELL_SEISMIC_LLM_BASE_URL": base_url,
    }, clear=True), pytest.raises(GaiaMiniMaxError) as error:
        GaiaMiniMaxSettings.from_environment()
    assert base_url not in str(error.value)


def test_conversation_history_and_actual_model_response():
    settings = GaiaMiniMaxSettings("existing/MiniMax-M2.7", "MiniMax-M2.7", "unit-test-only-key")
    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = [model_response("Noted."), model_response("HY-7319")]
        _, session, actual_model = chat_with_gaia_minimax(settings, question="Remember HY-7319", context={}, conversation_id=None)
        answer, second_session, _ = chat_with_gaia_minimax(settings, question="What was the code?", context={}, conversation_id=UUID(session))
    request = opener.return_value.open.call_args.args[0]
    payload = json.loads(request.data)
    assert second_session == session
    assert answer == "HY-7319"
    assert actual_model == "MiniMax-M2.7"
    assert payload["messages"] == [
        {"role": "user", "content": "Remember HY-7319"},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "What was the code?"},
    ]
    assert request.full_url == "https://api.minimaxi.com/anthropic/v1/messages"
    assert request.get_header("Authorization") == "Bearer unit-test-only-key"
    assert settings.api_key not in request.data.decode()


@pytest.mark.parametrize("question", [
    "你是什么大模型", "请问你是什么模型？", "你到底用的是哪个模型？",
    "你的模型版本是什么", "查看当前模型信息", "你是MiniMax-M3.1-Flash吗？",
    "What model are you?", "Which model are you using?",
])
def test_standalone_model_queries_are_recognized(question):
    assert is_model_identity_question(question)


@pytest.mark.parametrize("question", [
    "请比较 MiniMax 和其他模型在井震解释中的用途", "什么是大语言模型？",
    "你是什么模型？顺便解释井震标定。", "Which model is best for seismic interpretation?",
])
def test_topic_and_compound_questions_remain_regular_chat(question):
    assert not is_model_identity_question(question)


def test_identity_answer_uses_verified_metadata_and_preserves_clean_history():
    settings = GaiaMiniMaxSettings("existing/MiniMax-M2.7", "MiniMax-M2.7", "unit-test-only-key")
    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = [
            model_response("I am MiniMax-M3.1-Flash; the footer says so.", model="MiniMax-response-id"),
            model_response("Use two control points."),
        ]
        answer, session, actual_model = chat_with_gaia_minimax(settings, question="你是什么大模型", context={}, conversation_id=None)
        followup, _, _ = chat_with_gaia_minimax(settings, question="Explain well ties.", context={}, conversation_id=UUID(session))
    assert "本次请求模型 ID：MiniMax-M2.7" in answer
    assert "本次接口返回模型 ID：MiniMax-response-id" in answer
    assert "M3.1-Flash" not in answer
    assert actual_model == "MiniMax-response-id"
    assert followup == "Use two control points."
    payload = json.loads(opener.return_value.open.call_args.args[0].data)
    assert payload["messages"][1]["content"] == answer


@pytest.mark.parametrize("question", ["Hello", "你是什么模型"])
def test_upstream_error_does_not_expose_key(question):
    settings = GaiaMiniMaxSettings("existing/MiniMax-M2.7", "MiniMax-M2.7", "unit-test-only-key")
    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = urllib.error.HTTPError("https://api.minimaxi.com", 401, "unit-test-only-key", {}, None)
        with pytest.raises(GaiaMiniMaxError) as raised:
            chat_with_gaia_minimax(settings, question=question, context={}, conversation_id=None)
    assert settings.api_key not in str(raised.value)


def test_assistant_route_uses_existing_config_and_sanitizes_context():
    from well_seismic import api
    from well_seismic.api_models import AssistantChatRequest
    from well_seismic.gaia_chat import GaiaChatSettings

    settings = GaiaMiniMaxSettings("existing/MiniMax-M2.7", "MiniMax-M2.7", "unit-test-only-key")
    with (
        patch.object(GaiaChatSettings, "from_environment", return_value=GaiaChatSettings("", "", "", 10)),
        patch.object(GaiaMiniMaxSettings, "from_environment", return_value=settings),
        patch.object(api, "_assistant_context", return_value={"path": "C:\\private\\data.las"}),
        patch("well_seismic.gaia_minimax._request_minimax", return_value=("Hello", "MiniMax-response-id")) as request,
    ):
        response = api.assistant_chat(AssistantChatRequest(message="Review C:\\private\\data.las"))
    assert response["provider"] == "gaia_minimax"
    assert response["model"] == "MiniMax-response-id"
    UUID(response["conversation_id"])
    assert settings.api_key not in json.dumps(response)
    assert "C:\\private" not in str(request.call_args.args[1])


def test_model_info_route_labels_backend_generated_answer():
    from well_seismic import api
    from well_seismic.api_models import AssistantChatRequest
    from well_seismic.gaia_chat import GaiaChatSettings

    settings = GaiaMiniMaxSettings("existing/MiniMax-M2.7", "MiniMax-M2.7", "unit-test-only-key")
    with (
        patch.object(GaiaChatSettings, "from_environment", return_value=GaiaChatSettings("", "", "", 10)),
        patch.object(GaiaMiniMaxSettings, "from_environment", return_value=settings),
        patch("well_seismic.gaia_minimax._request_minimax", return_value=("I am MiniMax-M3.1-Flash", "MiniMax-M2.7")),
    ):
        response = api.assistant_chat(AssistantChatRequest(message="你是什么大模型"))
    assert response["source"] == "后端模型信息（本次调用已核对）"
    assert response["model"] == "MiniMax-M2.7"
    assert "M3.1-Flash" not in response["answer"]


def test_bound_task_model_failure_exposes_a_safe_fallback_warning():
    from well_seismic import api
    from well_seismic.api_models import AssistantChatRequest
    from well_seismic.gaia_chat import GaiaChatSettings

    safe_error = "连接 MiniMax 超时或网络不可用，请稍后重试。"
    with (
        patch.object(GaiaChatSettings, "from_environment", return_value=GaiaChatSettings("", "", "", 10)),
        patch.object(GaiaMiniMaxSettings, "from_environment", side_effect=GaiaMiniMaxError(safe_error)),
        patch.object(api, "_assistant_context", return_value={"task_id": "task-a", "issues": []}),
        patch.object(api, "build_structured_generator", return_value=None),
    ):
        response = api.assistant_chat(AssistantChatRequest(message="检查数据", task_id="task-a"))
    assert response["provider"] == "local"
    assert safe_error in response["warning"]
    assert response["source"] == "本地工作流助手"


def test_fallback_generator_exception_never_returns_private_details():
    from unittest.mock import Mock
    from well_seismic import api
    from well_seismic.api_models import AssistantChatRequest
    from well_seismic.gaia_chat import GaiaChatSettings

    generator = Mock()
    generator.generate_json.side_effect = RuntimeError("Bearer private-test-key at C:\\private\\well.las")
    with (
        patch.object(GaiaChatSettings, "from_environment", return_value=GaiaChatSettings("", "", "", 10)),
        patch.object(GaiaMiniMaxSettings, "from_environment", return_value=None),
        patch.object(api, "_assistant_context", return_value={"task_id": "task-a", "issues": []}),
        patch.object(api, "build_structured_generator", return_value=generator),
    ):
        response = api.assistant_chat(AssistantChatRequest(message="检查数据", task_id="task-a"))
    assert response["provider"] == "local"
    assert response["warning"] == "本地备用模型调用失败，请检查后端服务后重试。"
    assert "private-test-key" not in json.dumps(response)
    assert "private" not in json.dumps(response)


def test_standalone_openai_chat_uses_tool_agent_with_local_owner():
    from well_seismic import api
    from well_seismic.api_models import AssistantChatRequest
    from well_seismic.gaia_chat import GaiaChatSettings

    settings = GaiaMiniMaxSettings(
        "deepseek-chat", "deepseek-chat", "unit-test-only-key", protocol="openai",
    )
    request = AssistantChatRequest(message="运行P21并查看结果")
    expected = {"answer": "已完成。", "executions": [{"operation": "p21_reconstruction", "status": "completed"}]}
    with (
        patch.object(GaiaChatSettings, "from_environment", return_value=GaiaChatSettings("", "", "", 10)),
        patch.object(GaiaMiniMaxSettings, "from_environment", return_value=settings),
        patch.object(api, "_assistant_context", return_value={}),
        patch.object(api._huiyan_harness, "assistant_context", return_value={}) as binding,
        patch("well_seismic.gaia_agent.chat_with_huiyan_agent", return_value=expected) as agent,
        patch.object(api, "chat_with_gaia_minimax") as plain_chat,
    ):
        result = api.assistant_chat(request)
    assert result == expected
    import hashlib
    from well_seismic.huiyan_harness import HuiyanHarness
    local_subject = hashlib.sha256(b"well-seismic:local-workbench").hexdigest()
    HuiyanHarness.subject(local_subject)  # Use the real unchanged identity guard.
    binding.assert_called_once_with(local_subject, None)
    assert agent.call_args.kwargs["subject"] == local_subject
    assert agent.call_args.kwargs["harness"] is api._huiyan_harness
    assert agent.call_args.kwargs["request_id"] == request.request_id
    plain_chat.assert_not_called()
