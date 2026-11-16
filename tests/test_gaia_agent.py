"""Provider-contract tests: actual calls, scoped ownership, partial failures."""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest

from well_seismic.gaia_agent import chat_with_huiyan_agent
from well_seismic.gaia_minimax import GaiaMiniMaxError, GaiaMiniMaxSettings

SETTINGS = GaiaMiniMaxSettings("unit-provider", "MiniMax-M2.7", "unit-only-key")
SUBJECT = "a" * 64


def harness(tmp_path):
    h = Mock()
    h.root = tmp_path
    h.catalog.return_value = {
        "actions": [
            {
                "name": "project_list",
                "description": "list owned projects",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
        ]
    }
    h.execute.return_value = {
        "execution_id": "hx_" + "1" * 32,
        "operation": "project_list",
        "status": "completed",
        "result": {"projects": []},
    }
    return h


def response(content, stop="end_turn"):
    return {"model": "MiniMax-M2.7", "content": content, "stop_reason": stop}


def text(value):
    return response([{"type": "text", "text": value}])


def call(name="project_list", args=None, id="call1"):
    return {"type": "tool_use", "id": id, "name": name, "input": args or {}}


def run(h, **kwargs):
    return chat_with_huiyan_agent(
        SETTINGS,
        subject=SUBJECT,
        question="请列出我的项目",
        context={},
        conversation_id=None,
        harness=h,
        **kwargs,
    )


def test_real_tool_receipt_returns_to_model_and_ui_with_full_blocks(tmp_path):
    h = harness(tmp_path)
    thought = {"type": "thinking", "thinking": "opaque-private-reasoning"}
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[response([thought, call()], "tool_use"), text("当前没有项目。")],
    ) as request:
        result = run(h)
    h.execute.assert_called_once()
    assert h.execute.call_args.args[:3] == (SUBJECT, "project_list", {})
    UUID(h.execute.call_args.args[3])
    second = request.call_args.args[1]
    assert second["messages"][1]["content"][0] == thought
    assert second["messages"][2]["content"][0]["tool_use_id"] == "call1"
    assert "projects" in second["messages"][2]["content"][0]["content"]
    assert result["executions"] == [h.execute.return_value]
    assert "opaque-private-reasoning" not in str(result)
    assert SETTINGS.api_key not in str(result)


def test_same_tool_signature_does_not_duplicate_mutation(tmp_path):
    h = harness(tmp_path)
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[
            response([call()], "tool_use"),
            response([call(id="call2")], "tool_use"),
            text("完成"),
        ],
    ):
        result = run(h)
    assert h.execute.call_count == 1
    assert len(result["executions"]) == 1


def test_provider_failure_after_execution_keeps_receipt(tmp_path):
    h = harness(tmp_path)
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[
            response([call()], "tool_use"),
            GaiaMiniMaxError("连接 MiniMax 超时或网络不可用，请稍后重试。"),
        ],
    ):
        result = run(h)
    assert result["executions"][0]["status"] == "completed"
    assert result["warning"]


def test_session_cannot_be_reused_by_another_subject(tmp_path):
    h = harness(tmp_path)
    with patch(
        "well_seismic.gaia_agent.request_minimax_message", return_value=text("ok")
    ):
        first = run(h)
        rejected = chat_with_huiyan_agent(
            SETTINGS,
            subject="b" * 64,
            question="list",
            context={},
            conversation_id=UUID(first["conversation_id"]),
            harness=h,
        )
    assert rejected["request_state"] == "finished"
    assert rejected["error"]["code"] == "conversation_unavailable"
    assert rejected["executions"] == []
    h.execute.assert_not_called()


def test_unknown_tool_is_never_executed(tmp_path):
    h = harness(tmp_path)
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[
            response([call("shell", {"cmd": "bad"})], "tool_use"),
            text("不能执行"),
        ],
    ) as request:
        run(h)
    h.execute.assert_not_called()
    assert request.call_args.args[1]["messages"][-1]["content"][0]["is_error"]


def test_malformed_or_excessive_calls_execute_nothing(tmp_path):
    h = harness(tmp_path)
    for blocks in ([call(id="../escape")], [call(id=str(i)) for i in range(13)]):
        with patch(
            "well_seismic.gaia_agent.request_minimax_message",
            return_value=response(blocks, "tool_use"),
        ):
            result = run(h)
        assert result["warning"]
    h.execute.assert_not_called()


def test_missing_authentication_never_calls_provider(tmp_path):
    with (
        patch("well_seismic.gaia_agent.request_minimax_message") as request,
        pytest.raises(GaiaMiniMaxError),
    ):
        chat_with_huiyan_agent(
            SETTINGS,
            subject="visitor",
            question="run",
            context={},
            conversation_id=None,
            harness=harness(tmp_path),
        )
    request.assert_not_called()


def test_repeated_request_reuses_durable_response_without_execution(tmp_path):
    from uuid import uuid4

    h = harness(tmp_path)
    request_id = uuid4()
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[response([call()], "tool_use"), text("完成")],
    ) as request:
        first = run(h, request_id=request_id)
        second = run(h, request_id=request_id)
    assert first == second
    assert h.execute.call_count == 1
    assert request.call_count == 2


def test_non_string_tool_name_cannot_crash_or_execute(tmp_path):
    h = harness(tmp_path)
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[response([call(["bad"])], "tool_use"), text("不能执行")],
    ):
        result = run(h)
    h.execute.assert_not_called()
    assert result["executions"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "state, expected",
    [
        ("blocked", "受阻，未完成"),
        ("running", "运行中，尚未完成"),
        ("failed", "失败，未完成"),
    ],
)
def test_provider_cannot_claim_success_over_noncompleted_receipt(
    tmp_path, state, expected
):
    h = harness(tmp_path)
    h.execute.return_value = {
        "execution_id": "hx_" + "2" * 32,
        "operation": "project_list",
        "status": state,
        "result": {},
        "error": {"code": "model_assets_missing", "message": "缺少原始模型权重。"},
    }
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[
            response([call()], "tool_use"),
            text("所有任务成功完成，预测已验收。"),
            text("继续"),
        ],
    ) as provider:
        result = run(h)
        chat_with_huiyan_agent(
            SETTINGS,
            subject=SUBJECT,
            question="现在状态如何",
            context={},
            conversation_id=UUID(result["conversation_id"]),
            harness=h,
        )
    assert expected in result["answer"]
    assert "缺少原始模型权重" in result["answer"]
    assert "预测已验收" not in result["answer"]
    assert "预测已验收" not in str(provider.call_args.args[1]["messages"])
    assert result["executions"] == [h.execute.return_value]


def test_partial_run_keeps_completed_items_and_rejects_false_overall_success(tmp_path):
    h = harness(tmp_path)
    completed = dict(h.execute.return_value)
    blocked = {
        "operation": "project_list",
        "status": "blocked",
        "result": {},
        "error": {"code": "missing", "message": "尚未提供必需资产。"},
    }
    h.execute.side_effect = [completed, blocked]
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[
            response([call(), call(args={"second": True}, id="call2")], "tool_use"),
            text("全部操作均已成功。"),
        ],
    ):
        result = run(h)
    assert "1. project_list：已完成" in result["answer"]
    assert "2. project_list：受阻，未完成" in result["answer"]
    assert "尚未提供必需资产" in result["answer"]
    assert "全部操作均已成功" not in result["answer"]
    assert result["executions"] == [completed, blocked]


def test_completed_explanation_and_plain_qa_are_preserved(tmp_path):
    h = harness(tmp_path)
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[
            response([call()], "tool_use"),
            text("当前没有项目，请先创建工程。"),
            text("井震标定用于建立测井深度与地震时间的对应关系。"),
        ],
    ):
        completed = run(h)
        qa = run(h)
    assert "project_list：已完成" in completed["answer"]
    assert "当前没有项目，请先创建工程。" in completed["answer"]
    assert qa["answer"] == "井震标定用于建立测井深度与地震时间的对应关系。"
    assert qa["executions"] == []


@pytest.mark.parametrize(
    "result, expected",
    [
        ({"native_status": "running"}, "原生任务：运行中，尚未完成"),
        ({"task": {"status": "failed"}}, "原生任务：失败，未完成"),
        (
            {
                "task": {
                    "status": "completed",
                    "result": {"registration": {"business_status": "blocked"}},
                }
            },
            "标定业务状态：受阻，未完成",
        ),
    ],
)
def test_completed_query_never_claims_underlying_task_completed(
    tmp_path, result, expected
):
    h = harness(tmp_path)
    # A large native result takes the bounded provider-context path too.
    h.execute.return_value = {
        "operation": "task_status",
        "status": "completed",
        "result": {**result, "large": "x" * 13000},
    }
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[
            response([call()], "tool_use"),
            text("预测和标定均已全部成功。"),
        ],
    ):
        answer = run(h)["answer"]
    assert "task_status：已完成" in answer
    assert expected in answer
    assert "预测和标定均已全部成功" not in answer
