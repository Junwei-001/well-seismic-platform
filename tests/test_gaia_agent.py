"""Provider-contract tests: actual calls, scoped ownership, partial failures."""

import io
import json
import urllib.error
from dataclasses import replace
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

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
        ("blocked", "暂未完成"),
        ("running", "仍在运行"),
        ("failed", "未能完成"),
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
    assert "查询工区暂未完成" in result["answer"]
    assert "project_list" not in result["answer"]
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
    assert completed["answer"] == "当前没有项目，请先创建工程。"
    assert qa["answer"] == "井震标定用于建立测井深度与地震时间的对应关系。"
    assert qa["executions"] == []


@pytest.mark.parametrize(
    "result, expected",
    [
        ({"native_status": "running"}, "当前任务仍在运行"),
        ({"task": {"status": "failed"}}, "当前任务未能完成"),
        (
            {
                "task": {
                    "status": "completed",
                    "result": {"registration": {"business_status": "blocked"}},
                }
            },
            "井震标定未通过质量检查",
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
    assert "task_status" not in answer
    assert expected in answer
    assert "预测和标定均已全部成功" not in answer


def openai_response(content=None, *, calls=None, reasoning=None, stop=None):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return io.BytesIO(json.dumps({
        "model": "deepseek-v4-flash",
        "choices": [{"message": message, "finish_reason": stop or ("tool_calls" if calls else "stop")}],
    }).encode())


def openai_call(name, args, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


@pytest.fixture
def deepseek():
    return GaiaMiniMaxSettings(
        "deepseek", "deepseek-v4-flash", "unit-only-key", protocol="openai",
        endpoint="https://api.deepseek.com/v1/chat/completions",
    )


def test_openai_tool_transcript_waits_for_p21_and_returns_its_archive(tmp_path, deepseek):
    h = harness(tmp_path)
    h.catalog.return_value["actions"].append({
        "name": "p21_reconstruction", "description": "Volve P21",
        "parameters": {"type": "object"},
    })
    project_id = "hp_" + "a" * 32
    p21 = {
        "execution_id": "hx_" + "b" * 32, "operation": "p21_reconstruction",
        "task_id": "c" * 32, "project_id": project_id, "status": "running",
        "result": {"native_task_submitted": True},
    }
    listed = {**h.execute.return_value, "result": {"projects": [{"id": project_id, "name": "Volve"}]}}
    h.execute.side_effect = [listed, p21]
    archived = {**p21, "status": "completed", "revision_id": "dv_" + "d" * 32,
                "result": {"native_status": "completed", "preview": {}, "viewer_available": True}}
    h.wait_for_fast_run.return_value = archived
    project_call = openai_call("project_list", "{}", "list-1")
    predict_call = openai_call("p21_reconstruction", json.dumps({"project_id": project_id, "fold_id": 0}), "p21-1")
    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = [
            openai_response(calls=[project_call], reasoning="opaque-project-reasoning"),
            openai_response(calls=[predict_call], reasoning="opaque-p21-reasoning"),
            openai_response("Volve P21 三维重建已完成并归档。"),
        ]
        result = chat_with_huiyan_agent(
            deepseek, subject=SUBJECT, question="请完成 Volve P21 第 0 折重建",
            context={}, conversation_id=None, harness=h,
        )
    assert result["provider"] == "gaia_openai" and result["display_label"] == "DeepSeek"
    assert result["answer"] == "Volve P21 三维重建已完成并归档。"
    assert result["executions"] == [listed, archived]
    assert "opaque-" not in str(result)
    assert h.execute.call_args.kwargs == {"defer_settlement": True}
    assert h.wait_for_fast_run.call_count == 1
    first = json.loads(opener.return_value.open.call_args_list[0].args[0].data)
    assert first["messages"][0]["role"] == "system"
    assert first["tools"][1]["function"]["name"] == "p21_reconstruction"
    last = json.loads(opener.return_value.open.call_args.args[0].data)
    assert [entry["role"] for entry in last["messages"]] == [
        "system", "user", "assistant", "tool", "assistant", "tool",
    ]
    assert last["messages"][2]["reasoning_content"] == "opaque-project-reasoning"
    assert last["messages"][4]["tool_calls"] == [predict_call]
    tool = last["messages"][-1]
    receipt = json.loads(tool["content"])
    assert tool["tool_call_id"] == "p21-1"
    assert receipt["status"] == "completed" and receipt["revision_id"] == archived["revision_id"]
    assert not {"preview", "viewer_available"} & receipt["result"].keys()
    assert "execution_id" not in receipt


@pytest.mark.parametrize("calls, stop", [
    ([openai_call("project_list", "{", "call-1")], "tool_calls"),
    ([openai_call("project_list", "{}", "same")] * 2, "tool_calls"),
    ([openai_call("project_list", "{}", "partial")], "length"),
])
def test_openai_invalid_or_truncated_tool_calls_never_execute(tmp_path, deepseek, calls, stop):
    h = harness(tmp_path)
    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = [
            openai_response(calls=calls, stop=stop),
            openai_response("已完成。"),
        ]
        result = chat_with_huiyan_agent(
            deepseek, subject=SUBJECT, question="列出工区",
            context={}, conversation_id=None, harness=h,
        )
    assert result["warning"] or result["executions"][0]["status"] == "failed"
    h.execute.assert_not_called()


def test_model_reply_keeps_prose_without_echoing_receipt_ids(tmp_path):
    h = harness(tmp_path)
    internal = "dv_" + "d" * 32
    with patch("well_seismic.gaia_agent.request_minimax_message", side_effect=[
        response([call()], "tool_use"),
        text(f"已查到工区。revision_id 为 {internal}。可继续导入测井数据。"),
    ]):
        result = run(h)
    assert result["answer"] == "已查到工区。可继续导入测井数据。"


def test_corrected_same_page_retry_uses_distinct_operation_ids_and_stays_successful(tmp_path, deepseek):
    h = harness(tmp_path)
    h.catalog.return_value["actions"][0]["parameters"]["properties"] = {
        "offset": {"type": "integer", "minimum": 0},
    }
    seen = set()

    def execute(subject, operation, arguments, operation_id):
        if operation_id in seen:
            raise HTTPException(409, "operation already registered")
        seen.add(operation_id)
        if arguments.get("offset") == -1:
            raise HTTPException(422, "invalid offset")
        return h.execute.return_value

    h.execute.side_effect = execute
    request = {
        "subject": SUBJECT, "question": "查询工区", "context": {},
        "conversation_id": None, "harness": h, "request_id": uuid4(),
    }
    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = [
            openai_response(calls=[openai_call("project_list", '{"offset":-1}', "call_0")]),
            openai_response(calls=[openai_call("project_list", '{}', "call_0")]),
            openai_response("已查询，当前账号下还没有工区。"),
        ]
        result = chat_with_huiyan_agent(deepseek, **request)
        replay = chat_with_huiyan_agent(deepseek, **request)
    assert result["answer"] == "已查询，当前账号下还没有工区。"
    assert replay == result
    assert [item["status"] for item in result["executions"]] == ["failed", "completed"]
    assert len(seen) == 2 and h.execute.call_count == 2


def test_successful_query_never_masks_invalid_arguments_for_another_task(tmp_path):
    h = harness(tmp_path)
    h.catalog.return_value["actions"] = [{
        "name": "task_status", "description": "Read an owned task",
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}},
    }]
    h.execute.side_effect = [
        HTTPException(422, "bad arguments"),
        {
            "operation": "task_status", "status": "completed",
            "task_id": "c" * 32, "result": {"task": {"status": "completed"}},
        },
    ]
    with patch("well_seismic.gaia_agent.request_minimax_message", side_effect=[
        response([call("task_status", {"task_id": "b" * 32, "unexpected": True})], "tool_use"),
        response([call("task_status", {"task_id": "c" * 32}, "call2")], "tool_use"),
        text("两个任务都查到了。"),
    ]):
        result = run(h)
    assert "当前任务未能完成" in result["answer"]
    assert "两个任务都查到了" not in result["answer"]
    assert [item["status"] for item in result["executions"]] == ["failed", "completed"]


def test_clean_reply_translates_fields_without_removing_scope_or_facts(tmp_path):
    h = harness(tmp_path)
    with patch("well_seismic.gaia_agent.request_minimax_message", side_effect=[
        text("列表 truncated，还有更多版本可继续查询。registration 已完成。P21 preview 已生成。"
             "viewer_available: false，窗口刷新后可重试。"),
    ]):
        answer = run(h)["answer"]
    assert answer == (
        "列表 未列全，还有更多版本可继续查询。井震标定 已完成。P21 预览 已生成。"
        "结果暂不可查看，窗口刷新后可重试。"
    )


def test_openai_interrupted_turn_keeps_ledger_without_synthetic_provider_history(tmp_path, deepseek):
    h = harness(tmp_path)
    thinking = replace(deepseek, extra_body={"thinking": {"type": "enabled"}})

    def ask(question, conversation_id=None):
        return chat_with_huiyan_agent(
            thinking, subject=SUBJECT, question=question, context={},
            conversation_id=conversation_id, harness=h,
        )

    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = [
            openai_response("可以查询工区。", reasoning="first-provider-reasoning"),
            openai_response(calls=[openai_call("project_list", "{}", "call1")], reasoning="interrupted-reasoning"),
            urllib.error.HTTPError(thinking.endpoint, 503, "unavailable", {}, None),
            openai_response("当前可继续查询。", reasoning="follow-up-provider-reasoning"),
        ]
        first = ask("能查询工区吗")
        conversation = UUID(first["conversation_id"])
        interrupted = ask("查询工区", conversation)
        follow_up = ask("现在可以继续吗", conversation)
    assert interrupted["warning"] and interrupted["executions"] == [h.execute.return_value]
    assert follow_up["answer"] == "当前可继续查询。"
    request = json.loads(opener.return_value.open.call_args.args[0].data)
    assert [message["role"] for message in request["messages"]] == [
        "system", "user", "assistant", "user",
    ]
    assert request["messages"][2]["reasoning_content"] == "first-provider-reasoning"
    assert "interrupted-reasoning" not in str(request)


def test_live_p21_answer_shape_keeps_facts_without_technical_receipt_noise(tmp_path):
    h = harness(tmp_path)
    h.catalog.return_value["actions"] = [{
        "name": "visualization", "description": "Check the workbench result",
        "parameters": {"type": "object"},
    }]
    task_id = "a" * 32
    api_path = f"/api/v1/gaia/harness/tasks/{task_id}/visualization"
    receipt = {
        "operation": "visualization", "status": "completed", "task_id": task_id,
        "result": {
            "viewer_url": api_path, "viewer_available": True, "viewer_kind": "p21_query_points",
            "preview": {}, "task": {"result": {"registration": {"business_status": None}}},
        },
    }
    h.execute.return_value = receipt
    reply = (
        "已全部完成。\n\n**工区**\n- 新建「Volve 演示工区」（工区 ID：对应记录），暂无其他资产登记。\n"
        "**第 0 折 P21 三维条件重建**\n- 任务 对应记录，状态：completed\n- 版本：对应记录\n"
        "**归档**\n- 版本存储状态为 archived。\n"
        "**三维视图**\n- 已生成可交互结果页面，视图类型为 P21 查询点（p21_query_points），入口位于慧眼结果页面：\n"
        "  /api/v1/gaia/harness/tasks/对应记录/结果查看\n"
        "**需要注意**\n- 返回的任务结果里，标定环节的业务状态字段为空。"
        f"当前显示第 0 折查询点，可通过[结果按钮]({api_path})查看。"
    )
    with patch("well_seismic.gaia_agent.request_minimax_message", side_effect=[
        response([call("visualization", {"task_id": task_id})], "tool_use"), text(reply),
    ]) as model:
        result = run(h)
    answer = result["answer"]
    assert all(word not in answer for word in (
        "**", "/api/", "对应记录", "completed", "archived", "p21_query_points", "业务状态字段为空", task_id,
    ))
    assert "新建「Volve 演示工区」，暂无其他资产登记。" in answer
    assert "当前显示第 0 折查询点，可通过结果按钮查看。" in answer
    assert all(word in answer for word in ("P21 查询点", "已归档", "已完成"))
    payload = model.call_args.args[1]
    tool_reply = json.loads(payload["messages"][-1]["content"][0]["content"])
    assert tool_reply["task_id"] == task_id
    assert tool_reply["result"]["可查看结果"] is True
    assert all(word not in json.dumps(tool_reply) for word in (
        "viewer_url", "viewer_kind", "business_status", "/api/", "p21_query_points",
    ))
    assert result["executions"] == [receipt]


@pytest.mark.parametrize("empty_calls", [[], None])
def test_openai_second_user_turn_replays_only_supported_assistant_fields(tmp_path, deepseek, empty_calls):
    h = harness(tmp_path)
    first_response = json.load(openai_response("工区已查询。", reasoning="opaque-first"))
    first_response["choices"][0]["message"].update({
        "tool_calls": empty_calls, "refusal": None, "annotations": [],
    })
    with patch("well_seismic.gaia_minimax.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = [
            io.BytesIO(json.dumps(first_response).encode()), openai_response("可以。"),
        ]
        first = chat_with_huiyan_agent(
            deepseek, subject=SUBJECT, question="查询工区", context={},
            conversation_id=None, harness=h,
        )
        chat_with_huiyan_agent(
            deepseek, subject=SUBJECT, question="能再查一次吗", context={},
            conversation_id=UUID(first["conversation_id"]), harness=h,
        )
    retained = json.loads(opener.return_value.open.call_args.args[0].data)["messages"][2]
    assert retained == {
        "role": "assistant", "content": "工区已查询。", "reasoning_content": "opaque-first",
    }
