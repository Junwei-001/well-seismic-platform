import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";
import { webcrypto } from "node:crypto";
import ts from "typescript";
import * as vue from "vue";
import { renderToString } from "vue/server-renderer";

const stateSource = await readFile(new URL("../src/components/assistant/conversationState.ts", import.meta.url), "utf8");
const componentSource = await readFile(new URL("../src/components/assistant/FloatingAssistant.vue", import.meta.url), "utf8");
const componentScript = componentSource.match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
const componentTemplate = componentSource.match(/<template>([\s\S]*?)<\/template>/)[1];
const taskReferenceSource = await readFile(new URL("../src/assistantTaskResult.ts", import.meta.url), "utf8");
const flush = () => new Promise(setImmediate);

// Execute the production state and SFC script with real Vue refs/watchers. Only
// the HTTP boundary is replaced; no browser, network or duplicated state logic.
function evaluate(source, imports, globals = {}) {
  const module = { exports: {} };
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  });
  vm.runInNewContext(outputText, {
    module, exports: module.exports, Error, AbortController, crypto: webcrypto,
    require: (name) => {
      assert.ok(Object.hasOwn(imports, name), `Unexpected import: ${name}`);
      return imports[name];
    },
    ...globals,
  });
  return module.exports;
}

function harness(t, taskId = "task-a") {
  const state = evaluate(stateSource, { vue });
  const props = vue.reactive({ taskId, canOpenTaskResults: false, resultOpening: false, resultError: "", visible: true, contextLabel: "测试工作区", llmAvailable: true, llmModel: "MiniMax-M2.7" });
  const calls = [], events = [];
  const tickBarriers = [];
  const componentVue = { ...vue, nextTick: () => tickBarriers.shift()?.promise ?? vue.nextTick() };
  const scope = vue.effectScope();
  t.after(() => scope.stop());
  const api = {
    chatWithAssistant(...args) {
      return new Promise((resolve, reject) => calls.push({ args, resolve, reject }));
    },
  };
  const component = scope.run(() => evaluate(componentScript + `
    exports.sendMessage = sendMessage;
    exports.openTaskResult = openTaskResult;
    exports.bindings = { ...props, connectionLabel, quickPrompts, messages, input, sending, navigate, sendMessage, messagesRef, inputRef, assistantTaskResultReference, openTaskResult };
    exports.inputRef = inputRef;
  `, { vue: componentVue, "../../api": api, "./conversationState": state, "../../assistantTaskResult": evaluate(taskReferenceSource, {}) }, {
    defineProps: () => props,
    defineEmits: () => (...args) => events.push(args),
  }));
  function pauseNextTick() {
    let resume;
    const promise = new Promise((resolve) => { resume = resolve; });
    tickBarriers.push({ promise });
    return resume;
  }
  return { state, props, calls, events, pauseNextTick, ...component };
}

function response(answer, conversation_id = "session-a", extra = {}) {
  return { answer, conversation_id, source: "盖亚 MiniMax（复用配置）", provider: "gaia_minimax", model: "MiniMax-M2.7", actions: [], request_id: "", ...extra };
}

test("successful turns reuse the session and duplicate sends are blocked", async (t) => {
  const h = harness(t);
  const first = h.sendMessage("first question");
  await flush();
  await h.sendMessage("duplicate");
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].args[1], "task-a");
  assert.equal(h.calls[0].args[2], undefined);
  h.calls[0].resolve(response("first answer"));
  await first;
  const second = h.sendMessage("follow up");
  await flush();
  assert.equal(h.calls[1].args[2], "session-a");
  h.calls[1].resolve(response("second answer"));
  await second;
  assert.equal(h.state.assistantMessages.value.at(-1).text, "second answer");
  assert.equal(h.state.assistantMessages.value.at(-1).warning, undefined);
  assert.equal(h.state.assistantSending.value, false);
  h.props.taskId = "task-b";
  assert.equal(h.state.assistantConversationId.value, undefined);
  assert.ok(!h.state.assistantMessages.value.some((message) => message.text === "first answer"));
});

test("a task switch before nextTick never sends the question to either task", async (t) => {
  const h = harness(t);
  const pending = h.sendMessage("old question");
  h.props.taskId = "task-b";
  await pending;
  assert.equal(h.calls.length, 0);
  assert.equal(h.state.assistantSending.value, false);
  assert.ok(!h.state.assistantMessages.value.some((message) => message.text === "old question"));
});

for (const outcome of ["success", "error"]) {
  for (const returnToOriginalTask of [false, true]) {
    test(`late ${outcome} cannot affect a new request after ${returnToOriginalTask ? "A → B → A" : "A → B"}`, async (t) => {
      const h = harness(t);
      const old = h.sendMessage("old question");
      await flush();
      const oldSignal = h.calls[0].args[3];
      assert.equal(oldSignal.aborted, false);
      // The first request has no session ID yet.
      h.props.taskId = "task-b";
      if (returnToOriginalTask) h.props.taskId = "task-a";
      assert.equal(oldSignal.aborted, true);
      assert.equal(h.state.assistantSending.value, false);
      const current = h.sendMessage("new question");
      await flush();
      assert.equal(h.calls[1].args[2], undefined);
      if (outcome === "success") h.calls[0].resolve(response("stale answer", "stale-session"));
      else h.calls[0].reject(new Error("stale error"));
      await old;
      assert.equal(h.state.assistantSending.value, true, "old finally must not finish the new request");
      assert.equal(h.state.assistantConversationId.value, undefined);
      assert.ok(!h.state.assistantMessages.value.some((message) => /old question|stale/.test(message.text)));
      h.calls[1].resolve(response("current answer", "current-session"));
      await current;
      assert.equal(h.state.assistantSending.value, false);
      assert.equal(h.state.assistantConversationId.value, "current-session");
      assert.equal(h.state.assistantMessages.value.at(-1).text, "current answer");
    });
  }
}

test("a late error after switching with no new send remains invisible", async (t) => {
  const h = harness(t);
  const pending = h.sendMessage("old question");
  await flush();
  h.props.taskId = "task-b";
  h.calls[0].reject(new Error("old transport error"));
  await pending;
  assert.equal(h.state.assistantSending.value, false);
  assert.equal(h.state.assistantConversationId.value, undefined);
  assert.ok(!h.state.assistantMessages.value.some((message) => message.source === "请求异常"));
});

test("old post-render cleanup does not focus or finish a newer request", async (t) => {
  const h = harness(t);
  let focused = 0;
  h.inputRef.value = { focus: () => { focused += 1; } };
  const old = h.sendMessage("first");
  await flush();
  const resumeCleanup = h.pauseNextTick();
  h.calls[0].resolve(response("first answer"));
  await flush(); // old response accepted; its finally is awaiting the held nextTick
  assert.equal(h.state.assistantSending.value, false);
  const current = h.sendMessage("second");
  await flush();
  resumeCleanup();
  await old;
  assert.equal(focused, 0);
  assert.equal(h.state.assistantSending.value, true);
  h.calls[1].resolve(response("second answer"));
  await current;
  assert.equal(focused, 1);
});

test("local fallback visibly shows safe reasons and never echoes unknown warning details", async (t) => {
  const h = harness(t);
  const pending = h.sendMessage("check current task");
  await flush();
  h.calls[0].resolve(response("当前任务有一项告警。", null, {
    provider: "local", source: "本地工作流助手", model: "workflow-context-v1",
    warning: "MiniMax 当前限流或额度不足，请稍后重试。; api_key=FAKE_SECRET_ONLY C:\\private\\config.json <script>alert(1)</script>",
  }));
  await pending;
  const message = h.state.assistantMessages.value.at(-1);
  assert.match(message.warning, /本次为本地规则回复/);
  assert.match(message.warning, /限流或额度不足/);
  assert.doesNotMatch(message.warning, /FAKE_SECRET_ONLY|private|script/);
  assert.equal(message.model, undefined);
  const html = await renderToString(vue.createSSRApp({
    setup: () => h.bindings,
    render: vue.compile(componentTemplate),
  }));
  assert.match(html, /class="assistant-warning" role="status"/);
  assert.match(html, /本次为本地规则回复/);
  assert.doesNotMatch(html, /FAKE_SECRET_ONLY|private|<script>/);
});

test("older responses without warning remain usable and local replies remain identifiable", async (t) => {
  const h = harness(t);
  const pending = h.sendMessage("question");
  await flush();
  h.calls[0].resolve(response("local guidance", null, { provider: "local", source: "本地工作流助手" }));
  await pending;
  assert.match(h.state.assistantMessages.value.at(-1).warning, /本地规则回复/);
});


test("transport retry reuses the operation identity until its receipt is final", async (t) => {
  const h = harness(t);
  const first = h.sendMessage("map RHOB to DEN");
  await flush();
  const requestId = h.calls[0].args[4];
  assert.match(requestId, /^[0-9a-f-]{36}$/);
  h.calls[0].reject(new Error("transport interrupted"));
  await first;
  const retry = h.sendMessage("map RHOB to DEN");
  await flush();
  assert.equal(h.calls[1].args[4], requestId);
  h.calls[1].resolve(response("already executing", "server-session", {request_state: "in_progress", executions: []}));
  await retry;
  assert.equal(h.state.assistantConversationId.value, undefined, "progress must not change the retry signature");
  const poll = h.sendMessage("map RHOB to DEN");
  await flush();
  assert.equal(h.calls[2].args[4], requestId);
  assert.equal(h.calls[2].args[2], undefined);
  h.calls[2].resolve(response("mapping completed", "server-session", {request_state: "finished", executions: []}));
  await poll;
  const next = h.sendMessage("map RHOB to DEN");
  await flush();
  assert.notEqual(h.calls[3].args[4], requestId);
  assert.equal(h.calls[3].args[2], "server-session");
  h.calls[3].resolve(response("new confirmed turn"));
  await next;
});

test("receipt navigation emits only checked IDs and is unavailable outside the Gaia workspace", t => {
  const h = harness(t);
  const receipt = { operation: "sample_build", status: "completed", project_id: "hp_" + "a".repeat(32), task_id: "b".repeat(32),
    result: {}, viewer_url: "https://untrusted.invalid", owner: "another-user" };
  h.openTaskResult(receipt);
  assert.equal(h.events.length, 0);
  h.props.canOpenTaskResults = true;
  h.openTaskResult(receipt);
  assert.equal(h.events.length, 1);
  assert.equal(h.events[0][0], "openTaskResult");
  assert.deepEqual(JSON.parse(JSON.stringify(h.events[0][1])), { projectId: receipt.project_id, taskId: receipt.task_id });
  h.props.resultOpening = true;
  h.openTaskResult(receipt);
  h.props.resultOpening = false;
  h.state.assistantSending.value = true;
  h.openTaskResult(receipt);
  h.state.assistantSending.value = false;
  h.openTaskResult({ ...receipt, status: "blocked" });
  h.openTaskResult({ ...receipt, task_id: "https://untrusted.invalid" });
  assert.equal(h.events.length, 1);
});
