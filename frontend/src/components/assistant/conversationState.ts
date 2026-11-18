import { ref } from "vue";
import type { AssistantResponse } from "../../api";

export interface AssistantMessage {
  role: "assistant" | "user";
  text: string;
  source?: string;
  model?: string;
  modelLabel?: string;
  warning?: string;
  executions?: AssistantResponse["executions"];
  actions?: Array<{ label: string; target: string }>;
}

// The full assistant page and the in-page sidebar share one conversation.
export const assistantSending = ref(false);
export const assistantInput = ref("");
export const assistantConversationId = ref<string | undefined>();
export const assistantTaskId = ref("");
function initialMessages(): AssistantMessage[] {
  return [{
    role: "assistant",
    text: "你好，这里是慧眼AI。我会结合地震、测井与井轨迹数据诊断质量问题、解释预处理告警，并在执行转换前说明依据。",
  }];
}
export const assistantMessages = ref<AssistantMessage[]>(initialMessages());

interface AssistantRequest {
  requestId: string;
  taskId: string;
  conversationId?: string;
  generation: number;
  controller: AbortController;
}

let generation = 0;
let resultNavigation = false;
let activeRequest: AssistantRequest | undefined;
let uncertainRequest: { signature: string; requestId: string } | undefined;

/** Start a fresh conversation without changing the current task binding. */
export function resetConversation(): void {
  generation += 1;
  activeRequest?.controller.abort();
  activeRequest = undefined;
  uncertainRequest = undefined;
  assistantConversationId.value = undefined;
  assistantSending.value = false;
  assistantInput.value = "";
  assistantMessages.value = initialMessages();
}

/** Presentation navigation of an already owner-checked result keeps this conversation. */
export function beginAssistantResultNavigation(): boolean {
  if (assistantSending.value || resultNavigation) return false;
  resultNavigation = true;
  return true;
}
export function endAssistantResultNavigation(): void { resultNavigation = false; }

export function bindAssistantTask(taskId: string): void {
  if (assistantTaskId.value === taskId) return;
  if (resultNavigation && !assistantSending.value) {
    generation += 1;
    activeRequest?.controller.abort();
    activeRequest = undefined;
    uncertainRequest = undefined;
    assistantTaskId.value = taskId;
    return;
  }
  const hadConversation = assistantMessages.value.length > 1 || assistantSending.value;
  assistantTaskId.value = taskId;
  resetConversation();
  if (hadConversation) {
    assistantMessages.value.push({
      role: "assistant",
      text: "当前任务已切换，已开始新对话。",
    });
  }
}

export function beginAssistantRequest(message: string, taskId: string): AssistantRequest | undefined {
  bindAssistantTask(taskId);
  const text = message.trim();
  if (!text || assistantSending.value) return;
  const signature = JSON.stringify([taskId, assistantConversationId.value || "", text]);
  if (uncertainRequest?.signature !== signature) uncertainRequest = { signature, requestId: crypto.randomUUID() };
  const request = {
    requestId: uncertainRequest.requestId,
    taskId,
    conversationId: assistantConversationId.value,
    generation,
    controller: new AbortController(),
  };
  activeRequest = request;
  assistantMessages.value.push({ role: "user", text });
  assistantInput.value = "";
  assistantSending.value = true;
  return request;
}

export function isCurrentAssistantRequest(request: AssistantRequest): boolean {
  return activeRequest === request && generation === request.generation
    && assistantTaskId.value === request.taskId;
}

// Only these backend-owned messages may be displayed; never render arbitrary
// upstream exception text, which can contain credentials or private paths.
const safeWarnings = new Set([
  "本轮操作已到等待时限，请查看执行记录后继续查询。",
  "模型回复达到长度限制，请缩小本次操作范围。",
  "本轮工具调用达到上限，请查看已有执行记录后继续。",
  "模型工具请求格式无效，本轮未执行新的操作。",
  "此请求已登记，未重复执行；请查看操作状态。",
  "盖亚对话配置无效",
  "盖亚对话暂不可用",
  "盖亚本地 MiniMax 配置无效，请检查后端配置文件和已启用的模型。",
  "MiniMax 回复超过长度限制，请缩短问题后重试。",
  "MiniMax 凭据无效或当前模型未获授权，请检查盖亚的模型配置。",
  "MiniMax 当前限流或额度不足，请稍后重试。",
  "MiniMax 服务暂时不可用，请稍后重试。",
  "连接 MiniMax 超时或网络不可用，请稍后重试。",
  "MiniMax 未返回有效回答，请重试。",
  "LLM未启用或未完成配置",
  "本地备用模型调用失败，请检查后端服务后重试。",
]);

function publicWarning(warning: string | undefined, provider: string, agent = false): string | undefined {
  if (!warning?.trim() && provider !== "local") return;
  const reasons = new Set((warning || "").split(";").map((part) => {
    const text = part.trim();
    return safeWarnings.has(text) ? text : "模型服务暂不可用，请稍后重试或检查后端配置。";
  }));
  if (agent) return [...reasons].join("\n");
  const mode = provider === "local"
    ? "AI 模型暂不可用，本次为本地规则回复。"
    : "首选 AI 服务暂不可用，本次使用备用服务。";
  return `${mode}\n${[...reasons].join("\n")}`;
}

export function acceptAssistantResponse(request: AssistantRequest, response: AssistantResponse): void {
  if (!isCurrentAssistantRequest(request)) return;
  if (response.conversation_id && response.request_state !== "in_progress") assistantConversationId.value = response.conversation_id;
  if (response.request_state !== "in_progress") uncertainRequest = undefined;
  assistantMessages.value.push({
    role: "assistant",
    text: response.answer,
    source: response.source,
    model: response.provider === "local" ? undefined : response.model,
    modelLabel: response.provider === "gaia_minimax" ? "接口返回模型" : "配置模型",
    warning: publicWarning(response.warning, response.provider, response.executions !== undefined),
    actions: response.actions,
    executions: response.executions,
  });
}

export function rejectAssistantRequest(request: AssistantRequest, error: unknown): void {
  if (!isCurrentAssistantRequest(request)) return;
  assistantMessages.value.push({
    role: "assistant",
    text: error instanceof Error ? error.message : "助手请求失败，请检查后端连接。",
    source: "请求异常",
  });
}

export function finishAssistantRequest(request: AssistantRequest): boolean {
  if (!isCurrentAssistantRequest(request)) return false;
  assistantSending.value = false;
  return true;
}
