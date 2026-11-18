<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";

import { chatWithAssistant, type AssistantExecution } from "../../api";
import { assistantTaskResultReference, type AssistantTaskReference } from "../../assistantTaskResult";
import {
  assistantInput as input,
  assistantMessages as messages,
  assistantSending as sending,
  bindAssistantTask,
  resetConversation,
  beginAssistantRequest,
  isCurrentAssistantRequest,
  acceptAssistantResponse,
  rejectAssistantRequest,
  finishAssistantRequest,
} from "./conversationState";

interface Props {
  visible: boolean;
  closable?: boolean;
  taskId?: string;
  contextLabel: string;
  llmAvailable: boolean;
  llmModel?: string;
  canOpenTaskResults?: boolean;
  resultOpening?: boolean;
  resultError?: string;
}

const props = defineProps<Props>();
const emit = defineEmits<{ navigate: [target: string]; openTaskResult: [reference: AssistantTaskReference]; close: [] }>();

function openTaskResult(receipt: AssistantExecution) {
  const reference = assistantTaskResultReference(receipt);
  if (reference && props.canOpenTaskResults && !props.resultOpening && !sending.value) emit("openTaskResult", reference);
}

const attachmentOpen = ref(false);
const executionLabels: Record<string, string> = {
  completed: "已完成", running: "执行中", blocked: "需要处理", failed: "未完成",
};
const operationLabels: Record<string, string> = {
  dvc_list: "核查资料", dvc_get: "读取资料", project_list: "查看工区",
  project_create: "创建工区", curve_mapping: "检查测井曲线", curve_preview: "查看测井曲线",
  data_prepare: "准备数据", registration: "井震标定", sample_build: "构建样本",
  sample_building: "构建样本", p21_reconstruction: "三维属性重建",
  prediction: "模型推理", task_status: "核查任务进度", visualization: "准备结果视图",
};
function readableAnswer(text: string) {
  return text
    .replace(/\b(?:[0-9a-f]{32,64}|(?:hp_|dv_|j_)[0-9a-f]{16,64})\b/gi, "相关记录")
    .replace(/\b(?:task_id|revision_id|execution_id|job_id)\b/g, "记录")
    .replace(/\bviewer_available\b/g, "视图状态")
    .replace(/\bpreview\b/g, "预览")
    .replace(/\btruncated\b/g, "内容截取状态");
}
function executionLabel(execution: AssistantExecution) {
  return `${operationLabels[execution.operation] || "处理任务"} · ${executionLabels[execution.status] || "状态待确认"}`;
}
function lastResult(executions: AssistantExecution[]): AssistantExecution | undefined {
  return [...executions].reverse().find((execution) => !!assistantTaskResultReference(execution));
}
function resultActionLabel(executions: AssistantExecution[]): string {
  const result = lastResult(executions);
  return result?.operation === "p21_reconstruction"
    || (result?.operation === "visualization" && result.result?.viewer_kind === "p21_query_points")
    ? "查看三维结果 →" : "查看结果 →";
}
function startNewConversation() {
  resetConversation();
  attachmentOpen.value = false;
  void nextTick(() => inputRef.value?.focus());
}
function openLastResult(executions: AssistantExecution[]) {
  const result = lastResult(executions);
  if (result) openTaskResult(result);
}
function handleInputKey(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    void sendMessage();
  }
}

const messagesRef = ref<HTMLElement | null>(null);
const inputRef = ref<HTMLTextAreaElement | null>(null);

const quickPrompts = [
  { label: "诊断当前数据", prompt: "检查当前任务的数据质量，并告诉我最需要先处理的问题" },
  { label: "解释处理告警", prompt: "解释当前预处理告警的原因、影响和建议处理方案" },
  { label: "推荐下一步", prompt: "结合当前工作区和任务状态，告诉我下一步应该做什么" },
];

const modelLabel = computed(() => {
  if (!props.llmAvailable) return "本地助手";
  const model = props.llmModel?.trim() || "AI 助手";
  return /deepseek/i.test(model) ? "DeepSeek" : model;
});

function scrollToBottom() {
  const container = messagesRef.value;
  if (container) container.scrollTop = container.scrollHeight;
}

function navigate(target: string) {
  emit("navigate", target);
}

async function sendMessage(message = input.value) {
  const text = message.trim();
  const request = beginAssistantRequest(text, props.taskId || "");
  if (!request) return;
  await nextTick();
  if (!isCurrentAssistantRequest(request)) return;
  scrollToBottom();
  try {
    const response = await chatWithAssistant(text, request.taskId || undefined, request.conversationId, request.controller.signal, request.requestId);
    acceptAssistantResponse(request, response);
  } catch (error) {
    rejectAssistantRequest(request, error);
  } finally {
    if (finishAssistantRequest(request)) {
      await nextTick();
      if (isCurrentAssistantRequest(request) && props.visible) {
        scrollToBottom();
        inputRef.value?.focus();
      }
    }
  }
}

watch(
  () => props.taskId,
  (taskId) => bindAssistantTask(taskId || ""),
  { immediate: true, flush: "sync" },
);

watch(
  () => props.visible,
  async (visible) => {
    if (!visible) return;
    await nextTick();
    scrollToBottom();
    inputRef.value?.focus();
  },
);
</script>

<template>
  <section class="strata-assistant-workspace" aria-label="慧眼助手">
    <header class="strata-assistant-header">
      <span class="strata-assistant-avatar" aria-hidden="true">
        <svg viewBox="0 0 44 44"><path d="M4 22c4.8-8 11-12 18-12s13.2 4 18 12c-4.8 8-11 12-18 12S8.8 30 4 22Z" /><circle cx="22" cy="22" r="8.5" /><path d="M7 22h5l2-3 2.5 7 2.5-10 2.5 12 2.5-9 2.5 6 2.5-3h9" /></svg>
      </span>
      <div class="strata-assistant-identity"><h2>慧眼助手</h2><span>地震与测井工作伙伴</span></div>
      <span class="assistant-model" :title="llmModel || modelLabel">{{ modelLabel }}</span>
      <button type="button" class="assistant-new-chat" aria-label="新建对话" title="新建对话" @click="startNewConversation"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5.5h16v11H9l-5 3v-14Zm8 3v5m-2.5-2.5h5" /></svg></button>
      <button v-if="closable" type="button" class="assistant-close" aria-label="关闭慧眼助手" @click="emit('close')">×</button>
    </header>

    <section ref="messagesRef" class="strata-assistant-messages" aria-live="polite" :aria-busy="sending">
      <p v-if="resultError" class="assistant-result-error" role="alert">{{ resultError }}</p>
      <article v-for="(message, index) in messages" :key="index" :class="message.role">
        <span v-if="message.role === 'assistant'" class="message-avatar" aria-hidden="true">✦</span>
        <div class="message-bubble">
          <p>{{ message.role === "assistant" ? readableAnswer(message.text) : message.text }}</p>
          <p v-if="message.warning" class="assistant-warning" role="status">{{ message.warning }}</p>
          <details v-if="message.executions?.length" class="assistant-steps">
            <summary>查看处理步骤 <span>{{ message.executions.length }} 步</span></summary>
            <ol><li v-for="(execution, n) in message.executions" :key="n" :class="execution.status">
              <span aria-hidden="true">{{ execution.status === 'completed' ? '✓' : execution.status === 'running' ? '…' : '!' }}</span>
              {{ executionLabel(execution) }}
            </li></ol>
          </details>
          <button v-if="canOpenTaskResults && message.executions?.length && lastResult(message.executions)" class="assistant-result-link" type="button" :disabled="resultOpening || sending" @click="openLastResult(message.executions)">{{ resultOpening ? '正在打开…' : resultActionLabel(message.executions) }}</button>
          <nav v-if="message.actions?.length" class="strata-assistant-message-actions" aria-label="建议操作">
            <button v-for="action in message.actions" :key="action.target" type="button" @click="navigate(action.target)">{{ action.label }} <span>→</span></button>
          </nav>
        </div>
      </article>
      <article v-if="sending" class="assistant pending"><span class="message-avatar" aria-hidden="true">✦</span><div class="message-bubble"><p><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span> 正在处理</p></div></article>
    </section>

    <section v-if="messages.length < 2" class="strata-assistant-prompts" aria-label="快捷提问">
      <button v-for="prompt in quickPrompts" :key="prompt.label" type="button" :disabled="sending" @click="sendMessage(prompt.prompt)">{{ prompt.label }}</button>
    </section>

    <div class="composer-area">
      <section v-show="attachmentOpen" class="attachment-panel" aria-label="附件与任务" @keydown.esc.stop="attachmentOpen = false">
        <slot name="attachments"><p>当前入口暂不支持附件。</p></slot>
      </section>
      <form class="strata-assistant-composer" @submit.prevent="sendMessage()">
        <button type="button" class="attachment-toggle" aria-label="添加附件" :aria-expanded="attachmentOpen" @click="attachmentOpen = !attachmentOpen">+</button>
        <textarea ref="inputRef" v-model="input" :disabled="sending" aria-label="输入需求" rows="2" placeholder="输入需求，如：检查测井曲线或运行 P21 重建…" @keydown="handleInputKey"></textarea>
        <button class="send-button" type="submit" :disabled="sending || !input.trim()" aria-label="发送">↑</button>
      </form>
    </div>
  </section>
</template>

<style scoped>
.strata-assistant-workspace { display:flex; flex-direction:column; height:100%; min-height:0; overflow:hidden; color:var(--ink, #152234); background:var(--panel, #fff); }
.strata-assistant-header { display:flex; align-items:center; gap:10px; padding:14px 16px; border-bottom:1px solid var(--line-soft, #e9edf2); }
.strata-assistant-avatar { display:grid; width:38px; height:38px; flex:0 0 38px; place-items:center; color:#167778; background:#eaf6f4; border-radius:12px; }
.strata-assistant-avatar svg { width:29px; height:29px; fill:none; stroke:currentColor; stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; }
.strata-assistant-identity { min-width:0; flex:1; }
.strata-assistant-identity h2 { margin:0; font-size:16px; line-height:1.3; }
.strata-assistant-identity > span { color:var(--muted, #6f7e91); font-size:12px; }
.assistant-new-chat { display:grid; width:30px; height:30px; flex:0 0 30px; place-items:center; color:#627785; background:transparent; border:0; border-radius:8px; cursor:pointer; }
.assistant-new-chat:hover { background:#edf3f5; }
.assistant-new-chat svg { width:19px; height:19px; fill:none; stroke:currentColor; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.assistant-close { display:grid; width:30px; height:30px; flex:0 0 30px; place-items:center; color:#627785; font-size:22px; background:transparent; border:0; border-radius:8px; cursor:pointer; }
.assistant-close:hover { background:#edf3f5; }
.assistant-model { max-width:115px; overflow:hidden; padding:3px 8px; color:#327277; font-size:12px; text-overflow:ellipsis; white-space:nowrap; background:#eef7f6; border-radius:999px; }
.strata-assistant-messages { display:flex; min-height:0; flex:1; gap:18px; flex-direction:column; overflow-y:auto; padding:20px 16px; scrollbar-width:thin; }
.strata-assistant-messages > article { display:flex; width:100%; max-width:100%; gap:9px; align-items:flex-start; }
.strata-assistant-messages > article.user { justify-content:flex-end; }
.message-avatar { display:grid; width:28px; height:28px; flex:0 0 28px; place-items:center; color:#247b7e; background:#e9f5f2; border-radius:9px; }
.message-bubble { min-width:0; max-width:calc(100% - 37px); }
.message-bubble > p { margin:0; color:inherit; font-size:14px; line-height:1.7; white-space:pre-wrap; overflow-wrap:anywhere; }
.user .message-bubble { padding:9px 12px; background:#eef4f9; border-radius:14px 14px 4px 14px; }
.assistant-warning,.assistant-result-error { margin-top:8px !important; color:#a04d15 !important; font-size:12px !important; }
.assistant-steps { margin-top:10px; color:#607488; font-size:12px; }
.assistant-steps summary { width:fit-content; padding:4px 8px; background:#f1f5f7; border-radius:999px; cursor:pointer; list-style:none; }
.assistant-steps summary::-webkit-details-marker { display:none; }
.assistant-steps summary span { margin-left:5px; color:#8794a0; }
.assistant-steps ol { display:grid; gap:5px; margin:8px 0 0 7px; padding:0; list-style:none; }
.assistant-steps li { display:flex; gap:8px; align-items:center; line-height:1.5; }
.assistant-steps li span { color:#208677; }
.assistant-steps li.failed span,.assistant-steps li.blocked span { color:#b95542; }
.assistant-result-link,.strata-assistant-message-actions button { margin-top:10px; padding:7px 10px; color:#166d76; font-size:12px; font-weight:650; background:#eef8f6; border:1px solid #cde6e1; border-radius:9px; cursor:pointer; }
.strata-assistant-message-actions { display:flex; gap:6px; flex-wrap:wrap; }
.strata-assistant-prompts { display:flex; gap:7px; overflow-x:auto; padding:8px 12px; }
.strata-assistant-prompts button { flex:0 0 auto; padding:6px 9px; color:#456779; font-size:12px; background:#f3f7f8; border:1px solid #dce9e9; border-radius:999px; cursor:pointer; }
.composer-area { position:relative; flex:0 0 auto; padding:10px 12px 14px; background:var(--panel, #fff); }
.attachment-panel { position:absolute; z-index:5; right:12px; bottom:calc(100% + 3px); left:12px; max-height:min(60vh, 420px); overflow-y:auto; padding:12px; background:#fff; border:1px solid #dbe7eb; border-radius:14px; box-shadow:0 12px 35px rgb(19 48 65 / 15%); }
.attachment-panel > p { margin:0; color:#647889; font-size:13px; }
.strata-assistant-composer { display:flex; gap:8px; align-items:flex-end; padding:9px; background:#f8fafb; border:1px solid #dce5e9; border-radius:16px; }
.strata-assistant-composer:focus-within { border-color:#5ca5ad; box-shadow:0 0 0 3px rgb(56 139 145 / 11%); }
.strata-assistant-composer textarea { width:100%; min-width:0; min-height:42px; max-height:120px; flex:1; padding:5px 0; resize:vertical; color:var(--ink, #152234); font:inherit; font-size:14px; line-height:1.45; background:transparent; border:0; outline:0; }
.strata-assistant-composer textarea::placeholder { color:#91a0ac; }
.attachment-toggle,.send-button { display:grid; width:34px; height:34px; flex:0 0 34px; place-items:center; padding:0; border:0; border-radius:50%; cursor:pointer; }
.attachment-toggle { color:#426278; font-size:25px; background:transparent; }
.attachment-toggle:hover { background:#e8f0f2; }
.send-button { color:#fff; font-size:20px; font-weight:700; background:#176e78; }
.send-button:disabled { cursor:not-allowed; opacity:.4; }
button:focus-visible,summary:focus-visible,textarea:focus-visible { outline:2px solid #297f98; outline-offset:2px; }
.thinking-dot { display:inline-block; width:5px; height:5px; margin-right:3px; background:#478b8d; border-radius:50%; animation:thinking 1s infinite ease-in-out; }
.thinking-dot:nth-child(2) { animation-delay:.14s; }.thinking-dot:nth-child(3) { animation-delay:.28s; }
@keyframes thinking { 0%,70%,100% { opacity:.35; transform:translateY(0); } 35% { opacity:1; transform:translateY(-2px); } }
@media (prefers-reduced-motion:reduce) { .thinking-dot { animation:none; } }
</style>
