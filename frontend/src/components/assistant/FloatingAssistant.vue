<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";

import { chatWithAssistant } from "../../api";

interface Props {
  visible: boolean;
  taskId?: string;
  contextLabel: string;
  llmAvailable: boolean;
  llmModel?: string;
}

interface AssistantMessage {
  role: "assistant" | "user";
  text: string;
  source?: string;
  actions?: Array<{ label: string; target: string }>;
}

const props = defineProps<Props>();
const emit = defineEmits<{ navigate: [target: string] }>();

const sending = ref(false);
const input = ref("");
const messagesRef = ref<HTMLElement | null>(null);
const inputRef = ref<HTMLTextAreaElement | null>(null);

const messages = ref<AssistantMessage[]>([
  {
    role: "assistant",
    text: "你好，这里是慧眼AI。我会结合地震、测井与井轨迹数据诊断质量问题、解释预处理告警，并在执行转换前说明依据。",
  },
]);

const quickPrompts = [
  { label: "诊断当前数据", prompt: "检查当前任务的数据质量，并告诉我最需要先处理的问题" },
  { label: "解释处理告警", prompt: "解释当前预处理告警的原因、影响和建议处理方案" },
  { label: "推荐下一步", prompt: "结合当前工作区和任务状态，告诉我下一步应该做什么" },
];

const connectionLabel = computed(() =>
  props.llmAvailable ? `模型已连接${props.llmModel ? ` · ${props.llmModel}` : ""}` : "本地规则模式",
);

function scrollToBottom() {
  const container = messagesRef.value;
  if (container) container.scrollTop = container.scrollHeight;
}

function navigate(target: string) {
  emit("navigate", target);
}

async function sendMessage(message = input.value) {
  const text = message.trim();
  if (!text || sending.value) return;
  messages.value.push({ role: "user", text });
  input.value = "";
  sending.value = true;
  await nextTick();
  scrollToBottom();
  try {
    const response = await chatWithAssistant(text, props.taskId || undefined);
    messages.value.push({
      role: "assistant",
      text: response.answer,
      source: response.source,
      actions: response.actions,
    });
  } catch (error) {
    messages.value.push({
      role: "assistant",
      text: error instanceof Error ? error.message : "助手请求失败，请检查后端连接。",
      source: "请求异常",
    });
  } finally {
    sending.value = false;
    await nextTick();
    scrollToBottom();
    inputRef.value?.focus();
  }
}

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
  <section class="strata-assistant-workspace" aria-labelledby="strata-assistant-title">
    <header class="strata-assistant-header">
      <span class="strata-assistant-avatar" aria-hidden="true">
        <svg viewBox="0 0 44 44">
          <path class="eye-outline" d="M4 22c4.8-8 11-12 18-12s13.2 4 18 12c-4.8 8-11 12-18 12S8.8 30 4 22Z" />
          <circle class="eye-iris" cx="22" cy="22" r="8.5" />
          <path class="seismic" d="M7 22h5l2-3 2.5 7 2.5-10 2.5 12 2.5-9 2.5 6 2.5-3h9" />
          <path class="well" d="M22 12.5v12c0 3.4 1.8 5.5 5.4 7" />
          <circle class="sweet-halo" cx="27.4" cy="31.5" r="3.5" />
          <circle class="sweet-spot" cx="27.4" cy="31.5" r="1.7" />
        </svg>
      </span>
      <div class="strata-assistant-identity">
        <span>HUIYAN INTELLIGENCE</span>
        <div><h1 id="strata-assistant-title">慧眼AI</h1><strong>井震多模态研判</strong></div>
        <p>围绕当前数据、任务证据和处理状态连续对话</p>
      </div>
      <div class="strata-assistant-connection" role="status">
        <i :class="{ online: llmAvailable }" aria-hidden="true"></i>
        <div><span>研判服务</span><strong>{{ connectionLabel }}</strong></div>
      </div>
    </header>

    <section class="strata-assistant-context" aria-label="井震研判当前上下文">
      <div><span>当前工作区</span><strong>{{ contextLabel }}</strong></div>
      <nav aria-label="上下文状态">
        <span>{{ taskId ? `关联任务 ${taskId.slice(0, 8)}` : "尚未绑定任务" }}</span>
        <span>原始数据留在本地</span>
        <span>回答保留证据边界</span>
      </nav>
    </section>

    <section ref="messagesRef" class="strata-assistant-messages" aria-live="polite" :aria-busy="sending">
      <article v-for="(message, index) in messages" :key="index" :class="message.role">
        <span class="message-avatar" aria-hidden="true">
          <svg v-if="message.role === 'assistant'" viewBox="0 0 28 28"><path d="M4 9c4-3 7 2 10 0s6-3 10 0M4 14c4-3 7 2 10 0s6-3 10 0M4 19c4-3 7 2 10 0s6-3 10 0" /><path d="M14 5v15c0 2 2 3 5 4" /></svg>
          <b v-else>你</b>
        </span>
        <div class="message-bubble">
          <p>{{ message.text }}</p>
          <small v-if="message.source">来源 · {{ message.source }}</small>
          <nav v-if="message.actions?.length" class="strata-assistant-message-actions">
            <button v-for="action in message.actions" :key="action.target" type="button" @click="navigate(action.target)">{{ action.label }} <span>→</span></button>
          </nav>
        </div>
      </article>
      <article v-if="sending" class="assistant pending">
        <span class="message-avatar" aria-hidden="true"><svg viewBox="0 0 28 28"><path d="M4 9c4-3 7 2 10 0s6-3 10 0M4 14c4-3 7 2 10 0s6-3 10 0M4 19c4-3 7 2 10 0s6-3 10 0" /><path d="M14 5v15c0 2 2 3 5 4" /></svg></span>
        <div class="message-bubble"><p><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span> 正在结合任务摘要研判</p></div>
      </article>
    </section>

    <section class="strata-assistant-prompts" aria-label="快捷提问">
      <div><strong>建议从这里开始</strong><span>点击即可提问</span></div>
      <nav>
        <button v-for="prompt in quickPrompts" :key="prompt.label" type="button" :disabled="sending" @click="sendMessage(prompt.prompt)">{{ prompt.label }} <span>↗</span></button>
      </nav>
    </section>

    <form class="strata-assistant-composer" @submit.prevent="sendMessage()">
      <label>
        <textarea ref="inputRef" v-model="input" :disabled="sending" placeholder="输入关于数据质量、标定告警、模型结果或下一步的问题…" @keydown.ctrl.enter.prevent="sendMessage()"></textarea>
        <small>Ctrl + Enter 发送</small>
      </label>
      <button type="submit" :disabled="sending || !input.trim()"><span>{{ sending ? "研判中" : "发送" }}</span><svg viewBox="0 0 18 18" aria-hidden="true"><path d="m4 9 9-5-3 10-2-4-4-1Z" /></svg></button>
    </form>
  </section>
</template>

<style scoped>
.strata-assistant-workspace {
  display: grid;
  width: 100%;
  height: 100%;
  min-height: 560px;
  grid-template-rows: auto auto minmax(0, 1fr) auto auto;
  overflow: hidden;
  background: #fff;
  border: 1px solid #d8e2ea;
  border-radius: 18px;
  box-shadow: 0 18px 52px rgb(35 70 100 / 10%);
}

.strata-assistant-header {
  display: grid;
  grid-template-columns: 58px minmax(0, 1fr) auto;
  gap: 16px;
  align-items: center;
  padding: 20px 24px;
  background: linear-gradient(120deg, #f8fcff 0%, #f1f8fc 54%, #f2faf7 100%);
  border-bottom: 1px solid #dbe7ed;
}

.strata-assistant-avatar {
  display: grid;
  width: 56px;
  height: 56px;
  place-items: center;
  color: #147c79;
  background: #fff;
  border: 1px solid #cfe0e6;
  border-radius: 17px;
  box-shadow: 0 8px 22px rgb(30 91 107 / 11%);
}
.strata-assistant-avatar svg { width: 42px; height: 42px; overflow: visible; fill: none; stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; }
.strata-assistant-avatar .eye-outline { stroke: #355f78; stroke-width: 1.45; }
.strata-assistant-avatar .eye-iris { stroke: #8ebfc7; stroke-width: 1; stroke-dasharray: 2 2; }
.strata-assistant-avatar .seismic { stroke: #158994; stroke-width: 1.45; }
.strata-assistant-avatar .well { stroke: #173b54; stroke-width: 1.55; }
.strata-assistant-avatar .sweet-halo { stroke: #d19a36; stroke-width: 1; opacity: .5; }
.strata-assistant-avatar .sweet-spot { fill: #ce9026; stroke: #fff; stroke-width: .7; }

.strata-assistant-identity { min-width: 0; }
.strata-assistant-identity > span { color: #358397; font-size: 12px; font-weight: 760; letter-spacing: .11em; }
.strata-assistant-identity > div { display: flex; gap: 12px; align-items: baseline; margin-top: 2px; }
.strata-assistant-identity h1 { margin: 0; color: #16354d; font-size: 26px; font-weight: 780; letter-spacing: -.02em; }
.strata-assistant-identity strong { color: #60788a; font-size: 14px; font-weight: 650; }
.strata-assistant-identity p { margin: 2px 0 0; color: #728493; font-size: 13px; }

.strata-assistant-connection {
  display: flex;
  min-width: 220px;
  gap: 10px;
  align-items: center;
  padding: 10px 13px;
  background: rgb(255 255 255 / 78%);
  border: 1px solid #d9e5ea;
  border-radius: 12px;
}
.strata-assistant-connection > i { width: 9px; height: 9px; flex: 0 0 9px; background: #c99e48; border-radius: 50%; box-shadow: 0 0 0 4px rgb(201 158 72 / 12%); }
.strata-assistant-connection > i.online { background: #22ae7b; box-shadow: 0 0 0 4px rgb(34 174 123 / 12%); }
.strata-assistant-connection div { display: grid; min-width: 0; }
.strata-assistant-connection span { color: #84939f; font-size: 12px; }
.strata-assistant-connection strong { overflow: hidden; color: #345267; font-size: 13px; font-weight: 680; text-overflow: ellipsis; white-space: nowrap; }

.strata-assistant-context {
  display: flex;
  gap: 18px;
  align-items: center;
  justify-content: space-between;
  padding: 11px 24px;
  background: #fbfcfd;
  border-bottom: 1px solid #e3eaee;
}
.strata-assistant-context > div { display: flex; min-width: 0; gap: 9px; align-items: baseline; }
.strata-assistant-context > div span { color: #84919c; font-size: 12px; font-weight: 650; }
.strata-assistant-context > div strong { overflow: hidden; color: #314f64; font-size: 14px; text-overflow: ellipsis; white-space: nowrap; }
.strata-assistant-context nav { display: flex; gap: 7px; flex-wrap: wrap; justify-content: flex-end; }
.strata-assistant-context nav span { padding: 4px 9px; color: #607588; font-size: 12px; background: #f2f6f8; border: 1px solid #e0e8ec; border-radius: 999px; }

.strata-assistant-messages {
  display: flex;
  min-height: 0;
  gap: 18px;
  flex-direction: column;
  padding: 28px clamp(24px, 5vw, 76px);
  overflow-y: auto;
  background:
    linear-gradient(rgb(249 252 253 / 93%), rgb(248 251 252 / 97%)),
    repeating-linear-gradient(0deg, transparent 0 31px, rgb(47 111 137 / 5%) 31px 32px);
  scrollbar-color: #bacbd5 transparent;
  scrollbar-width: thin;
}
.strata-assistant-messages article { display: flex; width: fit-content; max-width: min(860px, 82%); gap: 11px; align-items: flex-start; }
.strata-assistant-messages article.user { align-self: flex-end; flex-direction: row-reverse; }
.message-avatar { display: grid; width: 34px; height: 34px; flex: 0 0 34px; place-items: center; color: #177579; background: #ddf2ed; border-radius: 11px; box-shadow: 0 4px 12px rgb(44 104 112 / 8%); }
.message-avatar svg { width: 24px; height: 24px; fill: none; stroke: currentColor; stroke-width: 1.35; stroke-linecap: round; }
.message-avatar b { color: #fff; font-size: 13px; }
.user .message-avatar { background: #547696; }
.message-bubble { padding: 13px 16px; background: #fff; border: 1px solid #dde8ec; border-radius: 5px 15px 15px; box-shadow: 0 7px 22px rgb(44 75 96 / 6%); }
.user .message-bubble { color: #fff; background: #285b7f; border-color: #285b7f; border-radius: 15px 5px 15px 15px; }
.message-bubble p { margin: 0; color: inherit; font-size: 15px; line-height: 1.7; white-space: pre-wrap; }
.message-bubble > small { display: block; margin-top: 7px; color: #84939f; font-size: 12px; }
.user .message-bubble > small { color: rgb(255 255 255 / 70%); }
.strata-assistant-message-actions { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 10px; }
.strata-assistant-message-actions button { padding: 6px 10px; color: #176e77; font-size: 12px; font-weight: 670; background: #f7fbfb; border: 1px solid #c8dfe1; border-radius: 8px; cursor: pointer; }
.strata-assistant-message-actions button:hover { background: #eaf6f5; border-color: #89bfc0; }
.pending .thinking-dot { display: inline-block; width: 5px; height: 5px; margin-right: 3px; background: #478b8d; border-radius: 50%; animation: thinking 1s infinite ease-in-out; }
.pending .thinking-dot:nth-child(2) { animation-delay: .14s; }
.pending .thinking-dot:nth-child(3) { animation-delay: .28s; }

.strata-assistant-prompts {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 18px;
  align-items: center;
  padding: 11px 20px;
  background: #fbfcfd;
  border-top: 1px solid #e5ebef;
}
.strata-assistant-prompts > div { display: grid; min-width: 150px; }
.strata-assistant-prompts > div strong { color: #4f6678; font-size: 13px; }
.strata-assistant-prompts > div span { color: #96a3ad; font-size: 12px; }
.strata-assistant-prompts nav { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
.strata-assistant-prompts button { min-height: 38px; padding: 7px 12px; color: #456779; font-size: 13px; font-weight: 660; background: #f1f7f8; border: 1px solid #d9e7e9; border-radius: 9px; cursor: pointer; }
.strata-assistant-prompts button:hover:not(:disabled) { color: #176c75; background: #e8f5f3; border-color: #a9ced0; }
.strata-assistant-prompts button:disabled { cursor: not-allowed; opacity: .5; }

.strata-assistant-composer { display: grid; grid-template-columns: minmax(0, 1fr) 92px; gap: 10px; padding: 13px 16px 16px; background: #fff; border-top: 1px solid #e0e7eb; }
.strata-assistant-composer label { position: relative; display: block; }
.strata-assistant-composer textarea { width: 100%; height: 78px; margin: 0; padding: 13px 14px 24px; resize: none; color: #263b4f; font: inherit; font-size: 14px; line-height: 1.5; background: #f7f9fa; border: 1px solid #d3dee5; border-radius: 12px; outline: none; }
.strata-assistant-composer textarea:focus { background: #fff; border-color: #5ca5ad; box-shadow: 0 0 0 3px rgb(56 139 145 / 11%); }
.strata-assistant-composer textarea::placeholder { color: #93a1ab; }
.strata-assistant-composer label small { position: absolute; right: 11px; bottom: 8px; color: #9ba6ae; font-size: 12px; pointer-events: none; }
.strata-assistant-composer > button { display: grid; place-content: center; gap: 5px; color: #fff; font-size: 13px; font-weight: 700; background: linear-gradient(145deg, #16727a, #1b607c); border: 0; border-radius: 11px; cursor: pointer; }
.strata-assistant-composer > button:hover:not(:disabled) { filter: brightness(1.06); }
.strata-assistant-composer > button:disabled { cursor: not-allowed; opacity: .42; }
.strata-assistant-composer > button svg { width: 20px; height: 20px; margin: 0 auto; fill: none; stroke: currentColor; stroke-width: 1.45; stroke-linejoin: round; }

button:focus-visible,
textarea:focus-visible { outline: 2px solid #297f98; outline-offset: 2px; }

@keyframes thinking {
  0%, 70%, 100% { opacity: .35; transform: translateY(0); }
  35% { opacity: 1; transform: translateY(-2px); }
}

@media (max-width: 900px) {
  .strata-assistant-workspace { min-height: 620px; }
  .strata-assistant-header { grid-template-columns: 48px minmax(0, 1fr); padding: 16px; }
  .strata-assistant-avatar { width: 46px; height: 46px; border-radius: 14px; }
  .strata-assistant-avatar svg { width: 35px; height: 35px; }
  .strata-assistant-identity h1 { font-size: 22px; }
  .strata-assistant-identity strong { display: none; }
  .strata-assistant-connection { grid-column: 1 / -1; min-width: 0; }
  .strata-assistant-context { align-items: flex-start; flex-direction: column; padding: 10px 16px; }
  .strata-assistant-context nav { justify-content: flex-start; }
  .strata-assistant-messages { padding: 22px 16px; }
  .strata-assistant-messages article { max-width: 94%; }
  .strata-assistant-prompts { grid-template-columns: 1fr; gap: 8px; }
  .strata-assistant-prompts > div { display: none; }
  .strata-assistant-prompts nav { display: flex; overflow-x: auto; }
  .strata-assistant-prompts button { min-width: 150px; }
  .strata-assistant-composer { grid-template-columns: minmax(0, 1fr) 72px; }
}

@media (prefers-reduced-motion: reduce) {
  .thinking-dot { animation: none; }
}
</style>
