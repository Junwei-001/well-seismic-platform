<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

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

interface DragSession {
  pointerId: number;
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
}

const props = defineProps<Props>();
const emit = defineEmits<{ navigate: [target: string] }>();

const POSITION_STORAGE_KEY = "地层慧眼_慧眼AI浮窗位置";
const panelOpen = ref(false);
const sending = ref(false);
const dragging = ref(false);
const input = ref("");
const panelRef = ref<HTMLElement | null>(null);
const messagesRef = ref<HTMLElement | null>(null);
const inputRef = ref<HTMLTextAreaElement | null>(null);
const position = ref<{ x: number; y: number } | null>(null);
let dragSession: DragSession | null = null;

const messages = ref<AssistantMessage[]>([
  {
    role: "assistant",
    text: "你好，这里是慧眼AI。我会结合地震、测井与井轨迹数据诊断质量问题、解释预处理告警，并在执行转换前说明依据。",
  },
]);

const quickPrompts = [
  { label: "诊断数据", prompt: "检查当前任务的数据质量，并告诉我最需要先处理的问题" },
  { label: "解释告警", prompt: "解释当前预处理告警的原因、影响和建议处理方案" },
  { label: "推荐下一步", prompt: "结合当前工作区和任务状态，告诉我下一步应该做什么" },
];

const panelStyle = computed(() => {
  if (!position.value) {
    return {
      left: "auto",
      top: "auto",
      right: "var(--assistant-edge)",
      bottom: "var(--assistant-bottom)",
    };
  }
  return {
    left: `${position.value.x}px`,
    top: `${position.value.y}px`,
    right: "auto",
    bottom: "auto",
  };
});

const connectionLabel = computed(() =>
  props.llmAvailable ? `模型已连接${props.llmModel ? ` · ${props.llmModel}` : ""}` : "本地规则模式",
);

function clampPosition(x: number, y: number, width: number, height: number) {
  const margin = 12;
  const maxX = Math.max(margin, window.innerWidth - width - margin);
  const maxY = Math.max(margin, window.innerHeight - height - margin);
  return {
    x: Math.min(Math.max(x, margin), maxX),
    y: Math.min(Math.max(y, margin), maxY),
  };
}

function persistPosition() {
  if (!position.value) return;
  window.sessionStorage.setItem(POSITION_STORAGE_KEY, JSON.stringify(position.value));
}

function restorePosition() {
  try {
    const raw = window.sessionStorage.getItem(POSITION_STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw) as { x?: unknown; y?: unknown };
    if (typeof saved.x === "number" && Number.isFinite(saved.x) && typeof saved.y === "number" && Number.isFinite(saved.y)) {
      position.value = { x: saved.x, y: saved.y };
    }
  } catch {
    window.sessionStorage.removeItem(POSITION_STORAGE_KEY);
  }
}

function keepInViewport() {
  if (!position.value || !panelRef.value) return;
  const rect = panelRef.value.getBoundingClientRect();
  position.value = clampPosition(position.value.x, position.value.y, rect.width, rect.height);
  persistPosition();
}

function resetPosition() {
  position.value = null;
  window.sessionStorage.removeItem(POSITION_STORAGE_KEY);
}

function resetPositionFromHeader(event: MouseEvent) {
  if ((event.target as HTMLElement).closest("button")) return;
  resetPosition();
}

function startDrag(event: PointerEvent) {
  if (event.button !== 0 || !panelRef.value || (event.target as HTMLElement).closest("button")) return;
  const rect = panelRef.value.getBoundingClientRect();
  dragSession = {
    pointerId: event.pointerId,
    offsetX: event.clientX - rect.left,
    offsetY: event.clientY - rect.top,
    width: rect.width,
    height: rect.height,
  };
  position.value = { x: rect.left, y: rect.top };
  dragging.value = true;
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  event.preventDefault();
}

function moveDrag(event: PointerEvent) {
  if (!dragSession || event.pointerId !== dragSession.pointerId) return;
  position.value = clampPosition(
    event.clientX - dragSession.offsetX,
    event.clientY - dragSession.offsetY,
    dragSession.width,
    dragSession.height,
  );
}

function stopDrag(event?: PointerEvent) {
  if (!dragSession || (event && event.pointerId !== dragSession.pointerId)) return;
  if (event && (event.currentTarget as HTMLElement).hasPointerCapture(event.pointerId)) {
    (event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId);
  }
  dragSession = null;
  dragging.value = false;
  persistPosition();
}

function scrollToBottom() {
  const container = messagesRef.value;
  if (container) container.scrollTop = container.scrollHeight;
}

async function togglePanel() {
  panelOpen.value = !panelOpen.value;
  if (!panelOpen.value) return;
  await nextTick();
  keepInViewport();
  scrollToBottom();
  inputRef.value?.focus();
}

function closePanel() {
  panelOpen.value = false;
  dragSession = null;
  dragging.value = false;
}

function navigate(target: string) {
  emit("navigate", target);
  closePanel();
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

function handleKeydown(event: KeyboardEvent) {
  if (event.key === "Escape" && panelOpen.value) closePanel();
}

watch(
  () => props.visible,
  (visible) => {
    if (!visible) closePanel();
  },
);

onMounted(() => {
  restorePosition();
  window.addEventListener("resize", keepInViewport);
  window.addEventListener("keydown", handleKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", keepInViewport);
  window.removeEventListener("keydown", handleKeydown);
});
</script>

<template>
  <template v-if="visible">
    <button
      type="button"
      :class="['strata-assistant-launcher', { open: panelOpen }]"
      aria-controls="strata-floating-assistant"
      :aria-expanded="panelOpen"
      :aria-label="panelOpen ? '收起慧眼AI' : '打开慧眼AI'"
      @click="togglePanel"
    >
      <span class="strata-assistant-mark" aria-hidden="true">
        <svg viewBox="0 0 44 44">
          <path class="eye-outline" d="M4 22c4.8-8 11-12 18-12s13.2 4 18 12c-4.8 8-11 12-18 12S8.8 30 4 22Z" />
          <circle class="eye-iris" cx="22" cy="22" r="8.5" />
          <path class="seismic" d="M7 22h5l2-3 2.5 7 2.5-10 2.5 12 2.5-9 2.5 6 2.5-3h9" />
          <path class="well" d="M22 12.5v12c0 3.4 1.8 5.5 5.4 7" />
          <circle class="sweet-halo" cx="27.4" cy="31.5" r="3.5" />
          <circle class="sweet-spot" cx="27.4" cy="31.5" r="1.7" />
        </svg>
      </span>
      <span class="strata-assistant-launcher-copy">
        <strong>慧眼<span>AI</span></strong>
      </span>
      <i :class="['strata-assistant-presence', { online: llmAvailable }]" aria-hidden="true"></i>
    </button>

    <aside
      v-if="panelOpen"
      id="strata-floating-assistant"
      ref="panelRef"
      :class="['strata-assistant-window', { dragging }]"
      :style="panelStyle"
      role="dialog"
      aria-modal="false"
      aria-labelledby="strata-assistant-title"
    >
      <header
        class="strata-assistant-header"
        title="拖动标题栏移动；双击恢复默认位置"
        @pointerdown="startDrag"
        @pointermove="moveDrag"
        @pointerup="stopDrag"
        @pointercancel="stopDrag"
        @dblclick="resetPositionFromHeader"
      >
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
          <div><strong id="strata-assistant-title">慧眼AI</strong><span>井震多模态研判</span></div>
          <small><i :class="{ online: llmAvailable }"></i>{{ connectionLabel }}</small>
        </div>
        <div class="strata-assistant-window-actions">
          <button type="button" title="恢复默认位置" aria-label="恢复研判窗口默认位置" @click="resetPosition">
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 8V4h4M16 12v4h-4M5 5l3 3M15 15l-3-3" /></svg>
          </button>
          <button type="button" title="关闭" aria-label="关闭慧眼AI" @click="closePanel">×</button>
        </div>
        <span class="strata-assistant-drag-hint" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i>拖动</span>
      </header>

      <section class="strata-assistant-context" aria-label="井震研判当前上下文">
        <div><span>当前工作区</span><strong>{{ contextLabel }}</strong></div>
        <nav>
          <span>{{ taskId ? `任务 ${taskId.slice(0, 8)}` : "尚未绑定任务" }}</span>
          <span>原始数据留在本地</span>
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

      <section class="strata-assistant-prompts">
        <header><strong>建议从这里开始</strong><small>点击即可提问</small></header>
        <div>
          <button v-for="prompt in quickPrompts" :key="prompt.label" type="button" :disabled="sending" @click="sendMessage(prompt.prompt)">{{ prompt.label }} <span>↗</span></button>
        </div>
      </section>

      <form class="strata-assistant-composer" @submit.prevent="sendMessage()">
        <label>
          <textarea ref="inputRef" v-model="input" :disabled="sending" placeholder="输入关于数据、告警或下一步的问题…" @keydown.ctrl.enter.prevent="sendMessage()"></textarea>
          <small>Ctrl + Enter 发送</small>
        </label>
        <button type="submit" :disabled="sending || !input.trim()"><span>发送</span><svg viewBox="0 0 18 18" aria-hidden="true"><path d="m4 9 9-5-3 10-2-4-4-1Z" /></svg></button>
      </form>
    </aside>
  </template>
</template>

<style scoped>
.strata-assistant-launcher {
  position: fixed;
  right: 24px;
  bottom: 22px;
  z-index: 80;
  display: grid;
  width: 158px;
  min-height: 58px;
  grid-template-columns: 40px minmax(0, 1fr) 8px;
  gap: 10px;
  align-items: center;
  padding: 8px 10px 8px 8px;
  overflow: hidden;
  color: #18354c;
  background:
    linear-gradient(90deg, rgb(255 255 255 / 97%), rgb(246 251 252 / 95%)),
    repeating-linear-gradient(0deg, transparent 0 8px, rgb(24 91 111 / 4%) 8px 9px);
  border: 1px solid rgb(177 200 211 / 92%);
  border-radius: 15px 5px 15px 15px;
  box-shadow: 0 12px 34px rgb(24 54 82 / 15%), inset 0 1px rgb(255 255 255 / 96%);
  backdrop-filter: blur(18px) saturate(1.15);
  cursor: pointer;
  isolation: isolate;
  transition: opacity .18s ease, visibility .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}
.strata-assistant-launcher::before {
  position: absolute;
  z-index: 0;
  top: -12px;
  bottom: -12px;
  left: 0;
  width: 38px;
  background: linear-gradient(90deg, transparent, rgb(50 172 183 / 13%), transparent);
  content: "";
  transform: translateX(-58px) skewX(-13deg);
  animation: launcher-scan 5.6s ease-in-out infinite;
  pointer-events: none;
}
.strata-assistant-launcher::after {
  position: absolute;
  z-index: 1;
  top: 0;
  right: 0;
  width: 34px;
  height: 2px;
  background: linear-gradient(90deg, transparent, #2a9aa4);
  content: "";
  pointer-events: none;
}
.strata-assistant-launcher > * { position: relative; z-index: 2; }
.strata-assistant-launcher:hover { border-color: #78abb8; box-shadow: 0 15px 38px rgb(24 54 82 / 19%); transform: translateY(-1px); }
.strata-assistant-launcher.open { visibility: hidden; opacity: 0; pointer-events: none; transform: translateY(6px) scale(.94); }
.strata-assistant-mark,
.strata-assistant-avatar { display: grid; place-items: center; color: #147c79; }
.strata-assistant-mark {
  width: 40px;
  height: 40px;
  background: radial-gradient(circle at 50% 50%, #f7ffff 0 34%, #edf8f7 35% 60%, #edf4f9 100%);
  border: 1px solid #c8dfe2;
  border-radius: 11px 3px 11px 11px;
  box-shadow: inset 0 0 0 3px rgb(255 255 255 / 52%);
}
.strata-assistant-mark svg { width: 32px; height: 32px; }
.strata-assistant-mark svg,
.strata-assistant-avatar svg { overflow: visible; fill: none; stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; }
.strata-assistant-mark .eye-outline,
.strata-assistant-avatar .eye-outline { stroke: #355f78; stroke-width: 1.45; }
.strata-assistant-mark .eye-iris,
.strata-assistant-avatar .eye-iris { stroke: #9fc8cd; stroke-width: 1; stroke-dasharray: 2 2; transform-origin: 22px 22px; transition: transform .24s ease; }
.strata-assistant-launcher:hover .eye-iris { transform: rotate(20deg); }
.strata-assistant-mark .seismic,
.strata-assistant-avatar .seismic { stroke: #158994; stroke-width: 1.45; }
.strata-assistant-mark .well,
.strata-assistant-avatar .well { stroke: #173b54; stroke-width: 1.55; }
.strata-assistant-mark .sweet-halo,
.strata-assistant-avatar .sweet-halo { fill: none; stroke: #d19a36; stroke-width: 1; opacity: .42; }
.strata-assistant-mark .sweet-spot,
.strata-assistant-avatar .sweet-spot { fill: #ce9026; stroke: #fff; stroke-width: .7; }
.strata-assistant-launcher-copy { display: grid; min-width: 0; text-align: left; }
.strata-assistant-launcher-copy strong { color: #17364d; font-size: 15.5px; font-weight: 720; line-height: 1.2; white-space: nowrap; }
.strata-assistant-launcher-copy strong span { margin-left: 2px; color: #168792; font-family: "Segoe UI", sans-serif; font-size: 13px; font-weight: 780; letter-spacing: .04em; }
.strata-assistant-presence {
  width: 8px;
  height: 8px;
  background: #c99e48;
  border: 2px solid #fff;
  border-radius: 50%;
  box-shadow: 0 0 0 3px rgb(201 158 72 / 12%);
}
.strata-assistant-presence.online { background: #22ae7b; box-shadow: 0 0 0 3px rgb(34 174 123 / 12%); }

.strata-assistant-window {
  --assistant-edge: 24px;
  --assistant-bottom: 88px;
  position: fixed;
  right: 24px;
  bottom: 88px;
  z-index: 79;
  display: grid;
  width: min(410px, calc(100vw - 32px));
  height: min(620px, calc(100vh - 112px));
  grid-template-rows: auto auto minmax(170px, 1fr) auto auto;
  overflow: hidden;
  color: #24394e;
  background: rgb(255 255 255 / 97%);
  border: 1px solid rgb(184 200 214 / 90%);
  border-radius: 19px;
  box-shadow: 0 30px 90px rgb(20 45 70 / 25%), inset 0 1px rgb(255 255 255 / 92%);
  backdrop-filter: blur(22px) saturate(1.15);
  transform-origin: right bottom;
  animation: assistant-enter .2s ease-out both;
}
.strata-assistant-window.dragging { cursor: grabbing; user-select: none; box-shadow: 0 34px 100px rgb(20 45 70 / 31%); animation: none; }
.strata-assistant-header {
  position: relative;
  display: grid;
  min-height: 76px;
  grid-template-columns: 44px minmax(0, 1fr) auto;
  gap: 11px;
  align-items: center;
  padding: 12px 13px 15px 15px;
  background: linear-gradient(115deg, rgb(237 249 247 / 97%), rgb(241 247 253 / 97%));
  border-bottom: 1px solid #dfe8ed;
  cursor: grab;
  touch-action: none;
}
.dragging .strata-assistant-header { cursor: grabbing; }
.strata-assistant-avatar { width: 44px; height: 44px; background: #fff; border: 1px solid #d6e6e8; border-radius: 14px; box-shadow: 0 6px 18px rgb(26 92 102 / 9%); }
.strata-assistant-avatar svg { width: 33px; height: 33px; }
.strata-assistant-identity { display: grid; min-width: 0; }
.strata-assistant-identity > div { display: flex; gap: 7px; align-items: baseline; }
.strata-assistant-identity strong { color: #17354c; font-size: 17px; line-height: 1.2; }
.strata-assistant-identity > div span { color: #607486; font-size: 11px; font-weight: 650; }
.strata-assistant-identity small { display: flex; gap: 6px; align-items: center; overflow: hidden; margin-top: 3px; color: #768796; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
.strata-assistant-identity small i { width: 7px; height: 7px; flex: 0 0 7px; background: #c99e48; border-radius: 50%; }
.strata-assistant-identity small i.online { background: #22ae7b; }
.strata-assistant-window-actions { display: flex; gap: 3px; }
.strata-assistant-window-actions button { display: grid; width: 30px; height: 30px; place-items: center; padding: 0; color: #6b7e8e; font-size: 20px; line-height: 1; background: transparent; border: 0; border-radius: 8px; cursor: pointer; }
.strata-assistant-window-actions button:hover { color: #21455e; background: rgb(214 227 234 / 66%); }
.strata-assistant-window-actions svg { width: 17px; height: 17px; fill: none; stroke: currentColor; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }
.strata-assistant-drag-hint { position: absolute; right: 15px; bottom: 5px; display: flex; gap: 2px; align-items: center; color: #9aa8b2; font-size: 9px; }
.strata-assistant-drag-hint i { width: 2px; height: 2px; background: #91a1ad; border-radius: 50%; }

.strata-assistant-context { display: grid; gap: 7px; padding: 10px 15px 11px; background: #f8fafb; border-bottom: 1px solid #e4eaee; }
.strata-assistant-context > div { display: flex; gap: 12px; align-items: center; justify-content: space-between; }
.strata-assistant-context > div span { color: #81909d; font-size: 10px; font-weight: 650; }
.strata-assistant-context > div strong { overflow: hidden; color: #334e63; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.strata-assistant-context nav { display: flex; gap: 6px; }
.strata-assistant-context nav span { padding: 3px 7px; color: #637587; font-size: 9px; background: #fff; border: 1px solid #dfe7ec; border-radius: 999px; }

.strata-assistant-messages { display: flex; gap: 12px; flex-direction: column; padding: 16px; overflow-y: auto; scrollbar-color: #c8d3db transparent; scrollbar-width: thin; }
.strata-assistant-messages article { display: flex; gap: 8px; max-width: 94%; align-items: flex-start; }
.strata-assistant-messages article.user { align-self: flex-end; flex-direction: row-reverse; }
.message-avatar { display: grid; width: 27px; height: 27px; flex: 0 0 27px; place-items: center; color: #177579; background: #ddf2ed; border-radius: 9px; }
.message-avatar svg { width: 20px; height: 20px; fill: none; stroke: currentColor; stroke-width: 1.35; stroke-linecap: round; }
.message-avatar b { color: #fff; font-size: 10px; }
.user .message-avatar { background: #547696; }
.message-bubble { padding: 10px 12px; background: #f0f5f7; border: 1px solid #e2eaed; border-radius: 4px 13px 13px; }
.user .message-bubble { color: #fff; background: #285b7f; border-color: #285b7f; border-radius: 13px 4px 13px 13px; }
.message-bubble p { margin: 0; color: inherit; font-size: 13px; line-height: 1.62; }
.message-bubble > small { display: block; margin-top: 6px; color: #84939f; font-size: 10px; }
.user .message-bubble > small { color: rgb(255 255 255 / 68%); }
.strata-assistant-message-actions { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.strata-assistant-message-actions button { padding: 5px 8px; color: #176e77; font-size: 10px; font-weight: 650; background: #fff; border: 1px solid #c8dfe1; border-radius: 7px; cursor: pointer; }
.strata-assistant-message-actions button:hover { background: #eaf6f5; border-color: #89bfc0; }
.pending .thinking-dot { display: inline-block; width: 4px; height: 4px; margin-right: 2px; background: #478b8d; border-radius: 50%; animation: thinking 1s infinite ease-in-out; }
.pending .thinking-dot:nth-child(2) { animation-delay: .14s; }
.pending .thinking-dot:nth-child(3) { animation-delay: .28s; }

.strata-assistant-prompts { display: grid; gap: 8px; padding: 10px 15px 11px; background: #fbfcfd; border-top: 1px solid #e8edf0; }
.strata-assistant-prompts header { display: flex; align-items: center; justify-content: space-between; }
.strata-assistant-prompts header strong { color: #536779; font-size: 11px; }
.strata-assistant-prompts header small { color: #9aa5ae; font-size: 9px; }
.strata-assistant-prompts > div { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
.strata-assistant-prompts button { min-height: 34px; padding: 0 8px; color: #49677a; font-size: 10px; font-weight: 650; background: #f3f7f9; border: 1px solid #dce6ea; border-radius: 8px; cursor: pointer; }
.strata-assistant-prompts button:hover:not(:disabled) { color: #176c75; background: #eaf5f4; border-color: #a9ced0; }
.strata-assistant-prompts button:disabled { cursor: not-allowed; opacity: .5; }
.strata-assistant-prompts button span { margin-left: 2px; color: #8aa0ae; }

.strata-assistant-composer { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; padding: 11px 12px 13px; background: #fff; border-top: 1px solid #e0e7eb; }
.strata-assistant-composer label { position: relative; display: block; }
.strata-assistant-composer textarea { width: 100%; height: 66px; margin: 0; padding: 10px 11px 22px; resize: none; color: #263b4f; font: inherit; font-size: 13px; line-height: 1.45; background: #f7f9fa; border: 1px solid #d5dfe5; border-radius: 11px; outline: none; }
.strata-assistant-composer textarea:focus { background: #fff; border-color: #74aeb4; box-shadow: 0 0 0 3px rgb(56 139 145 / 11%); }
.strata-assistant-composer textarea::placeholder { color: #98a4ad; }
.strata-assistant-composer label small { position: absolute; right: 9px; bottom: 7px; color: #a0aab2; font-size: 9px; pointer-events: none; }
.strata-assistant-composer > button { align-self: stretch; display: grid; min-width: 66px; place-content: center; gap: 3px; padding: 0 10px; color: #fff; font-size: 11px; font-weight: 680; background: linear-gradient(145deg, #16727a, #1b607c); border: 0; border-radius: 10px; cursor: pointer; }
.strata-assistant-composer > button:hover:not(:disabled) { filter: brightness(1.06); }
.strata-assistant-composer > button:disabled { cursor: not-allowed; opacity: .42; }
.strata-assistant-composer > button svg { width: 18px; height: 18px; margin: 0 auto; fill: none; stroke: currentColor; stroke-width: 1.45; stroke-linejoin: round; }

button:focus-visible,
textarea:focus-visible { outline: 2px solid #297f98; outline-offset: 2px; }

@supports not (backdrop-filter: blur(1px)) {
  .strata-assistant-launcher,
  .strata-assistant-window { background: #fff; }
}

@keyframes launcher-scan {
  0%, 60% { opacity: 0; transform: translateX(-58px) skewX(-13deg); }
  66% { opacity: 1; }
  88% { opacity: 0; transform: translateX(190px) skewX(-13deg); }
  100% { opacity: 0; transform: translateX(190px) skewX(-13deg); }
}
@keyframes assistant-enter {
  from { opacity: 0; transform: translateY(10px) scale(.97); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes thinking {
  0%, 70%, 100% { opacity: .35; transform: translateY(0); }
  35% { opacity: 1; transform: translateY(-2px); }
}

@media (max-width: 640px) {
  .strata-assistant-launcher { right: 12px; bottom: 12px; width: 58px; min-height: 58px; grid-template-columns: 40px; gap: 0; padding: 8px; border-radius: 14px 4px 14px 14px; }
  .strata-assistant-launcher-copy,
  .strata-assistant-presence { display: none; }
  .strata-assistant-window { --assistant-edge: 12px; --assistant-bottom: 78px; width: calc(100vw - 24px); height: min(600px, calc(100vh - 96px)); }
}

@media (prefers-reduced-motion: reduce) {
  .strata-assistant-launcher,
  .strata-assistant-launcher::before,
  .strata-assistant-window,
  .thinking-dot { animation: none; transition: none; }
}
</style>
