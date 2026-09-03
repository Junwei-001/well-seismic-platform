<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

interface FusionInferenceContext {
  snapshotId: string;
  registrationTaskId: string;
  preparedViewId: string;
  readyWellCount: number;
}

const props = withDefaults(defineProps<{
  open: boolean;
  snapshotId: string;
  registrationTaskId: string;
  preparedViewId: string;
  readyWellCount?: number;
}>(), {
  readyWellCount: 0,
});

const emit = defineEmits<{
  (event: "original", context: FusionInferenceContext): void;
  (event: "layerpulse", context: FusionInferenceContext): void;
  (event: "close"): void;
}>();

const dialog = ref<HTMLDialogElement | null>(null);
const originalChoice = ref<HTMLButtonElement | null>(null);
const normalizedReadyWellCount = computed(() => (
  Number.isFinite(props.readyWellCount)
    ? Math.max(0, Math.trunc(props.readyWellCount))
    : 0
));
const snapshotLabel = computed(() => props.snapshotId.slice(0, 12) || "等待快照");
const preparedViewLabel = computed(() => props.preparedViewId.slice(0, 12) || "等待融合视图");
const inferenceContext = computed<FusionInferenceContext>(() => ({
  snapshotId: props.snapshotId,
  registrationTaskId: props.registrationTaskId,
  preparedViewId: props.preparedViewId,
  readyWellCount: normalizedReadyWellCount.value,
}));

async function synchronizeDialog() {
  await nextTick();
  const element = dialog.value;
  if (!element) return;
  if (props.open && !element.open) {
    element.showModal();
    window.requestAnimationFrame(() => originalChoice.value?.focus());
  } else if (!props.open && element.open) {
    element.close();
  }
}

function chooseOriginal() {
  dialog.value?.close();
  emit("original", inferenceContext.value);
}

function chooseLayerPulse() {
  dialog.value?.close();
  emit("layerpulse", inferenceContext.value);
}

function requestClose() {
  dialog.value?.close();
  emit("close");
}

watch(() => props.open, () => void synchronizeDialog());
onMounted(() => void synchronizeDialog());
onBeforeUnmount(() => dialog.value?.close());
</script>

<template>
  <dialog
    ref="dialog"
    class="post-fusion-inference-dialog"
    aria-labelledby="post-fusion-inference-title"
    aria-describedby="post-fusion-inference-description"
    @cancel.prevent="requestClose"
  >
    <section class="inference-shell">
      <header class="inference-header">
        <span class="inference-mark" aria-hidden="true">
          <svg viewBox="0 0 36 36">
            <path d="M5 11c5-4 9-4 13 0s8 4 13 0M5 18c5-4 9-4 13 0s8 4 13 0M5 25c5-4 9-4 13 0s8 4 13 0" />
            <path d="M18 5v26" />
            <circle cx="18" cy="18" r="3" />
          </svg>
        </span>
        <div>
          <span>融合标定完成 · PreparedView {{ preparedViewLabel }}</span>
          <h2 id="post-fusion-inference-title">选择推理方式</h2>
          <p id="post-fusion-inference-description">当前融合视图已经封存。两种方式共享同一 SourceSnapshot，不会复制或改写原始数据。</p>
        </div>
        <button type="button" class="inference-close" aria-label="暂不选择推理方式" @click="requestClose">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5.5 5.5 9 9m0-9-9 9" /></svg>
        </button>
      </header>

      <div class="fusion-receipt" aria-label="本次融合完成信息">
        <span><i aria-hidden="true"></i>融合视图已就绪</span>
        <span><b>{{ normalizedReadyWellCount.toLocaleString() }}</b> 口融合井可用于下游解释</span>
      </div>

      <div class="inference-grid">
        <button ref="originalChoice" type="button" class="inference-card original" @click="chooseOriginal">
          <span class="choice-index">01 · SINGLE TASK</span>
          <span class="choice-copy">
            <strong>单任务推理模型</strong>
            <small>进入平台原有预测工作台，按任务选择断层、层位、地震相、储层物性等模型，并保留原有参数与成果流程。</small>
          </span>
          <span class="choice-tags"><i>原模型目录</i><i>任务独立运行</i><i>模型可切换</i></span>
          <b>选择任务与模型 <span aria-hidden="true">→</span></b>
        </button>

        <button type="button" class="inference-card layerpulse" @click="chooseLayerPulse">
          <span class="choice-index">02 · FOUNDATION MODEL</span>
          <span class="choice-copy">
            <strong>LayerPulse 多模态统一智能解释</strong>
            <small>以唯一共享 Backbone 一次生成构造、地层、沉积、属性和井震推理结果。</small>
          </span>
          <span class="choice-tags"><i>唯一共享 Backbone</i><i>单 checkpoint</i><i>一次 forward</i></span>
          <span class="unified-output-line"><i aria-hidden="true"></i>统一特征 · 多任务同时返回</span>
          <b>进入统一智能解释 <span aria-hidden="true">→</span></b>
        </button>
      </div>

      <footer class="lineage-footer">
        <div><span>SourceSnapshot</span><code :title="snapshotId">{{ snapshotLabel }}</code></div>
        <i aria-hidden="true">→</i>
        <div><span>PreparedView</span><code :title="preparedViewId">{{ preparedViewLabel }}</code></div>
      </footer>
    </section>
  </dialog>
</template>

<style scoped>
.post-fusion-inference-dialog {
  width: min(950px, calc(100vw - 44px));
  max-width: none;
  max-height: calc(100dvh - 44px);
  margin: auto;
  padding: 0;
  overflow: visible;
  color: #19334b;
  background: transparent;
  border: 0;
}

.post-fusion-inference-dialog::backdrop {
  background:
    radial-gradient(circle at 72% 17%, rgb(29 151 120 / 19%), transparent 33%),
    radial-gradient(circle at 22% 82%, rgb(31 112 186 / 13%), transparent 38%),
    rgb(9 27 45 / 56%);
  backdrop-filter: blur(13px) saturate(112%);
}

.inference-shell {
  overflow: hidden auto;
  max-height: calc(100dvh - 44px);
  background: rgb(252 254 255 / 98%);
  border: 1px solid rgb(255 255 255 / 84%);
  border-radius: 24px;
  box-shadow: 0 34px 92px rgb(5 31 61 / 31%), 0 8px 25px rgb(5 31 61 / 14%);
}

.inference-header {
  position: relative;
  display: grid;
  grid-template-columns: 58px minmax(0, 1fr) 36px;
  gap: 17px;
  align-items: center;
  padding: 28px 31px 22px;
  background: linear-gradient(122deg, #f7fcff 0%, #eef8f8 58%, #edf9f4 100%);
  border-bottom: 1px solid #dfeae9;
}

.inference-mark {
  display: grid;
  width: 58px;
  height: 58px;
  place-items: center;
  color: #167a70;
  background: #fff;
  border: 1px solid #cae1df;
  border-radius: 18px;
  box-shadow: 0 9px 24px rgb(30 113 104 / 12%);
}

.inference-mark svg {
  width: 37px;
  fill: none;
  stroke: currentColor;
  stroke-linecap: round;
  stroke-width: 1.55;
}

.inference-header > div { min-width: 0; }
.inference-header > div > span { color: #16806b; font-size: 12px; font-weight: 800; letter-spacing: .08em; }
.inference-header h2 { margin: 5px 0 6px; color: #17324c; font-size: 25px; letter-spacing: -.025em; }
.inference-header p { max-width: 690px; margin: 0; color: #6c8191; font-size: 13px; line-height: 1.65; }

.inference-close {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  align-self: start;
  padding: 0;
  color: #6e8391;
  background: rgb(255 255 255 / 72%);
  border: 1px solid #d6e3e6;
  border-radius: 10px;
  cursor: pointer;
}

.inference-close:hover,
.inference-close:focus-visible { color: #234e60; background: #fff; outline: 2px solid rgb(33 128 151 / 24%); outline-offset: 2px; }
.inference-close svg { width: 17px; fill: none; stroke: currentColor; stroke-linecap: round; stroke-width: 1.6; }

.fusion-receipt {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 20px;
  align-items: center;
  justify-content: space-between;
  padding: 12px 32px;
  color: #53726f;
  font-size: 12px;
  background: #f6fbf9;
  border-bottom: 1px solid #e2ece9;
}

.fusion-receipt span { display: inline-flex; gap: 7px; align-items: center; }
.fusion-receipt span:first-child { color: #16745e; font-weight: 780; }
.fusion-receipt i { width: 8px; height: 8px; background: #24a37d; border-radius: 50%; box-shadow: 0 0 0 4px rgb(36 163 125 / 12%); }
.fusion-receipt b { color: #205e54; font-size: 13px; }

.inference-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 15px;
  padding: 22px 32px 27px;
}

.inference-card {
  position: relative;
  display: grid;
  gap: 16px;
  min-height: 294px;
  padding: 22px;
  overflow: hidden;
  text-align: left;
  background: #fff;
  border: 1px solid #dbe5ed;
  border-radius: 19px;
  cursor: pointer;
  box-shadow: 0 10px 28px rgb(29 62 93 / 7%);
  transition: border-color .17s ease, box-shadow .17s ease, transform .17s ease;
}

.inference-card::after {
  position: absolute;
  top: -65px;
  right: -65px;
  width: 150px;
  height: 150px;
  pointer-events: none;
  background: radial-gradient(circle, rgb(43 121 182 / 9%), transparent 67%);
  border-radius: 50%;
  content: "";
}

.inference-card:hover,
.inference-card:focus-visible { border-color: #67a5d3; outline: none; box-shadow: 0 17px 36px rgb(31 91 142 / 15%); transform: translateY(-2px); }
.inference-card.layerpulse { background: linear-gradient(145deg, #f4fbff, #eefaf5); border-color: #bee0d8; }
.inference-card.layerpulse::after { background: radial-gradient(circle, rgb(29 153 119 / 14%), transparent 67%); }
.inference-card.layerpulse:hover,
.inference-card.layerpulse:focus-visible { border-color: #51a891; box-shadow: 0 17px 37px rgb(24 126 99 / 16%); }

.choice-index { color: #8299aa; font: 800 12px/1 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .1em; }
.layerpulse .choice-index { color: #438778; }
.choice-copy { display: grid; gap: 8px; }
.choice-copy strong { color: #17344e; font-size: 20px; line-height: 1.3; }
.choice-copy small { color: #637a8e; font-size: 13px; line-height: 1.65; }
.choice-tags { display: flex; flex-wrap: wrap; gap: 7px; align-content: start; }
.choice-tags i { padding: 5px 8px; color: #397092; font-size: 12px; font-style: normal; font-weight: 700; background: #edf5fa; border-radius: 999px; }
.layerpulse .choice-tags i { color: #27725f; background: #e2f3ec; }
.inference-card > b { display: flex; align-items: center; justify-content: space-between; align-self: end; color: #1c679e; font-size: 13px; }
.inference-card.layerpulse > b { color: #157860; }
.inference-card > b span { font-size: 20px; font-weight: 500; }

.unified-output-line { display: inline-flex; width: max-content; max-width: 100%; gap: 8px; align-items: center; color: #39766a; font-size: 12px; font-weight: 720; }
.unified-output-line i { width: 17px; height: 3px; background: linear-gradient(90deg, #2e8ca5, #29a67d); border-radius: 999px; }

.lineage-footer {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 14px 32px 17px;
  color: #7a8d9d;
  background: #f7f9fb;
  border-top: 1px solid #e5edf1;
}

.lineage-footer div { display: flex; min-width: 0; gap: 8px; align-items: center; }
.lineage-footer div:first-child { flex: 1 1 auto; }
.lineage-footer div:last-child { flex: 1 1 auto; }
.lineage-footer span { flex: 0 0 auto; font-size: 11px; font-weight: 760; letter-spacing: .06em; }
.lineage-footer code { overflow: hidden; color: #4c657a; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.lineage-footer > i { flex: 0 0 auto; color: #9eb0bd; font-style: normal; }

@media (max-width: 720px) {
  .post-fusion-inference-dialog { width: calc(100vw - 20px); max-height: calc(100dvh - 20px); }
  .inference-shell { max-height: calc(100dvh - 20px); border-radius: 19px; }
  .inference-header { grid-template-columns: 46px minmax(0, 1fr) 31px; gap: 11px; padding: 20px 18px 16px; }
  .inference-mark { width: 46px; height: 46px; border-radius: 14px; }
  .inference-mark svg { width: 30px; }
  .inference-header h2 { font-size: 20px; }
  .inference-close { width: 31px; height: 31px; }
  .fusion-receipt { align-items: flex-start; flex-direction: column; padding: 11px 19px; }
  .inference-grid { grid-template-columns: 1fr; padding: 17px 19px 21px; }
  .inference-card { min-height: 242px; padding: 18px; }
  .lineage-footer { align-items: flex-start; flex-direction: column; padding: 13px 19px 16px; }
  .lineage-footer > i { display: none; }
  .lineage-footer div { width: 100%; }
}

@media (prefers-reduced-motion: reduce) {
  .inference-card { transition: none; }
}
</style>
