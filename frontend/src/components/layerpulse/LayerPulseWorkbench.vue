<script setup lang="ts">
import { computed } from "vue";

import {
  createIdleLayerPulseTaskState,
  isLayerPulseTaskActive,
  layerPulseOutputByKey,
  layerPulseTaskStatusLabels,
  summarizeLayerPulseSupport,
  type LayerPulseSupportReceipt,
  type LayerPulseTaskState,
} from "../../domain/layerPulse";

interface LayerPulseDownloadLink {
  url: string;
  filename: string;
}

interface LayerPulseOutputDownloads {
  segy: LayerPulseDownloadLink;
  rawNpy: LayerPulseDownloadLink;
  classLegendCsv?: LayerPulseDownloadLink;
}

const props = withDefaults(defineProps<{
  snapshotId: string;
  fusionReady?: boolean;
  supportReceipt?: LayerPulseSupportReceipt | null;
  taskState?: LayerPulseTaskState;
  selectedOutputKey?: string;
  canvasMode?: "base" | "result";
  baseVisualizationUrl?: string;
  resultVisualizationUrl?: string;
  standaloneResultUrl?: string;
  outputDownloads?: Record<string, LayerPulseOutputDownloads>;
}>(), {
  supportReceipt: null,
  fusionReady: false,
  taskState: createIdleLayerPulseTaskState,
  selectedOutputKey: "fault_logits",
  canvasMode: "base",
  baseVisualizationUrl: "",
  resultVisualizationUrl: "",
  standaloneResultUrl: "",
  outputDownloads: () => ({}),
});

const emit = defineEmits<{
  (event: "run"): void;
  (event: "retry"): void;
  (event: "update:selectedOutputKey", outputKey: string): void;
  (event: "update:canvasMode", mode: "base" | "result"): void;
  (event: "open-standalone", url: string): void;
}>();

const matchingReceipt = computed(() => (
  props.supportReceipt
  && props.snapshotId
  && props.supportReceipt.snapshot_id === props.snapshotId
    ? props.supportReceipt
    : null
));
const support = computed(() => summarizeLayerPulseSupport(matchingReceipt.value));
const activeOutput = computed(() => layerPulseOutputByKey(props.selectedOutputKey));
const activeOutputDisplayName = computed(() => (
  activeOutput.value.key === "rgt"
    ? "RGT 派生等时面（展示）"
    : activeOutput.value.name
));
const activeOutputDescription = computed(() => (
  activeOutput.value.key === "rgt"
    ? "由唯一共享 Backbone 输出的 RGT 连续场在可视化端派生等时面；仅用于空间展示，不是独立层位 Head。"
    : activeOutput.value.description
));
const taskActive = computed(() => isLayerPulseTaskActive(props.taskState.status));
const taskProgress = computed(() => Math.min(100, Math.max(0, Number(props.taskState.progress) || 0)));
const canRun = computed(() => Boolean(
  props.snapshotId
  && props.fusionReady
  && matchingReceipt.value
  && support.value.status !== "blocked"
  && !taskActive.value,
));
const viewerUrl = computed(() => (
  props.canvasMode === "result" && props.resultVisualizationUrl
    ? props.resultVisualizationUrl
    : props.baseVisualizationUrl
));
const showingResult = computed(() => (
  props.canvasMode === "result" && Boolean(props.resultVisualizationUrl)
));
const availableOutputKeys = computed(() => new Set(props.taskState.availableOutputKeys || []));
const activeOutputReady = computed(() => availableOutputKeys.value.has(activeOutput.value.key));
const activeOutputDownloads = computed(() => props.outputDownloads[activeOutput.value.key]);
const activeSegyDownload = computed(() => activeOutputDownloads.value?.segy);
const activeLegendDownload = computed(() => activeOutputDownloads.value?.classLegendCsv);
const activeRawDownload = computed(() => activeOutputDownloads.value?.rawNpy);

function setCanvasMode(mode: "base" | "result") {
  emit("update:canvasMode", mode);
}
</script>

<template>
  <section
    id="layerpulse-output-panel"
    class="layerpulse-workbench"
    role="tabpanel"
    :aria-labelledby="`layerpulse-output-tab-${activeOutput.key}`"
    aria-label="LayerPulse 多模态融合基础模型工作台"
  >
    <aside class="layerpulse-control-panel">
      <header class="layerpulse-control-heading">
        <span>LAYERPULSE</span>
        <h2>多模态融合基础模型</h2>
        <p>单 checkpoint · 唯一共享 Backbone · 一次 forward 返回 11 项解释结果</p>
      </header>

      <section class="task-strip" :data-status="taskState.status" aria-live="polite">
        <div class="task-actions">
          <button v-if="taskState.status === 'failed'" type="button" class="secondary" :disabled="!canRun" @click="emit('retry')">重新运行</button>
          <button v-else type="button" class="primary" :disabled="!canRun" @click="emit('run')">
            {{ taskActive ? "统一推理中…" : taskState.status === "completed" ? "重新运行全部任务" : "运行全部 11 项任务" }}
          </button>
        </div>
        <div class="task-status-copy">
          <span>{{ layerPulseTaskStatusLabels[taskState.status] }}</span>
          <strong>{{ taskState.message }}</strong>
          <code v-if="taskState.taskId">{{ taskState.taskId }}</code>
        </div>
        <div v-if="taskActive" class="task-progress">
          <div><span :style="{ width: `${taskProgress}%` }"></span></div><b>{{ taskProgress.toFixed(0) }}%</b>
        </div>
        <p v-if="taskState.error" class="task-error">{{ taskState.error }}</p>
        <p v-else-if="!fusionReady" class="task-error">请先完成当前 SourceSnapshot 的精细标定与融合视图，再运行统一解释。</p>
        <p v-else-if="support.status === 'blocked'" class="task-error">当前数据尚未满足统一解释运行条件，请返回数据与融合步骤补充必需输入。</p>
      </section>

      <aside class="legend-panel" aria-label="当前输出说明与图例">
        <header><span>{{ activeOutput.kind === "classification" ? "完整分类 logits" : "连续预测场" }}</span><strong>{{ activeOutputDisplayName }}</strong></header>
        <p>{{ activeOutputDescription }}</p>
        <div v-if="activeOutput.key === 'rgt'" class="derived-display-note">
          展示派生：从 RGT 连续场生成等时面；模型仍为 11 个 Head，不包含独立层位 Head。
        </div>
        <div v-if="activeOutput.kind === 'classification'" class="class-contract">
          <span>{{ activeOutput.channels }} 个通道</span><span>背景索引 {{ activeOutput.backgroundIndex }}</span><span>沿类别维直接 argmax</span>
        </div>
        <div v-if="activeOutput.classes.length" class="discrete-legend">
          <article v-for="item in activeOutput.classes" :key="item.id">
            <i :style="{ background: item.color }"></i><span>{{ item.index }}</span><strong>{{ item.label }}</strong><small v-if="item.background">有效训练类别</small>
          </article>
        </div>
        <div v-else class="continuous-legend">
          <div :style="{ background: `linear-gradient(90deg, ${activeOutput.continuousLegend.map((stop) => `${stop.color} ${stop.position * 100}%`).join(', ')})` }"></div>
          <span v-for="stop in activeOutput.continuousLegend" :key="`${activeOutput.key}-${stop.position}`" :style="{ left: `${stop.position * 100}%` }">{{ stop.label }}</span>
        </div>
        <small class="legend-note">分类结果不使用 sigmoid、类别阈值或前端连通域清理；图例与 checkpoint 输出合同一致。</small>
      </aside>

      <section v-if="activeOutputReady && (activeSegyDownload || activeRawDownload)" class="active-output-downloads" aria-label="当前任务结果下载">
        <header><span>标准成果</span><strong>{{ activeOutputDisplayName }}</strong></header>
        <div>
          <a v-if="activeSegyDownload" :href="activeSegyDownload.url" :download="activeSegyDownload.filename" target="_blank" rel="noopener">下载 SEG-Y</a>
          <a v-if="activeLegendDownload" :href="activeLegendDownload.url" :download="activeLegendDownload.filename" target="_blank" rel="noopener">类别码表 CSV</a>
          <a v-if="activeRawDownload" :href="activeRawDownload.url" :download="activeRawDownload.filename" target="_blank" rel="noopener">原始 NPY</a>
        </div>
      </section>
    </aside>

    <section class="viewer-panel">
      <header>
        <div><span>LIVE VIEW</span><strong>{{ showingResult ? activeOutputDisplayName : "当前工区基础数据" }}</strong><small>{{ showingResult ? "解释结果" : "基础数据" }}</small></div>
        <nav aria-label="可视化内容切换">
          <button type="button" :class="{ active: canvasMode === 'base' }" @click="setCanvasMode('base')">基础数据</button>
          <button type="button" :class="{ active: canvasMode === 'result' }" :disabled="!resultVisualizationUrl" @click="setCanvasMode('result')">解释结果</button>
        </nav>
      </header>
      <div v-if="viewerUrl" class="platform-visualization-shell">
        <iframe
          :key="viewerUrl"
          :src="viewerUrl"
          :title="showingResult ? `${activeOutputDisplayName}可视化` : '当前工区基础数据可视化'"
          allow="fullscreen"
          allowfullscreen
        ></iframe>
      </div>
      <div v-else class="viewer-empty">
        <span aria-hidden="true">⌁</span><strong>等待可视化资产</strong><p>基础数据或标准结果图层登记后将在这里显示。</p>
      </div>
      <footer v-if="standaloneResultUrl && showingResult">
        <button type="button" @click="emit('open-standalone', standaloneResultUrl)">在独立视图中打开</button>
      </footer>
    </section>
  </section>
</template>

<style scoped>
.layerpulse-workbench { display: grid; min-width: 0; grid-template-columns: minmax(330px, 360px) minmax(540px, 1fr); gap: 16px; align-items: start; color: #1b354d; }
.layerpulse-control-panel { max-height: calc(100dvh - var(--topbar-height) - 36px); overflow: auto; padding: 16px; background: #fff; border: 1px solid #d7dfe7; border-radius: 11px; box-shadow: 0 8px 24px rgb(30 52 79 / 5%); scrollbar-width: thin; }
.layerpulse-control-heading { display: grid; gap: 3px; padding-bottom: 13px; border-bottom: 1px solid #e0e7ed; }
.layerpulse-control-heading > span { color: #30746d; font-size: 12px; font-weight: 800; letter-spacing: .14em; }
.layerpulse-control-heading h2 { margin: 0; color: #20364b; font-size: 20px; line-height: 1.3; }
.layerpulse-control-heading p { margin: 0; color: #718294; font-size: 12px; line-height: 1.55; }
.task-strip { display: grid; gap: 9px; padding: 14px 0; border-bottom: 1px solid #e0e7ed; }
.task-status-copy { display: grid; gap: 3px; min-width: 0; }
.task-status-copy span { color: #32816c; font-size: 12px; font-weight: 800; }
.task-status-copy strong { color: #29465e; font-size: 13px; }
.task-status-copy code { overflow: hidden; color: #7a8c9c; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.task-progress { display: grid; grid-template-columns: minmax(0, 1fr) 44px; gap: 9px; align-items: center; }
.task-progress > div { height: 7px; overflow: hidden; background: #dfe8ee; border-radius: 999px; }
.task-progress > div span { display: block; height: 100%; background: linear-gradient(90deg, #1684c7, #28a17c); border-radius: inherit; transition: width .2s ease; }
.task-progress b { color: #42627a; font-size: 12px; }
.task-error { margin: 0; padding: 8px 10px; color: #9b433d; font-size: 12px; line-height: 1.45; background: #fff0ee; border-radius: 7px; }
.task-actions { display: flex; order: 2; }
.task-actions button { width: 100%; min-height: 58px; padding: 9px 17px; font-size: 13px; font-weight: 800; line-height: 1.4; border-radius: 9px; cursor: pointer; }
.task-actions .primary { color: #fff; background: linear-gradient(110deg, #126fe0, #118f92); border: 0; box-shadow: 0 8px 18px rgb(18 111 224 / 18%); }
.task-actions .secondary { color: #315d7d; background: #fff; border: 1px solid #cadbe6; }
.task-actions button:disabled { cursor: not-allowed; filter: grayscale(.45); opacity: .55; }

.viewer-panel { position: sticky; top: calc(var(--topbar-height) + 14px); display: grid; height: calc(100dvh - var(--topbar-height) - 36px); min-width: 0; min-height: 640px; grid-template-rows: auto minmax(0, 1fr) auto; overflow: hidden; background: radial-gradient(circle at 48% 42%, #f8fbfd 0%, #eef3f7 58%, #e3eaf0 100%); border: 1px solid #ccd6df; border-radius: 11px; box-shadow: 0 12px 34px rgb(21 45 72 / 8%); }
.viewer-panel > header { display: flex; min-height: 64px; gap: 16px; align-items: center; justify-content: space-between; padding: 10px 14px; background: #fff; border-bottom: 1px solid #d8e0e7; }
.viewer-panel > header > div { display: grid; gap: 2px; }
.viewer-panel > header span { color: #30746d; font-size: 12px; font-weight: 800; letter-spacing: .14em; }
.viewer-panel > header strong { color: #20364b; font-size: 15px; }
.viewer-panel > header small { color: #788796; font-size: 12px; }
.viewer-panel nav { display: flex; flex-wrap: wrap; gap: 5px; justify-content: flex-end; }
.viewer-panel nav button { min-height: 31px; padding: 6px 10px; color: #52677b; font-size: 12px; font-weight: 680; background: #f5f7f9; border: 1px solid #d7e0e7; border-radius: 6px; cursor: pointer; }
.viewer-panel nav button.active { color: #fff; background: #176f68; border-color: #176f68; }
.viewer-panel nav button:disabled { cursor: not-allowed; opacity: .45; }
.platform-visualization-shell { display: grid; grid-template-rows: minmax(0, 1fr); min-width: 0; min-height: 0; overflow: hidden; background: #edf2f6; }
.viewer-panel iframe { display: block; width: 100%; height: 100%; min-height: 0; border: 0; }
.viewer-empty { display: grid; align-content: center; justify-items: center; padding: 45px; color: #597184; text-align: center; }
.viewer-empty > span { font-size: 38px; }
.viewer-empty strong { margin-top: 8px; font-size: 16px; }
.viewer-empty p { margin: 6px 0 0; font-size: 12px; }
.viewer-panel > footer { display: flex; justify-content: flex-end; padding: 5px 9px; background: #f7fafc; border-top: 1px solid #dce6ed; }
.viewer-panel > footer button { color: #27678f; font-size: 12px; background: transparent; border: 0; cursor: pointer; }

.legend-panel { padding-top: 14px; }
.legend-panel > header { display: grid; gap: 2px; }
.legend-panel > header span { color: #3482a8; font-size: 12px; font-weight: 800; letter-spacing: .06em; }
.legend-panel > header strong { color: #27455e; font-size: 15px; }
.legend-panel > p { margin: 6px 0 8px; color: #667d90; font-size: 12px; line-height: 1.4; }
.derived-display-note { margin: -1px 0 8px; padding: 6px 7px; color: #76571e; font-size: 12px; line-height: 1.35; background: #fff7df; border: 1px solid #ebd59b; border-radius: 6px; }
.class-contract { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.class-contract span { padding: 3px 5px; color: #496f89; font-size: 12px; background: #eef5f9; border-radius: 999px; }
.discrete-legend { display: grid; gap: 4px; }
.discrete-legend article { display: grid; grid-template-columns: 15px 19px minmax(0, 1fr); gap: 5px; align-items: center; min-height: 25px; padding: 3px 5px; background: #f8fafb; border: 1px solid #e8edf1; border-radius: 6px; }
.discrete-legend article i { width: 13px; height: 13px; border: 1px solid rgb(28 55 76 / 16%); border-radius: 3px; }
.discrete-legend article span { color: #8998a5; font: 700 12px/1 ui-monospace, SFMono-Regular, Consolas, monospace; }
.discrete-legend article strong { color: #38546a; font-size: 12px; }
.discrete-legend article small { grid-column: 3; color: #8797a3; font-size: 12px; }
.continuous-legend { position: relative; height: 48px; margin: 16px 5px 8px; }
.continuous-legend > div { height: 14px; border: 1px solid rgb(35 60 78 / 14%); border-radius: 999px; }
.continuous-legend > span { position: absolute; top: 23px; color: #6f8291; font-size: 12px; transform: translateX(-50%); white-space: nowrap; }
.continuous-legend > span:first-of-type { transform: none; }
.continuous-legend > span:last-of-type { transform: translateX(-100%); }
.legend-note { display: block; margin-top: 8px; padding: 7px; color: #6e7f8c; font-size: 12px; line-height: 1.35; background: #f4f7f9; border-left: 3px solid #6aa0bf; border-radius: 6px; }
.active-output-downloads { display: grid; gap: 8px; margin-top: 14px; padding-top: 14px; border-top: 1px solid #e0e7ed; }
.active-output-downloads > header { display: grid; gap: 2px; }
.active-output-downloads > header span { color: #30746d; font-size: 12px; font-weight: 800; letter-spacing: .08em; }
.active-output-downloads > header strong { color: #20364b; font-size: 14px; }
.active-output-downloads > div { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; }
.active-output-downloads a { display: grid; min-height: 36px; place-items: center; padding: 7px 9px; color: #176a99; font-size: 12px; font-weight: 750; text-align: center; text-decoration: none; background: #f5f9fd; border: 1px solid #cbdde8; border-radius: 6px; }
.active-output-downloads a:hover { color: #0c527d; border-color: #8cb8cf; }
.active-output-downloads a:focus-visible { outline: 2px solid #2b87bd; outline-offset: 1px; }

@media (max-width: 1240px) {
  .layerpulse-workbench { grid-template-columns: minmax(0, 1fr); }
  .layerpulse-control-panel { max-height: none; overflow: visible; }
  .viewer-panel { position: relative; top: auto; height: 650px; min-height: 650px; }
}

@media (max-width: 820px) {
  .layerpulse-control-panel { padding: 14px; }
  .viewer-panel { height: 560px; min-height: 560px; }
  .discrete-legend { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 520px) {
  .viewer-panel > header { align-items: flex-start; flex-direction: column; }
  .discrete-legend { grid-template-columns: 1fr; }
}

@media (prefers-reduced-motion: reduce) {
  .task-progress > div span { transition: none; }
}
</style>
