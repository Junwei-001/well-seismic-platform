<script setup lang="ts">
import { computed } from "vue";
import type { Capabilities } from "../../api";
import {
  LAYER_PULSE_MODEL_ID,
  layerPulseModelContract,
  layerPulseOutputCatalog,
  type LayerPulseOutputDefinition,
} from "../../domain/layerPulse";
import {
  publicModelIdentifier,
  publicModelText,
  scientificStatusLabel,
} from "../../domain/modelPresentation";

const props = defineProps<{ capabilities: Capabilities | null }>();

const layerPulseModel = computed(() =>
  (props.capabilities?.models || []).find((model) => model.id === LAYER_PULSE_MODEL_ID) || null,
);

function metadataNumber(key: string, fallback: number): number {
  const value = Number(layerPulseModel.value?.metadata?.[key]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

const parameterCount = computed(() =>
  metadataNumber("parameter_count", layerPulseModelContract.parameterCount),
);
const fFinalChannels = computed(() =>
  metadataNumber("f_final_channels", layerPulseModelContract.fFinalChannels),
);
const headCount = computed(() =>
  metadataNumber("head_count", layerPulseModelContract.headCount),
);
const classificationHeadCount = layerPulseOutputCatalog.filter(
  (output) => output.kind === "classification",
).length;
const continuousHeadCount = layerPulseOutputCatalog.length - classificationHeadCount;
const outputChannelCount = layerPulseOutputCatalog.reduce(
  (total, output) => total + output.channels,
  0,
);

const patchShape = computed(() => {
  const value = layerPulseModel.value?.metadata?.default_patch_size_tix;
  if (Array.isArray(value) && value.length === 3 && value.every((item) => Number(item) > 0)) {
    return value.map((item) => Number(item));
  }
  return [128, 128, 128];
});

const inputAssets = computed(() => {
  const inputs = layerPulseModel.value?.inputs || [];
  return inputs.length
    ? inputs.map((input) => publicModelText(input))
    : ["三维后叠加 SEG-Y", "登记井曲线与完整 MD 轨迹"];
});

const runtimeStatus = computed(() => String(layerPulseModel.value?.runtime_status || "unknown"));
const runtimeStatusLabel = computed(() => {
  const labels: Record<string, string> = {
    runnable: "推理服务已连接",
    unavailable: "外部资产待连接",
    blocked: "当前不可运行",
    adapter_required: "等待输入适配",
    precomputed_only: "仅封存成果",
    unknown: "等待能力目录",
  };
  return labels[runtimeStatus.value] || publicModelText(runtimeStatus.value);
});
const runtimeIsReady = computed(() => runtimeStatus.value === "runnable");

function formatParameters(value: number): string {
  return value.toLocaleString("zh-CN");
}

function parameterMillions(value: number): string {
  return `${(value / 1_000_000).toFixed(2)}M`;
}

function outputAccent(output: LayerPulseOutputDefinition): string {
  if (output.kind === "classification") {
    return output.classes.find((item) => !item.background)?.color || "#1887e8";
  }
  return output.continuousLegend.at(-1)?.color || "#7257d6";
}

function outputNodeStyle(output: LayerPulseOutputDefinition): Record<string, string> {
  return { "--output-accent": outputAccent(output) };
}
</script>

<template>
  <section class="layerpulse-model-center" aria-labelledby="layerpulse-model-title">
    <header class="lp-hero">
      <div class="lp-hero-copy">
        <div class="lp-release-line">
          <span>LayerPulse / F3X200CF</span>
          <span class="lp-runtime" :class="{ ready: runtimeIsReady }"><i></i>{{ runtimeStatusLabel }}</span>
        </div>
        <h1 id="layerpulse-model-title">LayerPulse 多模态融合基础模型</h1>
      </div>

      <div class="lp-parameter-visual" aria-label="LayerPulse 参数总量">
        <div class="lp-parameter-rings" aria-hidden="true"><i></i><i></i><i></i></div>
        <div class="lp-parameter-copy">
          <span>Parameters</span>
          <strong>{{ parameterMillions(parameterCount) }}</strong>
          <small>{{ formatParameters(parameterCount) }} parameters</small>
        </div>
      </div>
    </header>

    <section class="lp-metrics" aria-label="LayerPulse 核心指标">
      <article><strong>{{ fFinalChannels }}</strong><span>共享特征通道</span><p>F_final</p></article>
      <article><strong>{{ headCount }}</strong><span>并行任务头</span><p>{{ classificationHeadCount }} 分类 / {{ continuousHeadCount }} 连续</p></article>
      <article><strong>{{ outputChannelCount }}</strong><span>输出通道</span><p>完整 logits 与属性场</p></article>
      <article><strong>{{ patchShape[0] }}³</strong><span>默认预览子体</span><p>TWT / Inline / Crossline</p></article>
    </section>

    <section class="lp-model-architecture" aria-labelledby="lp-architecture-title">
      <header>
        <div><p>Model architecture</p><h2 id="lp-architecture-title">共享井震融合底座与多任务输出拓扑</h2></div>
        <span>ONE CHECKPOINT · ONE FORWARD · ELEVEN HEADS</span>
      </header>

      <div class="lp-model-flow">
        <div class="lp-core-flow">
          <div class="lp-input-stage">
            <p>Multimodal input</p>
            <article v-for="(input, index) in inputAssets" :key="input">
              <b>0{{ index + 1 }}</b>
              <div><strong>{{ index === 0 ? "三维地震体" : "测井与井轨迹" }}</strong><small>{{ input }}</small></div>
            </article>
          </div>

          <i class="lp-stage-arrow" aria-hidden="true"></i>

          <article class="lp-backbone-stage">
            <small>Shared backbone</small>
            <strong>LayerPulse</strong>
            <span>统一井震融合底座</span>
            <div aria-hidden="true"><i v-for="index in 8" :key="index"></i></div>
          </article>

          <i class="lp-stage-arrow" aria-hidden="true"></i>

          <article class="lp-feature-stage">
            <small>Shared feature</small>
            <strong>F<sub>final</sub></strong>
            <b>{{ fFinalChannels }} CH</b>
          </article>
        </div>

        <div class="lp-head-fanout" aria-hidden="true">
          <span></span><i v-for="index in 11" :key="index"></i>
        </div>

        <div class="lp-head-network" aria-label="LayerPulse 十一个输出头">
          <article
            v-for="output in layerPulseOutputCatalog"
            :key="output.key"
            class="lp-head-node"
            :style="outputNodeStyle(output)"
          >
            <i></i>
            <strong>{{ output.shortName }}</strong>
            <b>{{ output.channels }} CH</b>
          </article>
        </div>
      </div>
    </section>

    <footer class="lp-model-summary">
      <div>
        <strong>F3X200CF</strong>
        <span>{{ scientificStatusLabel(layerPulseModel?.scientific_status) }}</span>
        <span>{{ patchShape[0] }}³ 确定性预览子体</span>
        <span>Forward 无需时深表</span>
        <code>{{ publicModelIdentifier(layerPulseModel?.id || LAYER_PULSE_MODEL_ID) }}</code>
      </div>
      <details v-if="layerPulseModel?.warnings?.length">
        <summary>模型边界</summary>
        <p>{{ publicModelText(layerPulseModel.warnings.join("；")) }}</p>
      </details>
    </footer>
  </section>
</template>

<style scoped>
.layerpulse-model-center {
  --lp-text: #10283f;
  --lp-muted: #6c8293;
  --lp-rule: #dce6ed;
  display: grid;
  grid-template-rows: 166px 62px minmax(0, 1fr) 38px;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  border: 1px solid #dbe6ed;
  border-radius: 16px;
  color: var(--lp-text);
  background: #fff;
  box-shadow: 0 14px 42px rgb(31 70 100 / 7%);
}
.layerpulse-model-center sub { font-size: inherit; line-height: 0; }
:global(.main-content:has(.layerpulse-model-center)) {
  height: calc(100dvh - var(--topbar-height));
  min-height: 0;
  padding-top: 14px;
  padding-bottom: 8px;
  overflow: hidden;
}
:global(.main-content:has(.layerpulse-model-center) .page-header) {
  display: none;
}

.lp-hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  overflow: hidden;
  border-bottom: 1px solid var(--lp-rule);
  background: linear-gradient(110deg, #fff 0%, #fbfdff 62%, #eef8ff 100%);
}
.lp-hero-copy { align-self: center; min-width: 0; padding: 16px 28px; }
.lp-release-line { display: flex; gap: 20px; align-items: center; margin-bottom: 10px; color: #5a7386; font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.lp-runtime { display: inline-flex; gap: 6px; align-items: center; color: #718594; font-weight: 600; letter-spacing: 0; text-transform: none; }
.lp-runtime i { width: 6px; height: 6px; border-radius: 50%; background: #98aab7; }
.lp-runtime.ready { color: #26775d; }
.lp-runtime.ready i { background: #28a779; box-shadow: 0 0 0 3px rgb(40 167 121 / 10%); }
.lp-hero h1 { margin: 0; color: #112e46; font-size: 27px; font-weight: 650; letter-spacing: -0.035em; }
.lp-parameter-visual { position: relative; display: grid; place-items: center; overflow: hidden; }
.lp-parameter-rings,
.lp-parameter-rings i { position: absolute; border: 1px solid rgb(74 161 214 / 19%); border-radius: 50%; }
.lp-parameter-rings { right: -4px; width: 340px; height: 340px; background: radial-gradient(circle, #dcefff 0%, #edf7fc 45%, transparent 69%); animation: lp-orbit-spin 28s linear infinite; }
.lp-parameter-rings i:nth-child(1) { inset: 38px; border-style: dashed; }
.lp-parameter-rings i:nth-child(2) { inset: 76px; border-color: rgb(120 99 213 / 24%); }
.lp-parameter-rings i:nth-child(3) { top: 35px; right: 30px; width: 8px; height: 8px; border: 0; background: #7262d5; box-shadow: 0 0 10px rgb(114 98 213 / 32%); }
.lp-parameter-copy { position: relative; z-index: 1; padding-right: 8px; text-align: center; }
.lp-parameter-copy span { display: block; color: #597487; font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
.lp-parameter-copy strong { display: block; margin: 5px 0 4px; color: #0b4f7d; font-size: 72px; font-weight: 650; line-height: 0.92; letter-spacing: -0.07em; }
.lp-parameter-copy small { color: #6b8799; font: 600 12px/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; }

.lp-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-bottom: 1px solid var(--lp-rule); }
.lp-metrics article { display: grid; grid-template-columns: auto 1fr; column-gap: 8px; align-content: center; min-width: 0; padding: 8px 16px; }
.lp-metrics article + article { border-left: 1px solid var(--lp-rule); }
.lp-metrics strong { grid-row: 1 / 3; color: #123b58; font-size: 30px; font-weight: 620; line-height: 1; letter-spacing: -0.04em; }
.lp-metrics span { overflow: hidden; color: #36576d; font-size: 13px; font-weight: 700; text-overflow: ellipsis; white-space: nowrap; }
.lp-metrics p { margin: 2px 0 0; overflow: hidden; color: #7c919f; font-size: 13px; text-overflow: ellipsis; white-space: nowrap; }

.lp-model-architecture { display: grid; grid-template-rows: auto minmax(0, 1fr); min-width: 0; min-height: 0; overflow: hidden; padding: 8px 14px 10px; background: #f8fbfd; }
.lp-model-architecture > header { display: flex; gap: 20px; align-items: end; justify-content: space-between; padding: 0 2px 8px; }
.lp-model-architecture > header p { margin: 0 0 2px; color: #1680c6; font-size: 12px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
.lp-model-architecture > header h2 { margin: 0; color: #17384f; font-size: 17px; font-weight: 650; letter-spacing: -0.02em; }
.lp-model-architecture > header > span { color: #758b9a; font-size: 12px; letter-spacing: 0.04em; }
.lp-model-flow {
  position: relative;
  display: grid;
  grid-template-rows: clamp(112px, 30%, 152px) 36px clamp(108px, 29%, 144px);
  align-content: center;
  min-height: 0;
  padding: 14px 16px;
  border: 1px solid #dce7ed;
  border-radius: 11px;
  background:
    radial-gradient(circle at 55% 42%, rgb(52 143 194 / 6%), transparent 42%),
    linear-gradient(180deg, #fff 0%, #fbfdfe 100%);
}
.lp-core-flow {
  display: grid;
  grid-template-columns: minmax(170px, 1.05fr) 42px minmax(210px, 1.2fr) 42px minmax(135px, 0.75fr);
  gap: 10px;
  align-items: stretch;
  width: min(1000px, 78%);
  height: 100%;
  margin: 0 auto;
}
.lp-input-stage {
  display: flex;
  min-width: 0;
  height: 100%;
  padding: 10px 12px;
  border: 1px solid #d9e6ed;
  border-radius: 9px;
  flex-direction: column;
  justify-content: center;
  background: rgb(249 252 254 / 92%);
}
.lp-input-stage > p { margin: 0 0 4px; color: #6d899c; font-size: 13px; font-weight: 800; text-transform: uppercase; }
.lp-input-stage article { display: flex; gap: 7px; align-items: center; min-width: 0; padding: 2px 0; border-top: 1px solid #e2eaef; }
.lp-input-stage article:last-child { border-bottom: 1px solid #e2eaef; }
.lp-input-stage article > b { color: #8195a3; font: 700 13px/1 ui-monospace, SFMono-Regular, Consolas, monospace; }
.lp-input-stage article > div { min-width: 0; }
.lp-input-stage strong,
.lp-input-stage small { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lp-input-stage strong { color: #2d5066; font-size: 14px; line-height: 1.18; }
.lp-input-stage small { margin-top: 1px; color: #748b9a; font-size: 13px; line-height: 1.18; }
.lp-stage-arrow { position: relative; display: block; align-self: center; width: 100%; height: 1px; background: #aac0cd; }
.lp-stage-arrow::after { position: absolute; top: -3px; right: 0; width: 6px; height: 6px; border-top: 1px solid #8ba8b9; border-right: 1px solid #8ba8b9; content: ""; transform: rotate(45deg); }
.lp-backbone-stage { position: relative; min-height: 0; overflow: hidden; padding: 12px 14px; border: 1px solid #cde1ec; border-radius: 9px; background: linear-gradient(140deg, #fff, #eaf6fc); }
.lp-backbone-stage > small,
.lp-feature-stage > small { display: block; color: #6d899c; font-size: 13px; font-weight: 800; text-transform: uppercase; }
.lp-backbone-stage > strong { display: block; margin-top: 4px; color: #0e6095; font-size: 26px; }
.lp-backbone-stage > span { color: #647f91; font-size: 13px; }
.lp-backbone-stage > div { position: absolute; right: 10px; bottom: 8px; left: 10px; display: grid; grid-template-columns: repeat(8, 1fr); gap: 3px; align-items: end; height: 24px; }
.lp-backbone-stage > div i { height: 11px; background: #b9dfef; transform-origin: bottom; animation: lp-layer-breathe 3.5s ease-in-out infinite alternate; }
.lp-backbone-stage > div i:nth-child(2n) { height: 19px; animation-delay: -1.2s; }
.lp-backbone-stage > div i:nth-child(3n) { height: 24px; animation-delay: -2.3s; }
.lp-feature-stage { position: relative; display: grid; align-content: center; min-height: 0; padding: 12px 14px; border-left: 3px solid #52a3d3; background: #eef7fb; }
.lp-feature-stage::after { position: absolute; top: 100%; left: 50%; width: 1px; height: 18px; background: #91b0c1; content: ""; }
.lp-feature-stage > strong { display: block; margin-top: 7px; color: #173f5b; font-size: 28px; line-height: 1; }
.lp-feature-stage > b { display: block; margin-top: 8px; color: #1680bd; font-size: 13px; }
.lp-head-fanout { position: relative; display: grid; grid-template-columns: repeat(11, minmax(0, 1fr)); gap: 8px; height: 36px; }
.lp-head-fanout > span { display: none; }
.lp-head-fanout::after { position: absolute; top: 50%; right: calc((100% - 80px) / 22); left: calc((100% - 80px) / 22); height: 1px; background: #91b0c1; content: ""; }
.lp-head-fanout > i { position: relative; z-index: 1; display: block; height: 100%; }
.lp-head-fanout > i::after { position: absolute; top: 50%; bottom: 0; left: 50%; width: 1px; background: #a9bfcb; content: ""; }
.lp-head-network { display: grid; grid-template-columns: repeat(11, minmax(0, 1fr)); gap: 8px; align-self: center; width: 100%; height: 100%; max-height: 144px; }
.lp-head-node { position: relative; display: grid; grid-template-rows: 8px max-content max-content; place-items: center; align-content: center; row-gap: 9px; min-width: 0; min-height: 0; height: 100%; padding: 10px 5px; border: 1px solid #dce7ed; border-radius: 8px; background: rgb(251 253 254 / 96%); box-shadow: 0 7px 18px rgb(37 83 111 / 4%); text-align: center; }
.lp-head-node::before { position: absolute; bottom: 100%; left: 50%; width: 1px; height: 5px; background: #a9bfcb; content: ""; }
.lp-head-node > i { width: 7px; height: 7px; border-radius: 50%; background: var(--output-accent); }
.lp-head-node > strong { width: 100%; overflow: hidden; color: #274b62; font-size: 14px; line-height: 1.25; text-overflow: ellipsis; white-space: nowrap; }
.lp-head-node > b { color: #678090; font: 700 12px/1 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: nowrap; }

.lp-model-summary { position: relative; display: flex; gap: 18px; align-items: center; justify-content: space-between; padding: 0 16px; border-top: 1px solid var(--lp-rule); background: #fff; }
.lp-model-summary > div { display: flex; min-width: 0; gap: 16px; align-items: center; }
.lp-model-summary strong { color: #214861; font-size: 12px; }
.lp-model-summary span { color: #758a98; font-size: 12px; white-space: nowrap; }
.lp-model-summary code { overflow: hidden; color: #3180ae; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.lp-model-summary details { position: relative; flex: 0 0 auto; }
.lp-model-summary summary { color: #4d6b7e; cursor: pointer; font-size: 12px; font-weight: 700; }
.lp-model-summary details p { position: absolute; right: 0; bottom: 24px; z-index: 10; width: min(640px, 70vw); margin: 0; padding: 14px; border: 1px solid #d7e3ea; border-radius: 9px; color: #6d8290; background: #fff; box-shadow: 0 12px 34px rgb(33 70 98 / 15%); font-size: 12px; line-height: 1.6; }

@keyframes lp-orbit-spin { to { transform: rotate(360deg); } }
@keyframes lp-layer-breathe { from { opacity: 0.6; transform: scaleY(0.72); } to { opacity: 1; transform: scaleY(1); } }

@media (max-width: 1080px) {
  .lp-hero { grid-template-columns: minmax(0, 1fr) 250px; }
  .lp-core-flow {
    grid-template-columns: minmax(140px, 1fr) 30px minmax(170px, 1.15fr) 30px minmax(110px, 0.72fr);
    gap: 7px;
    width: min(820px, 90%);
  }
  .lp-model-summary code { display: none; }
}

@media (min-width: 1500px) {
  .layerpulse-model-center { height: 100%; max-height: none; }
  .lp-model-architecture { padding: 14px 18px 16px; }
  .lp-model-architecture > header { padding-bottom: 12px; }
  .lp-model-architecture > header h2 { font-size: 20px; }
  .lp-model-flow { padding: 20px 24px; }
  .lp-core-flow { width: min(1180px, 78%); }
  .lp-input-stage { padding: 14px 16px; }
  .lp-backbone-stage,
  .lp-feature-stage { padding: 16px 18px; }
  .lp-input-stage > p,
  .lp-backbone-stage > small,
  .lp-feature-stage > small { font-size: 14px; }
  .lp-input-stage strong { font-size: 15px; }
  .lp-backbone-stage > strong { font-size: 28px; }
  .lp-feature-stage > strong { font-size: 31px; }
  .lp-head-node { padding: 13px 6px; }
  .lp-head-node > strong { font-size: 16px; }
  .lp-head-node > b { font-size: 13px; }
}

@media (max-width: 900px) {
  .layerpulse-model-center { grid-template-rows: auto; height: auto; min-height: 0; overflow: visible; }
  .lp-hero { grid-template-columns: 1fr; min-height: 220px; }
  .lp-parameter-visual { min-height: 150px; }
  .lp-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .lp-metrics article:nth-child(3) { border-top: 1px solid var(--lp-rule); border-left: 0; }
  .lp-metrics article:nth-child(4) { border-top: 1px solid var(--lp-rule); }
  .lp-model-flow { grid-template-rows: auto; gap: 12px; }
  .lp-core-flow { grid-template-columns: 1fr; width: 100%; height: auto; }
  .lp-input-stage { height: auto; min-height: 96px; }
  .lp-stage-arrow { width: 1px; height: 18px; margin: auto; }
  .lp-stage-arrow::after { top: auto; right: -3px; bottom: 0; transform: rotate(135deg); }
  .lp-head-fanout { display: none; }
  .lp-feature-stage::after { display: none; }
  .lp-head-network { grid-template-columns: repeat(3, minmax(0, 1fr)); height: auto; max-height: none; }
  .lp-head-node { min-height: 88px; }
}

@media (max-width: 560px) {
  .lp-release-line { align-items: flex-start; flex-direction: column; gap: 6px; }
  .lp-metrics { grid-template-columns: 1fr; }
  .lp-metrics article + article { border-top: 1px solid var(--lp-rule); border-left: 0; }
  .lp-head-network { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .lp-model-summary span,
  .lp-model-summary code { display: none; }
}

@media (max-height: 630px) and (min-width: 901px) {
  .layerpulse-model-center { height: 100%; min-height: 0; overflow: hidden; }
}

@media (prefers-reduced-motion: reduce) {
  .lp-parameter-rings,
  .lp-backbone-stage > div i { animation: none; }
}
</style>
