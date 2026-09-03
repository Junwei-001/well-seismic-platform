<script setup lang="ts">
import { computed, ref } from "vue";
import type {
  ArtifactRelease,
  ReleaseArtifact,
  ReleaseArtifactCollection,
  ReleaseArtifactValue,
  ReleaseCatalogResponse,
} from "../../api";
import {
  publicModelIdentifier,
  publicModelText,
  releasePresentationName,
} from "../../domain/modelPresentation";
import { isWellPropertyCompletionModelId } from "../../domain/platform";

const props = withDefaults(defineProps<{
  catalog: ReleaseCatalogResponse | null;
  loading?: boolean;
  error?: string;
  uiRunnableModelIds?: string[];
}>(), {
  loading: false,
  error: "",
  uiRunnableModelIds: () => [],
});
const emit = defineEmits<{
  run: [taskId: string, modelId: string];
}>();

const search = ref("");
const scientificFilter = ref("all");
const runtimeFilter = ref("all");

interface ParameterBreakdownView {
  core: number;
  anchor: number;
  deployment: number;
  members: number;
}

interface TaskAdapterEvidenceView {
  id: string;
  label: string;
  status: string;
  metric: string;
  attribution: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function parameterBreakdown(release: ArtifactRelease): ParameterBreakdownView | null {
  const metadata = release.metadata;
  if (!metadata) return null;
  const core = Number(metadata.core_logical_parameter_count);
  const anchor = Number(metadata.production_anchor_logical_parameter_count);
  const deployment = Number(metadata.deployment_logical_parameter_count);
  const members = Number(metadata.production_anchor_member_count);
  if (![core, anchor, deployment, members].every(Number.isFinite)) return null;
  if (core + anchor !== deployment || members !== 18) return null;
  return { core, anchor, deployment, members };
}

function formatParameters(value: number): string {
  return value.toLocaleString("zh-CN");
}

function taskAdapterEvidence(release: ArtifactRelease): TaskAdapterEvidenceView[] {
  const rawEntries = release.metadata?.task_adapter_evidence;
  if (!Array.isArray(rawEntries)) return [];
  return rawEntries.flatMap((raw, index) => {
    const entry = asRecord(raw);
    if (!entry) return [];
    const task = String(entry.task || `task-${index + 1}`);
    const effect = asRecord(entry.effect_attribution) || {};
    if (task === "facies_1d") {
      const gain = Number(effect.macro_f1_gain_over_protected_expert);
      const nonzero = Number(effect.nonzero_residual_cells);
      const protectedZero = Number(effect.protected_update0_cells);
      return [{
        id: task,
        label: "Facies1D v2",
        status: String(entry.status || "unknown"),
        metric: `整井 Macro-F1 ${Number(entry.whole_well_macro_f1).toFixed(4)}`,
        attribution: `12B residual 净增 ${gain >= 0 ? "+" : ""}${gain.toFixed(4)}；${nonzero}/18 单元采用非零残差，${protectedZero}/18 由 update=0 保护。`,
      }];
    }
    if (task === "den") {
      return [{
        id: task,
        label: "DEN v2",
        status: String(entry.status || "unknown"),
        metric: `整井 Macro-MAE ${Number(entry.macro_mae_g_cm3).toFixed(4)} g/cm³`,
        attribution: `多模态残差门=${Number(effect.direct12b_residual_gate).toFixed(1)}；指标仅归因于保护专家，不计作多模态残差增益。`,
      }];
    }
    return [{
      id: task,
      label: task,
      status: String(entry.status || "unknown"),
      metric: "任务证据已登记",
      attribution: String(effect.direct12b_residual_contribution || "归因待登记"),
    }];
  });
}

const scientificLabels: Record<string, string> = {
  validated: "科学验证通过",
  passed: "科学验证通过",
  conditional: "条件通过",
  candidate: "实验候选",
  experimental: "实验候选",
  failed: "未通过",
  blocked: "未通过",
  legacy: "既有基线",
  descriptive: "描述性证据",
  unknown: "证据待登记",
};

const runtimeLabels: Record<string, string> = {
  runnable: "可运行",
  precomputed_only: "仅冻结成果",
  adapter_required: "待运行适配器",
  blocked: "禁止运行",
  unavailable: "不可运行",
  unknown: "运行状态待登记",
};

const evidenceLabels: Record<string, string> = {
  whole_well_oof: "整井 OOF",
  spatial_oof: "空间块 OOF",
  fixed_holdout: "固定留出",
  sparse_labels: "稀疏解释标签",
  weak_labels: "弱标签",
  synthetic: "合成数据",
  descriptive: "描述性评估",
  legacy_external: "外部既有模型",
};

const taskLabels: Record<string, string> = {
  alignment: "井震标定",
  fault: "断层识别",
  horizon: "层位识别",
  strata: "地层实例分割",
  facies_1d: "井侧沉积相分类",
  facies_3d: "三维地震相分割",
  well_property: "储层物性预测",
  fluid_interpretation: "流体解释",
  fracture_development: "井侧裂缝发育排序",
  channel: "河道地质体识别",
  karst: "岩溶地质体识别",
  hydrocarbon_evidence: "含烃证据",
};

const archivedPropertyCompletionReleaseIds = new Set([
  "wellfuse_p18_den",
  "wellfuse_p18_por",
  "wellfuse_p18_log_perm",
  "wellfuse_p18_sw",
  "wellfuse_p18_vsh",
]);
const releases = computed(() => (props.catalog?.releases || []).filter((release) =>
  !isWellPropertyCompletionModelId(release.model_id)
  && !archivedPropertyCompletionReleaseIds.has(release.id),
));
const filteredReleases = computed(() => {
  const query = search.value.trim().toLowerCase();
  return releases.value.filter((release) => {
    if (scientificFilter.value !== "all" && release.scientific_status !== scientificFilter.value) return false;
    if (runtimeFilter.value !== "all" && release.runtime_status !== runtimeFilter.value) return false;
    if (!query) return true;
    const haystack = [
      release.id,
      release.name,
      release.display_name,
      release.family,
      release.task,
      release.task_id,
      release.description,
      release.summary,
      release.evidence_class,
      ...(release.outputs || []),
    ].filter(Boolean).join(" ").toLowerCase();
    return haystack.includes(query);
  });
});

const scientificOptions = computed(() =>
  [...new Set(releases.value.map((release) => release.scientific_status).filter(Boolean))].sort(),
);
const runtimeOptions = computed(() =>
  [...new Set(releases.value.map((release) => release.runtime_status).filter(Boolean))].sort(),
);
const validatedCount = computed(() => releases.value.filter((release) =>
  ["validated", "passed"].includes(release.scientific_status),
).length);
const runnableCount = computed(() => releases.value.filter((release) => release.runtime_status === "runnable").length);
const frozenOnlyCount = computed(() => releases.value.filter((release) => release.runtime_status === "precomputed_only").length);

function labelForScientific(status: string): string {
  return scientificLabels[status] || status || scientificLabels.unknown;
}

function labelForRuntime(status: string): string {
  return runtimeLabels[status] || status || runtimeLabels.unknown;
}

function labelForEvidence(evidence: string): string {
  return publicModelText(evidenceLabels[evidence] || evidence, "证据待登记");
}

function statusTone(status: string): string {
  if (["validated", "passed", "runnable"].includes(status)) return "positive";
  if (["conditional", "candidate", "experimental", "adapter_required", "descriptive"].includes(status)) return "caution";
  if (["failed", "blocked", "unavailable"].includes(status)) return "negative";
  if (["legacy", "precomputed_only"].includes(status)) return "neutral";
  return "unknown";
}

function releaseTitle(release: ArtifactRelease): string {
  return releasePresentationName(release);
}

function taskLabel(release: ArtifactRelease): string {
  const key = release.task_id || release.task || "";
  return publicModelText(taskLabels[key] || release.task || release.family, "未分类任务");
}

function artifactName(path: string): string {
  return path.replaceAll("\\", "/").split("/").filter(Boolean).at(-1) || path;
}

function publicArtifactLabel(value: unknown, fallback = "成果文件"): string {
  const raw = String(value || fallback);
  return publicModelText(artifactName(raw), fallback);
}

function normalizeArtifact(value: ReleaseArtifactValue, key: string, index: number): ReleaseArtifact | null {
  if (typeof value === "string") return { name: key || artifactName(value), path: value };
  if (!value) return null;
  return {
    ...value,
    name: value.name || key || value.role || value.kind || `成果 ${index + 1}`,
  };
}

function normalizeCollection(collection?: ReleaseArtifactCollection): ReleaseArtifact[] {
  if (!collection) return [];
  if (Array.isArray(collection)) {
    return collection
      .map((value, index) => normalizeArtifact(value, "", index))
      .filter((value): value is ReleaseArtifact => Boolean(value));
  }
  return Object.entries(collection)
    .map(([key, value], index) => normalizeArtifact(value, key, index))
    .filter((value): value is ReleaseArtifact => Boolean(value));
}

function artifactsFor(release: ArtifactRelease): ReleaseArtifact[] {
  const values = [
    ...normalizeCollection(release.precomputed_artifacts),
    ...normalizeCollection(release.artifacts),
  ];
  const seen = new Set<string>();
  return values.filter((artifact) => {
    const key = artifact.download_url || artifact.downloadable_url || artifact.path || artifact.name || JSON.stringify(artifact);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function metricEntries(release: ArtifactRelease): Array<[string, string | number | boolean | null]> {
  return Object.entries(release.metrics || {}).slice(0, 6);
}

function formatMetric(value: string | number | boolean | null): string {
  if (value === null) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    return Math.abs(value) < 0.01 && value !== 0 ? value.toExponential(2) : value.toLocaleString("zh-CN", { maximumFractionDigits: 4 });
  }
  return String(value);
}

function formatBytes(value?: number): string {
  if (!value || value < 0) return "";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit ? 2 : 0)} ${units[unit]}`;
}

function canRun(release: ArtifactRelease): boolean {
  return release.runtime_status === "runnable"
    && Boolean(release.task_id && release.model_id)
    && props.uiRunnableModelIds.includes(String(release.model_id));
}

function isApiOnlyRunnable(release: ArtifactRelease): boolean {
  return release.runtime_status === "runnable"
    && Boolean(release.task_id && release.model_id)
    && !canRun(release);
}
</script>

<template>
  <section class="section-panel release-center">
    <div class="section-heading release-heading">
      <div>
        <p class="release-kicker">ARTIFACT RELEASE CATALOG</p>
        <h2>模型与冻结成果发布中心</h2>
        <p>科学证据状态与工程运行状态分别管理；“存在 checkpoint”不再自动等同于模型可信或当前可运行。</p>
      </div>
      <span class="count-badge">{{ releases.length }} 个发布</span>
    </div>

    <div v-if="loading" class="release-empty">正在读取只读发布目录…</div>
    <div v-else-if="error" class="release-error">
      <strong>发布目录暂不可用</strong>
      <p>{{ publicModelText(error) }}</p>
      <small>原有三维断层识别、地层实例分割模型中心与预测入口不受影响。</small>
    </div>
    <template v-else>
      <div class="release-summary">
        <article><span>全部发布</span><strong>{{ releases.length }}</strong><small>模型、冻结结果与既有基线</small></article>
        <article><span>科学验证通过</span><strong>{{ validatedCount }}</strong><small>仍需服从适用域和限制</small></article>
        <article><span>当前可运行</span><strong>{{ runnableCount }}</strong><small>具备完整运行适配器</small></article>
        <article><span>仅冻结成果</span><strong>{{ frozenOnlyCount }}</strong><small>可浏览、不可直接推断新数据</small></article>
      </div>

      <div class="release-filters" aria-label="发布筛选">
        <label>
          <span>搜索</span>
          <input v-model="search" type="search" placeholder="任务、模型、输出或证据" />
        </label>
        <label>
          <span>科学状态</span>
          <select v-model="scientificFilter">
            <option value="all">全部科学状态</option>
            <option v-for="status in scientificOptions" :key="status" :value="status">{{ labelForScientific(status) }}</option>
          </select>
        </label>
        <label>
          <span>运行状态</span>
          <select v-model="runtimeFilter">
            <option value="all">全部运行状态</option>
            <option v-for="status in runtimeOptions" :key="status" :value="status">{{ labelForRuntime(status) }}</option>
          </select>
        </label>
      </div>

      <div v-if="filteredReleases.length" class="release-grid">
        <article v-for="release in filteredReleases" :key="release.id" class="release-card">
          <header>
            <div>
              <span class="release-task">{{ taskLabel(release) }}</span>
              <h3>{{ releaseTitle(release) }}</h3>
            </div>
            <span v-if="release.legacy" class="legacy-badge">既有模型保留</span>
          </header>

          <div class="status-row">
            <span :class="['status-chip', statusTone(release.scientific_status)]">
              科学 · {{ labelForScientific(release.scientific_status) }}
            </span>
            <span :class="['status-chip', statusTone(release.runtime_status)]">
              工程 · {{ labelForRuntime(release.runtime_status) }}
            </span>
            <span class="evidence-chip">证据 · {{ labelForEvidence(release.evidence_class) }}</span>
          </div>

          <p class="release-description">{{ publicModelText(release.summary || release.description, "该发布尚未登记摘要。") }}</p>

          <div v-if="parameterBreakdown(release)" class="parameter-breakdown">
            <article>
              <small>4-bit 多模态主体</small>
              <strong>{{ formatParameters(parameterBreakdown(release)!.core) }}</strong>
              <span>core</span>
            </article>
            <b>+</b>
            <article>
              <small>冻结概率标定保护集成 · {{ parameterBreakdown(release)!.members }}成员</small>
              <strong>{{ formatParameters(parameterBreakdown(release)!.anchor) }}</strong>
              <span>anchor ensemble</span>
            </article>
            <b>=</b>
            <article class="deployment-total">
              <small>部署加载总量</small>
              <strong>{{ formatParameters(parameterBreakdown(release)!.deployment) }}</strong>
              <span>deployment</span>
            </article>
          </div>

          <div v-if="taskAdapterEvidence(release).length" class="task-adapter-evidence">
            <strong>任务 Adapter 证据与效果归因</strong>
            <article v-for="evidence in taskAdapterEvidence(release)" :key="evidence.id">
              <header><b>{{ publicModelText(evidence.label) }}</b><span>{{ publicModelText(evidence.status) }}</span></header>
              <p>{{ publicModelText(evidence.metric) }}</p>
              <small>{{ publicModelText(evidence.attribution) }}</small>
            </article>
          </div>

          <button
            v-if="canRun(release)"
            type="button"
            class="release-run-button"
            @click="emit('run', release.task_id || '', release.model_id || '')"
          >
            进入在线推理
          </button>
          <p v-else-if="isApiOnlyRunnable(release)" class="release-api-only">
            运行适配器已登记；当前发布仅通过标准 API 或专用研究流程启动，不显示无效的页面跳转。
          </p>

          <dl v-if="release.inputs?.length || release.outputs?.length" class="release-contract">
            <div v-if="release.inputs?.length"><dt>合法输入</dt><dd>{{ publicModelText(release.inputs.join(" / ")) }}</dd></div>
            <div v-if="release.outputs?.length"><dt>标准输出</dt><dd>{{ publicModelText(release.outputs.join(" / ")) }}</dd></div>
          </dl>

          <div v-if="metricEntries(release).length" class="release-metrics">
            <div v-for="[name, value] in metricEntries(release)" :key="name">
              <span>{{ publicModelText(name) }}</span><strong>{{ publicModelText(formatMetric(value)) }}</strong>
            </div>
          </div>

          <div v-if="release.warnings?.length || release.limitations?.length" class="release-warnings">
            <strong>适用限制</strong>
            <ul>
              <li v-for="warning in [...(release.warnings || []), ...(release.limitations || [])]" :key="warning">{{ publicModelText(warning) }}</li>
            </ul>
          </div>

          <details v-if="artifactsFor(release).length" class="artifact-list">
            <summary>冻结成果 {{ artifactsFor(release).length }} 项</summary>
            <div v-for="(artifact, index) in artifactsFor(release)" :key="artifact.id || artifact.download_url || artifact.downloadable_url || artifact.path || index" class="artifact-row">
              <div>
                <strong>{{ publicArtifactLabel(artifact.name || artifact.role || artifact.kind, `成果 ${index + 1}`) }}</strong>
                <code v-if="artifact.path">{{ publicArtifactLabel(artifact.path) }}</code>
                <small>{{ publicModelText([artifact.format || artifact.media_type, formatBytes(artifact.size_bytes), artifact.exists === false ? "文件未找到" : "", artifact.integrity_status, artifact.description].filter(Boolean).join(" · ")) }}</small>
              </div>
              <a v-if="artifact.download_url || artifact.downloadable_url" :href="artifact.download_url || artifact.downloadable_url" target="_blank" rel="noopener">打开 / 下载</a>
              <span v-else>只读清单</span>
            </div>
          </details>

          <footer>
            <span v-if="release.manifest_path">Manifest</span><code v-if="release.manifest_path">{{ publicArtifactLabel(release.manifest_path) }}</code>
            <span v-else-if="release.source_root">来源</span><code v-if="!release.manifest_path && release.source_root">{{ publicArtifactLabel(release.source_root, "项目成果源") }}</code>
          </footer>
          <details class="release-technical-identity">
            <summary>技术标识（用于 API 与审计）</summary>
            <code>{{ publicModelIdentifier(release.id) }}<template v-if="release.model_id"> · {{ publicModelIdentifier(release.model_id) }}</template><template v-if="release.version"> · {{ publicModelText(release.version) }}</template></code>
          </details>
        </article>
      </div>
      <div v-else class="release-empty">
        {{ releases.length ? "没有符合当前筛选条件的发布。" : "发布目录为空；既有运行模型仍显示在下方组件中心。" }}
      </div>

      <p v-if="catalog?.artifact_root" class="artifact-root"><strong>只读成果根目录</strong><code>{{ catalog.artifact_root }}</code></p>
    </template>
  </section>
</template>

<style scoped>
.release-center { overflow: hidden; }
.release-heading { align-items: flex-start; }
.release-kicker { margin: 0 0 5px; color: #2b6da9; font: 700 12px/1.2 ui-monospace, monospace; letter-spacing: .12em; }
.release-empty,
.release-error { padding: 26px; border: 1px dashed #cbd8e5; border-radius: 12px; color: #607286; background: #f8fafc; }
.release-error { border-color: #e7c5c2; color: #8a3c35; background: #fff8f7; }
.release-error p { margin: 6px 0; }
.release-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }
.release-summary article { padding: 15px; border: 1px solid #dce5ed; border-radius: 10px; background: linear-gradient(145deg, #fff, #f7fafc); }
.release-summary span,
.release-summary small { display: block; color: #6f7f90; font-size: 12px; }
.release-summary strong { display: block; margin: 5px 0; color: #183b5f; font-size: 24px; }
.release-filters { display: grid; grid-template-columns: minmax(240px, 1fr) repeat(2, minmax(170px, .36fr)); gap: 12px; margin-bottom: 16px; }
.release-filters label { display: grid; gap: 5px; color: #52687d; font-size: 12px; font-weight: 700; }
.release-filters input,
.release-filters select { width: 100%; min-height: 38px; padding: 8px 10px; border: 1px solid #ccd8e3; border-radius: 8px; color: #263b50; background: #fff; }
.release-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; }
.release-card { display: flex; flex-direction: column; min-width: 0; padding: 18px; border: 1px solid #dce5ed; border-radius: 12px; background: #fff; box-shadow: 0 7px 22px rgb(33 64 93 / 5%); }
.release-card > header { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.release-card h3 { margin: 5px 0; color: #183b5f; font-size: 18px; }
.release-card code { overflow-wrap: anywhere; color: #607286; font-size: 12px; }
.release-api-only { margin: 12px 0 0; padding: 9px 11px; border: 1px solid #d8e3ec; border-radius: 8px; color: #607286; background: #f7fafc; font-size: 12px; }
.release-task { color: #2b6da9; font-size: 12px; font-weight: 800; letter-spacing: .04em; }
.legacy-badge { flex: none; padding: 4px 8px; border-radius: 999px; color: #66563d; background: #f3eee4; font-size: 12px; font-weight: 700; }
.status-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.status-chip,
.evidence-chip { padding: 5px 8px; border-radius: 999px; font-size: 12px; font-weight: 750; }
.status-chip.positive { color: #176448; background: #e5f5ed; }
.status-chip.caution { color: #7d5a18; background: #fff3d6; }
.status-chip.negative { color: #9a3f3a; background: #fde9e7; }
.status-chip.neutral { color: #526477; background: #edf1f5; }
.status-chip.unknown,
.evidence-chip { color: #45627f; background: #edf4fa; }
.release-description { min-height: 40px; margin: 13px 0; color: #5d6f80; font-size: 13px; line-height: 1.55; }
.parameter-breakdown { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr); gap: 7px; align-items: center; margin: 0 0 13px; }
.parameter-breakdown > b { color: #6d8193; font-size: 17px; }
.parameter-breakdown article { padding: 9px; border: 1px solid #dce6ee; border-radius: 8px; background: #f6f9fb; }
.parameter-breakdown article.deployment-total { border-color: #9cc5e6; background: #edf6fc; }
.parameter-breakdown small,
.parameter-breakdown strong,
.parameter-breakdown span { display: block; }
.parameter-breakdown small { color: #687c8f; font-size: 12px; }
.parameter-breakdown strong { margin: 3px 0; color: #173f62; font-size: 12px; overflow-wrap: anywhere; }
.parameter-breakdown span { color: #8795a2; font: 12px/1.3 ui-monospace, monospace; }
.task-adapter-evidence { display: grid; gap: 7px; margin: 0 0 13px; padding: 11px; border: 1px solid #d9e5ee; border-radius: 9px; background: #f8fbfd; }
.task-adapter-evidence > strong { color: #315d84; font-size: 12px; }
.task-adapter-evidence article { padding: 9px; border-radius: 7px; background: #fff; }
.task-adapter-evidence header { display: flex; justify-content: space-between; gap: 8px; color: #294b69; font-size: 12px; }
.task-adapter-evidence header span { color: #287555; }
.task-adapter-evidence p { margin: 5px 0; color: #334f66; font-size: 12px; }
.task-adapter-evidence small { color: #687b8d; font-size: 12px; line-height: 1.45; }
.release-run-button { align-self: flex-start; margin: 0 0 13px; padding: 8px 12px; border: 0; border-radius: 7px; color: #fff; background: #286ba5; cursor: pointer; font-size: 12px; font-weight: 800; }
.release-run-button:hover { background: #1f5789; }
.release-contract { display: grid; gap: 8px; margin: 0 0 12px; }
.release-contract div { display: grid; grid-template-columns: 58px 1fr; gap: 8px; }
.release-contract dt { color: #778899; font-size: 12px; }
.release-contract dd { margin: 0; color: #334a60; font-size: 12px; overflow-wrap: anywhere; }
.release-metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; margin-bottom: 12px; }
.release-metrics div { padding: 8px; border-radius: 7px; background: #f4f7fa; }
.release-metrics span,
.release-metrics strong { display: block; overflow-wrap: anywhere; }
.release-metrics span { color: #718295; font-size: 12px; }
.release-metrics strong { margin-top: 3px; color: #214360; font-size: 12px; }
.release-warnings { margin-bottom: 12px; padding: 10px 12px; border-left: 3px solid #d8a23c; color: #6c552a; background: #fff9eb; font-size: 12px; }
.release-warnings ul { margin: 6px 0 0; padding-left: 17px; }
.release-warnings li + li { margin-top: 4px; }
.artifact-list { margin-top: auto; border-top: 1px solid #e5ebf1; }
.artifact-list summary { padding: 12px 0 5px; color: #315d84; cursor: pointer; font-size: 12px; font-weight: 700; }
.artifact-row { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 9px 0; border-top: 1px solid #edf1f4; }
.artifact-row div { min-width: 0; }
.artifact-row strong,
.artifact-row code,
.artifact-row small { display: block; }
.artifact-row strong { color: #31485e; font-size: 12px; }
.artifact-row small { margin-top: 3px; color: #8693a0; font-size: 12px; }
.artifact-row a,
.artifact-row > span { flex: none; color: #286ba5; font-size: 12px; font-weight: 700; }
.artifact-row > span { color: #8996a2; }
.release-card > footer { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 8px; margin-top: 12px; padding-top: 10px; border-top: 1px solid #e5ebf1; }
.release-card > footer span { color: #718295; font-size: 12px; }
.artifact-root { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 10px; margin: 16px 0 0; padding: 10px 12px; border-radius: 8px; color: #5d6f80; background: #f4f7fa; font-size: 12px; }
.artifact-root code { overflow-wrap: anywhere; }
@media (max-width: 1050px) {
  .release-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .release-grid { grid-template-columns: 1fr; }
}
@media (max-width: 720px) {
  .release-filters,
  .release-summary { grid-template-columns: 1fr; }
  .release-card > header { flex-direction: column; }
  .parameter-breakdown { grid-template-columns: 1fr; }
  .parameter-breakdown > b { text-align: center; }
}
</style>
