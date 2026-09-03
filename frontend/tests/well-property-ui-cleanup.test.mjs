import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { build } from "esbuild";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const modelCenterSource = await readFile(new URL("../src/components/models/ModelCenter.vue", import.meta.url), "utf8");
const releaseCenterSource = await readFile(new URL("../src/components/releases/ReleaseCenter.vue", import.meta.url), "utf8");
const presentationSource = await readFile(new URL("../src/domain/modelPresentation.ts", import.meta.url), "utf8");
const temporaryDirectory = await mkdtemp(path.join(os.tmpdir(), "wellfuse-property-ui-"));
const bundlePath = path.join(temporaryDirectory, "platform.mjs");
await build({
  entryPoints: [fileURLToPath(new URL("../src/domain/platform.ts", import.meta.url))],
  outfile: bundlePath,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "es2022",
});
const { primaryPredictionTasks } = await import(`${pathToFileURL(bundlePath).href}?${Date.now()}`);

test.after(async () => {
  await rm(temporaryDirectory, { recursive: true, force: true });
});

test("reservoir-property new runs expose prediction models, not legacy completion models", () => {
  const [task] = primaryPredictionTasks([{
    id: "well_property",
    name: "old name",
    short_name: "old name",
    description: "property",
    outputs: [],
    output: "",
    required_modalities: [],
    evaluation_metrics: [],
    order: 30,
    contract_version: "1.0",
    model_id: "wellfuse_den_p18",
    model_ids: ["wellfuse_den_p18", "wellfuse_por_p18", "wellfuse_por_northwest_fast"],
    runnable_model_ids: ["wellfuse_den_p18", "wellfuse_por_p18", "wellfuse_por_northwest_fast"],
    available: true,
    status: "可运行",
  }]);

  assert.deepEqual(task.model_ids, ["wellfuse_por_northwest_fast"]);
  assert.deepEqual(task.runnable_model_ids, ["wellfuse_por_northwest_fast"]);
  assert.equal(task.model_id, "wellfuse_por_northwest_fast");
});

test("all downstream well forms are snapshot-only and omit source selectors", () => {
  assert.match(appSource, /const isWellPropertyTask = computed\(\(\) => activePredictionTask\.value === "well_property"\)/);
  assert.match(appSource, /SNAPSHOT_ONLY_DOWNSTREAM_WELL_TASK_IDS = new Set\(\[[\s\S]*?"well_property"[\s\S]*?"fluid_interpretation"[\s\S]*?"facies_1d"[\s\S]*?"fracture_development"/);
  assert.match(appSource, /isSnapshotOnlyDownstreamWellTask\.value\s*\? \["sealed_snapshot"\]/);
  assert.match(appSource, /usesSealedSnapshotWellInput = computed\(\(\) =>\s*isSnapshotOnlyDownstreamWellTask\.value/);
  assert.doesNotMatch(appSource, />井数据来源</);
  assert.doesNotMatch(appSource, /内置已验证数据集|显式 LAS \/ CSV \/ TXT 文件|已验证数据集|推理井号/);
  assert.doesNotMatch(appSource, /usesRawWellInput|usesRegisteredWellInput|uploadRawWellFile|rawWellPathsText|fastWellDataset/);
  assert.doesNotMatch(appSource, /model-runtime-notice|compact-runtime-notice|<strong>运行范围<\/strong>/);
  assert.match(appSource, /<article v-if="!isSnapshotOnlyDownstreamWellTask">\s*<span>数据来源<\/span>/);
  assert.match(appSource, /source_task_id: sourceTaskId \|\| undefined/);
  assert.match(appSource, /prepared_view_task_id: sampleBuildingTaskId\.value \|\| undefined/);
  assert.match(appSource, /const created = runIntent\.usesSealedSnapshotWellInput[\s\S]*?options: \{ batch_size: 4 \}/);
});

test("the LayerPulse-only model center and release catalog exclude archived curve-completion models", () => {
  assert.match(modelCenterSource, /model\.id\s*===\s*LAYER_PULSE_MODEL_ID/);
  assert.doesNotMatch(modelCenterSource, /isWellPropertyCompletionModelId|publicModels/);
  assert.match(releaseCenterSource, /!isWellPropertyCompletionModelId\(release\.model_id\)/);
  assert.match(releaseCenterSource, /archivedPropertyCompletionReleaseIds/);
  assert.doesNotMatch(presentationSource, /井侧(?:密度|孔隙度|渗透率|含水饱和度|泥质含量)曲线补全/);
});
