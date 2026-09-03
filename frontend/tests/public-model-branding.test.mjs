import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { build } from "esbuild";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const presentationSource = await readFile(new URL("../src/domain/modelPresentation.ts", import.meta.url), "utf8");
const modelCenterSource = await readFile(new URL("../src/components/models/ModelCenter.vue", import.meta.url), "utf8");
const releaseCenterSource = await readFile(new URL("../src/components/releases/ReleaseCenter.vue", import.meta.url), "utf8");
const temporaryDirectory = await mkdtemp(path.join(os.tmpdir(), "wellfuse-public-model-"));
const bundlePath = path.join(temporaryDirectory, "model-presentation.mjs");
await build({
  entryPoints: [fileURLToPath(new URL("../src/domain/modelPresentation.ts", import.meta.url))],
  outfile: bundlePath,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "es2022",
});
const { publicModelIdentifier, publicModelText } = await import(`${pathToFileURL(bundlePath).href}?${Date.now()}`);

test.after(async () => {
  await rm(temporaryDirectory, { recursive: true, force: true });
});

test("prediction workbench uses project-facing visualization and model labels", () => {
  assert.match(appSource, /<h2>井震联合解释工作台<\/h2>/);
  assert.match(appSource, /项目可视化引擎 · 图层契约/);
  assert.doesNotMatch(appSource, /CIGVis 井震联合解释工作台|FaultSeg 输入门禁|SegFormer 批量|Mask2Former 批量/);
  assert.match(appSource, /基础分割批量/);
  assert.match(appSource, /精细分割批量/);
});

test("dynamic backend and catalog text crosses a public-name boundary", () => {
  assert.match(presentationSource, /export function publicModelText/);
  assert.match(presentationSource, /task: "慧眼三维断层识别"/);
  assert.match(presentationSource, /task: "慧眼区域增强断层识别"/);
  assert.match(appSource, /statusMessage\.value = publicModelText\(task\.message\)/);
  assert.match(appSource, /publicModelIdentifier\(predictionResult\.model_id\)/);
  assert.match(appSource, /publicModelIdentifier\(selectedPredictionModelId\)/);
  assert.match(appSource, /publicModelText\(artifact\.output_key\)/);
  assert.match(appSource, /publicModelText\(library\.name\)/);
  assert.doesNotMatch(modelCenterSource, /layerPulseModel(?:\.value)?\.description/);
  assert.match(modelCenterSource, /publicModelText\(layerPulseModel(?:\.value)?\.warnings(?:\?\.)?\.join/);
  assert.match(releaseCenterSource, /publicModelText\(release\.summary \|\| release\.description/);
  assert.match(releaseCenterSource, /publicModelIdentifier\(release\.model_id\)/);
  assert.match(releaseCenterSource, /publicArtifactLabel\(artifact\.path\)/);
});

test("public-name conversion executes against implementation names and technical ids", () => {
  const raw = "CIG-Bench CIGVis CIG Viser Plotly Matplotlib SFM SFM-Base-224 sfm_tokens sfm_view_mask MOMENT-1-small moment_tokens moment_patch_mask NCS ViT25D ViT3D FaultSeg FaultNet SurfaceSeg seismic_surface_seg SegFormer Mask2Former 3D U-Net u_net TorchScript Transformer";
  const converted = publicModelText(raw);
  for (const hidden of ["CIG-Bench", "CIGVis", " CIG ", "Viser", "Plotly", "Matplotlib", "SFM", "sfm_", "MOMENT", "moment_", "NCS", "ViT25D", "ViT3D", "FaultSeg", "FaultNet", "SurfaceSeg", "surface_seg", "SegFormer", "Mask2Former", "U-Net", "u_net", "TorchScript", "Transformer"]) {
    assert.equal(converted.includes(hidden), false, `${hidden} leaked from ${converted}`);
  }
  assert.equal(publicModelIdentifier("faultseg_3d"), "project-fault-3d-primary");
  assert.equal(publicModelIdentifier("faultnet_china_field"), "project-fault-3d-regional");
});

test("project-owned model and release names remain unchanged", () => {
  const owned = "LayerPulse WellFuse GeoPath GeoChronoGraph F3X200CF";
  assert.equal(publicModelText(owned), owned);
  assert.equal(publicModelText("SFMono-Regular"), "SFMono-Regular");
  assert.equal(publicModelIdentifier("layerpulse_geochronograph_f3x200cf"), "layerpulse_geochronograph_f3x200cf");
  assert.equal(publicModelIdentifier("wellfuse_align_geopath_tie_v1"), "wellfuse_align_geopath_tie_v1");
});

test("redundant prediction scope cards are absent while fault controls remain", () => {
  assert.doesNotMatch(appSource, /fault-seg-formal-scope|surface-seg-formal-scope|model-runtime-notice/);
  assert.doesNotMatch(appSource, /REPRESENTATIVE GRID · 128块快速预测|<strong>运行范围<\/strong>/);
  assert.match(appSource, /工区中心单块（默认）/);
  assert.doesNotMatch(appSource, /value="representative_grid_128"|128 个代表块（默认）/);
  assert.match(appSource, /全区连续重建/);
  assert.match(appSource, /window\.confirm\("全区断层预测通常约需 20–30 分钟/);
});
