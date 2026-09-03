import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { build } from "esbuild";

const temporaryDirectory = await mkdtemp(path.join(os.tmpdir(), "wellfuse-layerpulse-domain-"));
const bundlePath = path.join(temporaryDirectory, "layerpulse.mjs");
await build({
  entryPoints: [fileURLToPath(new URL("../src/domain/layerPulse.ts", import.meta.url))],
  outfile: bundlePath,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "es2022",
});

const layerPulse = await import(`${pathToFileURL(bundlePath).href}?${Date.now()}`);

test.after(async () => {
  await rm(temporaryDirectory, { recursive: true, force: true });
});

test("single-checkpoint output directory exposes the canonical eleven heads", () => {
  const catalog = layerPulse.layerPulseOutputCatalog;
  assert.equal(layerPulse.layerPulseModelContract.parameterCount, 174_697_519);
  assert.equal(layerPulse.layerPulseModelContract.fFinalChannels, 96);
  assert.equal(layerPulse.layerPulseModelContract.headCount, 11);
  assert.equal(layerPulse.layerPulseModelContract.timeDepthRequiredAtForward, false);
  assert.equal(catalog.length, 11);
  assert.equal(catalog.filter((output) => output.kind === "classification").length, 6);
  assert.deepEqual(catalog.map((output) => output.key), [
    "fault_logits",
    "unconformity_logits",
    "facies_logits",
    "channel_logits",
    "karst_logits",
    "rgt",
    "impedance",
    "porosity",
    "well_match",
    "connectivity_logits",
    "uncertainty",
  ]);
});

test("every classification keeps background-inclusive logits and direct argmax", () => {
  for (const output of layerPulse.layerPulseOutputCatalog.filter((item) => item.kind === "classification")) {
    assert.equal(output.channels, output.classes.length);
    assert.equal(output.backgroundIndex, 0);
    assert.equal(output.classes[0].index, 0);
    assert.equal(output.classes[0].id, "background");
    assert.equal(output.classes[0].background, true);
    assert.equal(output.decode, "direct_argmax_dim1");
    assert.equal(output.usesThreshold, false);
  }
  const facies = layerPulse.layerPulseOutputByKey("facies_logits");
  assert.equal(facies.channels, 7);
  assert.deepEqual(facies.classes.map((item) => item.id), [
    "background",
    "upper_ns",
    "middle_ns",
    "lower_ns",
    "rijnland_chalk",
    "scruff",
    "zechstein",
  ]);
});

test("support summary fails closed and respects required versus optional blocks", () => {
  assert.deepEqual(layerPulse.summarizeLayerPulseSupport(null), {
    status: "blocked",
    label: "等待支持核验",
    detail: "尚未收到与当前 SourceSnapshot 绑定的支持能力收据。",
    readyCount: 0,
    degradedCount: 0,
    blockedCount: 0,
  });

  const receipt = {
    contract_version: "layerpulse.platform-support.v1",
    snapshot_id: "snapshot-1",
    model_id: layerPulse.LAYER_PULSE_MODEL_ID,
    status: "ready",
    warnings: [],
    checks: [
      { id: "seismic", label: "三维地震", status: "ready", detail: "可读", required: true },
      { id: "logs", label: "测井", status: "blocked", detail: "未提供", required: false },
    ],
  };
  assert.equal(layerPulse.summarizeLayerPulseSupport(receipt).status, "degraded");
  receipt.checks[0].status = "blocked";
  assert.equal(layerPulse.summarizeLayerPulseSupport(receipt).status, "blocked");
});

test("LayerPulse task lifecycle is independent and only queued/running are active", () => {
  assert.equal(layerPulse.createIdleLayerPulseTaskState().status, "idle");
  assert.equal(layerPulse.isLayerPulseTaskActive("queued"), true);
  assert.equal(layerPulse.isLayerPulseTaskActive("running"), true);
  assert.equal(layerPulse.isLayerPulseTaskActive("completed"), false);
  assert.equal(layerPulse.isLayerPulseTaskActive("failed"), false);
});
