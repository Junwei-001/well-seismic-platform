import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { build } from "esbuild";

const temporaryDirectory = await mkdtemp(path.join(os.tmpdir(), "wellfuse-source-statistics-"));
const bundlePath = path.join(temporaryDirectory, "source-data-statistics.mjs");
await build({
  entryPoints: [fileURLToPath(new URL("../src/domain/sourceDataStatistics.ts", import.meta.url))],
  outfile: bundlePath,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "es2022",
});
const { buildSourceDataStatistics } = await import(`${pathToFileURL(bundlePath).href}?${Date.now()}`);

test.after(async () => {
  await rm(temporaryDirectory, { recursive: true, force: true });
});

test("builds seismic and well statistics from authoritative preparation fields", () => {
  const result = {
    summary: { wells: 2, seismic_files: 2, log_files: 2 },
    assets: [
      { role: "seismic", size: 1_000 },
      { role: "SEISMIC", size: 2_000 },
      { role: "well_logs", size: 300 },
      { role: "well_log", size: 400 },
      { role: "well_metadata", size: 9_000 },
    ],
    seismic: [
      {
        name: "small.sgy",
        dimension: "3D",
        trace_count: 100,
        samples_per_trace: 500,
        sample_interval_ms: 4,
        inline_count: 10,
        crossline_count: 10,
        grid_coverage: 0.9,
      },
      {
        name: "main.sgy",
        dimension: "3D",
        trace_count: 300,
        samples_per_trace: 1_000,
        sample_interval_ms: 2,
        inline_count: 20,
        crossline_count: 15,
        grid_coverage: 0.98,
      },
    ],
    wells: [
      {
        well_uid: "W-1",
        name: "Well 1",
        trajectory_count: 1,
        logs: [
          { samples: 120, curves: ["GR", "DTC"] },
          { samples: 30, curves: ["GR", "RHOB"] },
        ],
      },
      {
        well_uid: "W-2",
        name: "Well 2",
        trajectory_count: 0,
        logs: [{ samples: 80, curves: ["DTC"] }],
      },
    ],
    data_snapshot: {
      canonical_data: { seismic_geometry: { readable: 1 } },
    },
    visualization_preview: {
      trajectories: [{ name: "W-1", geometryType: "deviated" }],
      wellGeometrySummary: {
        counts: { vertical: 0, deviated: 1, horizontal: 0 },
      },
      wellLogs: [
        { wellName: "Well 1", curves: [{ id: "GR", label: "Gamma ray" }] },
        { wellName: "Well 2", curves: [{ id: "NPHI", label: "Neutron" }] },
      ],
    },
  };

  const statistics = buildSourceDataStatistics(result);

  assert.deepEqual(statistics.seismic, {
    fileCount: 2,
    readableFileCount: 1,
    totalSizeBytes: 3_000,
    totalTraceCount: 400,
    primaryVolume: {
      name: "main.sgy",
      dimension: "3D",
      inlineCount: 20,
      crosslineCount: 15,
      samplesPerTrace: 1_000,
      sampleIntervalMs: 2,
      gridCoverage: 0.98,
    },
  });
  assert.equal(statistics.wells.wellCount, 2);
  assert.equal(statistics.wells.logFileCount, 2);
  assert.equal(statistics.wells.totalSizeBytes, 700);
  assert.equal(statistics.wells.totalLogSamples, 230);
  assert.deepEqual([...statistics.wells.curveIdentifiers].sort(), ["DTC", "GR", "NPHI", "RHOB"]);
  assert.equal(statistics.wells.trajectoryWellCount, 1);
  assert.deepEqual(statistics.wells.wellTypeDistribution, {
    vertical: 0,
    deviated: 1,
    horizontal: 0,
    unknown: 0,
  });
});

test("uses legacy preview inventories without inventing real log sample counts", () => {
  const statistics = buildSourceDataStatistics({
    summary: {},
    assets: [{ role: "well_logs", size: 64 }],
    visualization_preview: {
      seismicInventory: [{
        name: "legacy.sgy",
        dimension: "2D",
        trace_count: 50,
        samples_per_trace: 250,
        sample_interval_ms: 1,
        inline_count: 0,
        crossline_count: 0,
        grid_coverage: 0.5,
      }],
      wellLogs: [{
        name: "legacy.las",
        wellName: "Legacy Well",
        depth: new Array(99).fill(1),
        curves: [{ id: "GR", label: "Gamma ray" }],
      }],
      trajectories: [{ name: "Legacy Well", geometryType: "horizontal" }],
    },
  });

  assert.equal(statistics.seismic.fileCount, 1);
  assert.equal(statistics.seismic.readableFileCount, 1);
  assert.equal(statistics.wells.wellCount, 1);
  assert.equal(statistics.wells.logFileCount, 1);
  assert.equal(statistics.wells.totalLogSamples, 0);
  assert.deepEqual([...statistics.wells.curveIdentifiers], ["GR"]);
  assert.equal(statistics.wells.trajectoryWellCount, 1);
  assert.equal(statistics.wells.wellTypeDistribution.horizontal, 1);
});

test("empty and malformed legacy results never leak NaN", () => {
  const empty = buildSourceDataStatistics(null);
  const malformed = buildSourceDataStatistics({
    summary: { wells: Number.NaN, seismic_files: -3, log_files: Number.POSITIVE_INFINITY },
    assets: [
      { role: "seismic", size: Number.NaN },
      { role: "well_logs", size: -100 },
    ],
    seismic: [{
      name: "broken.sgy",
      dimension: "",
      trace_count: Number.NaN,
      samples_per_trace: -1,
      sample_interval_ms: Number.POSITIVE_INFINITY,
      inline_count: Number.NaN,
      crossline_count: -1,
      grid_coverage: Number.NaN,
    }],
    wells: [{
      well_uid: "broken",
      name: "Broken",
      trajectory_count: Number.NaN,
      logs: [{ samples: Number.NaN, curves: [] }],
    }],
  });

  assert.equal(empty.seismic.primaryVolume, null);
  assert.equal(empty.wells.curveIdentifiers.size, 0);
  assert.equal(malformed.seismic.totalSizeBytes, 0);
  assert.equal(malformed.seismic.totalTraceCount, 0);
  assert.equal(malformed.seismic.primaryVolume.sampleIntervalMs, null);
  assert.equal(malformed.seismic.primaryVolume.gridCoverage, null);
  assert.equal(malformed.wells.totalSizeBytes, 0);
  assert.equal(malformed.wells.totalLogSamples, 0);
  assert.doesNotMatch(JSON.stringify(malformed), /NaN|Infinity/);
});
