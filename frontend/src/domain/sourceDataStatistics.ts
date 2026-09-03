import type {
  SeismicAssetSummary,
  WellEntity,
  WellLogPreview,
  WorkflowResult,
} from "../api";

type SourceAsset = WorkflowResult["assets"][number];
type WellGeometryType = "vertical" | "deviated" | "horizontal";

export interface PrimarySeismicVolumeStatistics {
  name: string;
  dimension: string;
  inlineCount: number;
  crosslineCount: number;
  samplesPerTrace: number;
  sampleIntervalMs: number | null;
  gridCoverage: number | null;
}

export interface SeismicSourceStatistics {
  fileCount: number;
  readableFileCount: number;
  totalSizeBytes: number;
  totalTraceCount: number;
  primaryVolume: PrimarySeismicVolumeStatistics | null;
}

export interface WellSourceStatistics {
  wellCount: number;
  logFileCount: number;
  totalSizeBytes: number;
  totalLogSamples: number;
  curveIdentifiers: ReadonlySet<string>;
  trajectoryWellCount: number;
  wellTypeDistribution: Record<WellGeometryType | "unknown", number>;
}

export interface SourceDataStatistics {
  seismic: SeismicSourceStatistics;
  wells: WellSourceStatistics;
}

function nonNegative(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
}

function optionalNonNegative(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

function normalizedCount(value: unknown): number {
  return Math.floor(nonNegative(value));
}

function normalizeRole(value: unknown): string {
  return typeof value === "string" ? value.trim().toLocaleLowerCase() : "";
}

function assetsWithRole(assets: readonly SourceAsset[] | undefined, roles: ReadonlySet<string>): SourceAsset[] {
  if (!Array.isArray(assets)) return [];
  return assets.filter((asset) => asset && roles.has(normalizeRole(asset.role)));
}

function totalAssetSize(assets: readonly SourceAsset[]): number {
  return assets.reduce((total, asset) => total + nonNegative(asset.size), 0);
}

function seismicInventory(result: WorkflowResult): readonly SeismicAssetSummary[] {
  if (Array.isArray(result.seismic) && result.seismic.length) return result.seismic;
  const legacyInventory = result.visualization_preview?.seismicInventory;
  return Array.isArray(legacyInventory) ? legacyInventory : [];
}

function entityInventory(result: WorkflowResult): readonly WellEntity[] {
  if (Array.isArray(result.well_entities) && result.well_entities.length) return result.well_entities;
  return Array.isArray(result.wells) ? result.wells : [];
}

function logPreviews(result: WorkflowResult): readonly WellLogPreview[] {
  const previews = result.visualization_preview?.wellLogs;
  return Array.isArray(previews) ? previews : [];
}

function uniqueNamedCount(values: readonly unknown[]): number {
  const names = new Set<string>();
  for (const value of values) {
    if (typeof value === "string" && value.trim()) names.add(value.trim().toLocaleLowerCase());
  }
  return names.size;
}

function choosePrimaryVolume(inventory: readonly SeismicAssetSummary[]): SeismicAssetSummary | null {
  let primary: SeismicAssetSummary | null = null;
  let primaryWeight = -1;
  for (const seismic of inventory) {
    if (!seismic || typeof seismic !== "object") continue;
    const weight = nonNegative(seismic.trace_count) * Math.max(1, nonNegative(seismic.samples_per_trace));
    if (!primary || weight > primaryWeight) {
      primary = seismic;
      primaryWeight = weight;
    }
  }
  return primary;
}

function summarizeSeismic(result: WorkflowResult): SeismicSourceStatistics {
  const inventory = seismicInventory(result);
  const seismicAssets = assetsWithRole(result.assets, new Set(["seismic"]));
  const reportedFileCount = Math.max(
    normalizedCount(result.summary?.seismic_files),
    normalizedCount(result.summary?.registered_seismic_files),
    normalizedCount(result.data_snapshot?.canonical_data?.seismic_geometry?.registered),
  );
  const fileCount = Math.max(reportedFileCount, inventory.length, seismicAssets.length);
  const canonicalReadable = optionalNonNegative(
    result.data_snapshot?.canonical_data?.seismic_geometry?.readable,
  );
  const inferredReadable = inventory.filter(
    (item) => nonNegative(item?.trace_count) > 0 && nonNegative(item?.samples_per_trace) > 0,
  ).length;
  const readableFileCount = Math.min(
    fileCount,
    canonicalReadable === null ? inferredReadable : normalizedCount(canonicalReadable),
  );
  const primary = choosePrimaryVolume(inventory);

  return {
    fileCount,
    readableFileCount,
    totalSizeBytes: totalAssetSize(seismicAssets),
    totalTraceCount: inventory.reduce((total, item) => total + normalizedCount(item?.trace_count), 0),
    primaryVolume: primary
      ? {
          name: typeof primary.name === "string" ? primary.name : "",
          dimension: typeof primary.dimension === "string" ? primary.dimension : "",
          inlineCount: normalizedCount(primary.inline_count),
          crosslineCount: normalizedCount(primary.crossline_count),
          samplesPerTrace: normalizedCount(primary.samples_per_trace),
          sampleIntervalMs: optionalNonNegative(primary.sample_interval_ms),
          gridCoverage: optionalNonNegative(primary.grid_coverage),
        }
      : null,
  };
}

function geometryDistribution(result: WorkflowResult, trajectoryWellCount: number): WellSourceStatistics["wellTypeDistribution"] {
  const distribution: WellSourceStatistics["wellTypeDistribution"] = {
    vertical: 0,
    deviated: 0,
    horizontal: 0,
    unknown: 0,
  };
  const reported = result.visualization_preview?.wellGeometrySummary?.counts;
  if (reported && typeof reported === "object") {
    distribution.vertical = normalizedCount(reported.vertical);
    distribution.deviated = normalizedCount(reported.deviated);
    distribution.horizontal = normalizedCount(reported.horizontal);
  } else {
    for (const trajectory of result.visualization_preview?.trajectories || []) {
      const geometryType = trajectory?.geometryType;
      if (geometryType === "vertical" || geometryType === "deviated" || geometryType === "horizontal") {
        distribution[geometryType] += 1;
      } else {
        distribution.unknown += 1;
      }
    }
  }
  const classified = distribution.vertical + distribution.deviated + distribution.horizontal + distribution.unknown;
  distribution.unknown += Math.max(0, trajectoryWellCount - classified);
  return distribution;
}

function summarizeWells(result: WorkflowResult): WellSourceStatistics {
  const entities = entityInventory(result);
  const previews = logPreviews(result);
  const logAssets = assetsWithRole(result.assets, new Set(["well_log", "well_logs"]));
  const entityNames = entities.map((well) => well?.well_uid || well?.name);
  const previewNames = previews.map((preview) => preview?.wellName || preview?.name);
  // Preview display names are aliases of the entities rather than additional wells.
  const wellCount = Math.max(
    normalizedCount(result.summary?.wells),
    normalizedCount(result.data_snapshot?.canonical_data?.well_entities?.count),
    uniqueNamedCount(entityNames),
    uniqueNamedCount(previewNames),
  );
  const curveIdentifiers = new Set<string>();
  let totalLogSamples = 0;
  let entityTrajectoryWellCount = 0;

  for (const well of entities) {
    if (!well || typeof well !== "object") continue;
    if (normalizedCount(well.trajectory_count) > 0) entityTrajectoryWellCount += 1;
    for (const log of Array.isArray(well.logs) ? well.logs : []) {
      totalLogSamples += normalizedCount(log?.samples);
      for (const curve of Array.isArray(log?.curves) ? log.curves : []) {
        if (typeof curve === "string" && curve.trim()) curveIdentifiers.add(curve.trim().toLocaleUpperCase());
      }
    }
  }
  for (const preview of previews) {
    for (const curve of Array.isArray(preview?.curves) ? preview.curves : []) {
      const identifier = typeof curve?.id === "string" && curve.id.trim()
        ? curve.id.trim()
        : typeof curve?.label === "string"
          ? curve.label.trim()
          : "";
      if (identifier) curveIdentifiers.add(identifier.toLocaleUpperCase());
    }
  }

  const trajectoryNames = (result.visualization_preview?.trajectories || []).map((item) => item?.name);
  const trajectoryWellCount = Math.max(entityTrajectoryWellCount, uniqueNamedCount(trajectoryNames));
  return {
    wellCount,
    logFileCount: Math.max(
      normalizedCount(result.summary?.log_files),
      normalizedCount(result.data_snapshot?.canonical_data?.well_logs?.count),
      logAssets.length,
    ),
    totalSizeBytes: totalAssetSize(logAssets),
    totalLogSamples,
    curveIdentifiers,
    trajectoryWellCount,
    wellTypeDistribution: geometryDistribution(result, trajectoryWellCount),
  };
}

/**
 * Derive UI-ready descriptive statistics from a completed preparation result.
 * Runtime guards deliberately tolerate partial results restored from legacy snapshots.
 */
export function buildSourceDataStatistics(result: WorkflowResult | null | undefined): SourceDataStatistics {
  if (!result || typeof result !== "object") {
    return {
      seismic: {
        fileCount: 0,
        readableFileCount: 0,
        totalSizeBytes: 0,
        totalTraceCount: 0,
        primaryVolume: null,
      },
      wells: {
        wellCount: 0,
        logFileCount: 0,
        totalSizeBytes: 0,
        totalLogSamples: 0,
        curveIdentifiers: new Set<string>(),
        trajectoryWellCount: 0,
        wellTypeDistribution: { vertical: 0, deviated: 0, horizontal: 0, unknown: 0 },
      },
    };
  }
  return {
    seismic: summarizeSeismic(result),
    wells: summarizeWells(result),
  };
}
