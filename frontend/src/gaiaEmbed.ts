// Keep the already-running local demonstration readable during migration.
// New Gaia only serves the job-based gateway; the old path selects compatibility explicitly.
const LEGACY_UI_PREFIX = "/api/workflow-runs/huiyan/ui/";
const pathname = typeof window === "undefined" ? "" : window.location.pathname;
export const isLegacyGaiaEmbedded = pathname.startsWith(LEGACY_UI_PREFIX);
export const GAIA_UI_PREFIX = isLegacyGaiaEmbedded ? LEGACY_UI_PREFIX : "/api/plug/huiyan/ui/";
export const isGaiaEmbedded = pathname.startsWith(GAIA_UI_PREFIX);

export function nativeApiUrl(path: string): string {
  return isGaiaEmbedded ? GAIA_UI_PREFIX + path.replace(/^\//, "") : path;
}
