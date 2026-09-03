export function estimateProgressRemainingSeconds(
  elapsedSeconds: number,
  progressPercent: number,
): number | null {
  if (!Number.isFinite(elapsedSeconds) || !Number.isFinite(progressPercent)) return null;
  if (elapsedSeconds < 5 || progressPercent < 2 || progressPercent >= 100) return null;
  const remaining = elapsedSeconds / progressPercent * (100 - progressPercent);
  return Number.isFinite(remaining) && remaining > 0 ? Math.max(5, Math.ceil(remaining)) : null;
}

export function countDownRemainingSeconds(
  baselineSeconds: number | null,
  elapsedSinceEstimateSeconds: number,
): number | null {
  if (baselineSeconds === null) return null;
  if (!Number.isFinite(baselineSeconds) || !Number.isFinite(elapsedSinceEstimateSeconds)) return null;
  const remaining = baselineSeconds - Math.max(0, elapsedSinceEstimateSeconds);
  return remaining >= 5 ? Math.ceil(remaining) : null;
}
