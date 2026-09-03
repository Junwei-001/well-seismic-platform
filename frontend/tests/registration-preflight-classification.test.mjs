import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { build } from "esbuild";

const temporaryDirectory = await mkdtemp(path.join(os.tmpdir(), "wellfuse-preflight-"));
const bundlePath = path.join(temporaryDirectory, "api.mjs");
await build({
  entryPoints: [fileURLToPath(new URL("../src/api.ts", import.meta.url))],
  outfile: bundlePath,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "es2022",
});
const {
  ApiRequestError,
  classifyRegistrationPreflightFailure,
} = await import(`${pathToFileURL(bundlePath).href}?${Date.now()}`);

function preflightError(detail) {
  return new ApiRequestError(409, detail.reason || detail.code, {
    responseBody: { detail },
  });
}

test.after(async () => {
  await rm(temporaryDirectory, { recursive: true, force: true });
});

test("source quality requiring a new snapshot is not mislabeled as integrity drift", () => {
  const result = classifyRegistrationPreflightFailure(preflightError({
    code: "trajectory_degraded_to_head_only",
    category: "source_quality_unavailable",
    horizontal_fallback_allowed: false,
    requires_new_snapshot: true,
    reason: "轨迹站点未确认",
  }));
  assert.equal(result, "needs_preparation");
});

test("only explicit integrity and semantic categories are integrity blocks", () => {
  assert.equal(classifyRegistrationPreflightFailure(preflightError({
    code: "source_snapshot_integrity_verification_failed",
    category: "source_snapshot_integrity",
    horizontal_fallback_allowed: false,
    requires_new_snapshot: true,
  })), "blocked_integrity");
  assert.equal(classifyRegistrationPreflightFailure(preflightError({
    code: "source_snapshot_semantic_drift",
    category: "source_snapshot_semantic_drift",
    horizontal_fallback_allowed: false,
    requires_new_snapshot: true,
  })), "blocked_integrity");
  assert.equal(classifyRegistrationPreflightFailure(preflightError({
    code: "source_snapshot_integrity_verification_failed",
    category: "source_snapshot_integrity",
    horizontal_fallback_allowed: true,
    requires_new_snapshot: true,
  })), "blocked_integrity");
});

test("explicit horizontal fallback permission remains required", () => {
  assert.equal(classifyRegistrationPreflightFailure(preflightError({
    code: "formal_and_native_relative_contract_unavailable",
    category: "formal_contract_unavailable",
    horizontal_fallback_allowed: true,
    requires_new_snapshot: false,
  })), "horizontal_only");
  assert.equal(classifyRegistrationPreflightFailure(preflightError({
    code: "formal_and_native_relative_contract_unavailable",
    category: "formal_contract_unavailable",
    horizontal_fallback_allowed: false,
    requires_new_snapshot: false,
  })), "failed");
});
