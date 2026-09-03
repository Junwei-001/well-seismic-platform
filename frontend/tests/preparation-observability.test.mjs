import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const apiSource = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const themeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");

test("running preparation exposes the current filename and its real size", () => {
  assert.match(apiSource, /current_item_size_bytes\?: number \| null/);
  assert.match(appSource, /const preparationCurrentItemSizeBytes = ref<number \| null>\(null\)/);
  assert.match(appSource, /detail\.current_item_size_bytes/);
  assert.match(appSource, /class="preparation-current-file"/);
  assert.match(appSource, /preparationCurrentItemName/);
  assert.match(appSource, /formatBytes\(preparationCurrentItemSizeBytes\)/);
});

test("the empty snapshot flow is replaced by source-data statistics", () => {
  assert.doesNotMatch(appSource, /<h2>数据快照<\/h2>/);
  assert.doesNotMatch(appSource, /class="snapshot-flow"/);
  assert.match(appSource, /aria-label="地震与测井数据统计"/);
  assert.match(appSource, /sourceDataStatistics\.seismic\.totalTraceCount/);
  assert.match(appSource, /sourceDataStatistics\.wells\.totalLogSamples/);
  assert.match(themeSource, /\.source-statistics-grid[\s\S]*?grid-template-columns: repeat\(2/);
});

test("pipeline timing is batch-level and unscoped stages keep their real status", () => {
  assert.match(apiSource, /completed_at\?: string/);
  assert.match(apiSource, /preparation_duration_seconds\?: number/);
  assert.match(appSource, /const finishedAt = task\.completed_at \|\| task\.updated_at \|\| ""/);
  assert.match(appSource, /class="preparation-run-meta"/);
  assert.match(appSource, /<time :datetime="preparationRunTiming\.startedAt"/);
  assert.match(appSource, /同一份盘点报告 · 逐节点时间未记录/);
  assert.doesNotMatch(appSource, /return "盘点完成"/);
  assert.match(appSource, /stage\.status === "未就绪" \|\| stage\.status === "阻断"\) return "未就绪"/);
});
