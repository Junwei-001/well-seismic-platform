import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const apiSource = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");

test("cache reset remains available while tasks are running", () => {
  assert.doesNotMatch(appSource, /platformTaskRunning/);
  assert.match(appSource, /async function clearPlatformCaches\(\) \{\s+if \(cacheClearing\.value\) return;/);
  assert.equal(
    appSource.match(/:disabled="cacheClearing \|\| backendStatus !== 'online'"/g)?.length,
    2,
  );
});

test("cache reset explicitly stops active tasks before clearing", () => {
  assert.match(appSource, /确定停止全部任务、清空可重建缓存并重新开始/);
  assert.match(appSource, /所有排队或运行中的任务都会被取消，且不会自动恢复/);
  assert.match(appSource, /result\.tasks_cancelled > 0/);
  assert.match(apiSource, /tasks_cancelled: number/);
  assert.match(apiSource, /cancelled_tasks: Array</);
  assert.match(apiSource, /confirmation: "CLEAR_REGENERABLE_CACHE"/);
});
