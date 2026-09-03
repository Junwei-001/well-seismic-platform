import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const platformSource = await readFile(new URL("../src/domain/platform.ts", import.meta.url), "utf8");
const assistantSource = await readFile(new URL("../src/components/assistant/FloatingAssistant.vue", import.meta.url), "utf8");
const themeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");

test("Huiyan AI is a first-level page immediately below platform settings", () => {
  const settingsIndex = platformSource.indexOf('{ id: "settings", label: "平台设置", section: "system" }');
  const assistantIndex = platformSource.indexOf('{ id: "assistant", label: "慧眼AI", section: "system" }');
  assert.ok(settingsIndex >= 0, "platform settings navigation item must exist");
  assert.ok(assistantIndex > settingsIndex, "Huiyan AI must follow platform settings");
  assert.match(platformSource, /\| "assistant";/);
  assert.match(platformSource, /assistant: \{\s*eyebrow: "慧眼AI"/);
  assert.match(appSource, /'assistant-content': activeView === 'assistant'/);
  assert.match(appSource, /v-show="activeView === 'assistant'"[\s\S]*?:visible="activeView === 'assistant'"/);
  assert.match(appSource, /if \(view === "assistant"\) return capabilities\.value\?\.llm\.available \? "已连接" : "本地模式"/);
});

test("Huiyan AI uses one persistent full-page conversation instead of a floating topbar window", () => {
  assert.doesNotMatch(appSource, /assistant-launcher-slot/);
  assert.doesNotMatch(themeSource, /topbar-assistant-slot/);
  assert.doesNotMatch(assistantSource, /<Teleport|position:\s*fixed|startDrag|strata-assistant-launcher/);
  assert.match(themeSource, /\.main-content\.assistant-content \{[\s\S]*?height: calc\(100dvh - var\(--topbar-height\)\);[\s\S]*?overflow: hidden;/);
  assert.match(assistantSource, /\.strata-assistant-workspace \{[\s\S]*?height: 100%;[\s\S]*?grid-template-rows: auto auto minmax\(0, 1fr\) auto auto;/);
  assert.match(assistantSource, /\.strata-assistant-messages \{[\s\S]*?min-height: 0;[\s\S]*?overflow-y: auto;/);
  assert.match(assistantSource, /chatWithAssistant\(text, props\.taskId \|\| undefined\)/);
  assert.match(assistantSource, /class="strata-assistant-composer"/);
});
