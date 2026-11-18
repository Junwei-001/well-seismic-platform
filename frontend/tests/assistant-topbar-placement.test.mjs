import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const platformSource = await readFile(new URL("../src/domain/platform.ts", import.meta.url), "utf8");
const assistantSource = await readFile(new URL("../src/components/assistant/FloatingAssistant.vue", import.meta.url), "utf8");
const apiSource = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const themeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");

test("Huiyan AI remains a first-level page and also opens from business pages", () => {
  const settingsIndex = platformSource.indexOf('{ id: "settings", label: "平台设置", section: "system" }');
  const assistantIndex = platformSource.indexOf('{ id: "assistant", label: "慧眼AI", section: "system" }');
  assert.ok(settingsIndex >= 0, "platform settings navigation item must exist");
  assert.ok(assistantIndex > settingsIndex, "Huiyan AI must follow platform settings");
  assert.match(platformSource, /\| "assistant";/);
  assert.match(platformSource, /assistant: \{\s*eyebrow: "慧眼AI"/);
  assert.match(appSource, /'assistant-content': activeView === 'assistant'/);
  assert.match(appSource, /v-show="activeView === 'assistant'"[\s\S]*?:visible="activeView === 'assistant'"/);
  assert.match(appSource, /class="assistant-drawer-launcher"/);
  assert.match(appSource, /aria-label="慧眼AI 对话侧栏"/);
  assert.match(appSource, /v-show="assistantDrawerOpen"/);
  assert.doesNotMatch(appSource, /class="assistant-drawer-toolbar"/);
  assert.match(assistantSource, /class="assistant-close"[^>]*aria-label="关闭慧眼助手"/);
  assert.match(appSource, /<FloatingAssistant[\s\S]*?closable[\s\S]*?@close="assistantDrawerOpen = false"/);
});

test("Huiyan AI shares one conversation between the full page and in-page sidebar", () => {
  assert.doesNotMatch(appSource, /assistant-launcher-slot/);
  assert.doesNotMatch(themeSource, /topbar-assistant-slot/);
  assert.doesNotMatch(assistantSource, /<Teleport|position:\s*fixed|startDrag|strata-assistant-launcher/);
  assert.match(themeSource, /\.main-content\.assistant-content \{[\s\S]*?height: calc\(100dvh - var\(--topbar-height\)\);[\s\S]*?overflow: hidden;/);
  assert.match(assistantSource, /\.strata-assistant-workspace\s*\{[^}]*flex-direction:\s*column;[^}]*height:\s*100%;/);
  assert.match(assistantSource, /\.strata-assistant-messages\s*\{[^}]*min-height:\s*0;[^}]*overflow-y:\s*auto;/);
  assert.match(apiSource, /conversation_id: conversationId \|\| null/);
  assert.match(assistantSource, /assistantMessages as messages/);
  assert.match(assistantSource, /class="strata-assistant-composer"/);
});
