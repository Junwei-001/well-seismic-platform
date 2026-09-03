import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const themeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");

test("knowledge and algorithm configuration is the first platform settings section", () => {
  assert.match(
    appSource,
    /<template v-else-if="activeView === 'settings'">\s*<section class="section-panel knowledge-config-panel"/,
  );
});

test("configuration libraries form a data-driven radial knowledge map", () => {
  assert.match(appSource, /const configurationLibraryNodes = computed/);
  assert.match(appSource, /CONFIGURATION_LIBRARY_POSITIONS\[library\.id\] \?\? fallbackPosition/);
  assert.match(appSource, /capabilities\.value\?\.configuration_libraries \?\? \[\]/);
  assert.match(appSource, /class="knowledge-orbit"[\s\S]*?role="list"/);
  assert.match(appSource, /v-for="library in configurationLibraryNodes"[\s\S]*?:key="library\.id"[\s\S]*?role="listitem"/);
  assert.match(appSource, /publicModelText\(library\.name\)/);
  assert.match(appSource, /publicModelText\(library\.file\)/);
  assert.doesNotMatch(appSource, /configurationLibraryNodes[\s\S]{0,80}\.(?:slice|filter)\(/);
});

test("radial decoration stays semantic, readable, and non-interactive", () => {
  assert.match(appSource, /aria-labelledby="knowledge-config-title"/);
  assert.match(appSource, /class="knowledge-orbit-links"[\s\S]*?aria-hidden="true"[\s\S]*?focusable="false"/);
  assert.match(appSource, /<h3>\{\{ publicModelText\(library\.name\) \}\}<\/h3>/);
  assert.match(themeSource, /\.knowledge-orbit-node-copy \{ min-width: 0; \}/);
  assert.match(themeSource, /\.knowledge-orbit-node-copy h3 \{[\s\S]*?overflow-wrap: anywhere/);
  assert.match(themeSource, /\.knowledge-orbit-node-copy code \{[\s\S]*?overflow-wrap: anywhere[\s\S]*?white-space: normal/);
  assert.doesNotMatch(appSource, /class="knowledge-orbit-node"[^>]*tabindex/);
});

test("radial layout floats on desktop and safely reflows for narrow screens", () => {
  assert.match(themeSource, /\.knowledge-orbit \{[\s\S]*?position: relative[\s\S]*?min-height: 610px/);
  assert.match(themeSource, /\.knowledge-orbit-node \{[\s\S]*?position: absolute[\s\S]*?top: var\(--knowledge-y\)[\s\S]*?left: var\(--knowledge-x\)/);
  assert.match(themeSource, /\.knowledge-orbit-links \{[\s\S]*?pointer-events: none/);
  assert.match(themeSource, /@keyframes knowledge-node-drift/);
  assert.match(themeSource, /@media \(max-width: 1180px\) \{[\s\S]*?\.knowledge-orbit \{[\s\S]*?grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/);
  assert.match(themeSource, /@media \(max-width: 620px\) \{[\s\S]*?\.knowledge-orbit \{ grid-template-columns: minmax\(0, 1fr\); \}/);
  assert.match(themeSource, /@media \(prefers-reduced-motion: reduce\) \{[\s\S]*?\.knowledge-orbit-link,[\s\S]*?animation: none/);
});

test("knowledge map uses the higher-contrast marine and sandstone palette", () => {
  assert.match(themeSource, /linear-gradient\(145deg, #103b59 0%, #205f7d 56%, #3e7f8c 100%\)/);
  assert.match(themeSource, /--knowledge-cyan: #987b45/);
  assert.match(themeSource, /stroke: rgb\(43 86 119 \/ 42%\)/);
  assert.match(themeSource, /border: 1px solid #91aec2/);
  assert.doesNotMatch(themeSource, /linear-gradient\(145deg, #277fe0 0%, #167ca6 54%, #128d8d 100%\)/);
});
