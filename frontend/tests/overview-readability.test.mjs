import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const themeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");
const readableStyleFiles = [
  "../src/styles.css",
  "../src/product-theme.css",
  "../src/components/models/ModelCenter.vue",
  "../src/components/releases/ReleaseCenter.vue",
  "../src/components/assistant/FloatingAssistant.vue",
];
const readableStyleSources = await Promise.all(readableStyleFiles.map(async (path) => ({
  path,
  source: await readFile(new URL(path, import.meta.url), "utf8"),
})));

function microFontDeclarations(path, source) {
  const withoutComments = source.replace(/\/\*[\s\S]*?\*\//g, "");
  // Geometry (width/height/stroke/SVG coordinates) is intentionally outside this text-only audit.
  const declarationPattern = /(?:^|[;{])\s*(font-size|font)\s*:\s*([^;{}]+)/gim;
  const violations = [];
  for (const match of withoutComments.matchAll(declarationPattern)) {
    const [, property, value] = match;
    const pixelValues = property.toLowerCase() === "font-size"
      ? [...value.matchAll(/(\d+(?:\.\d+)?)px\b/gi)]
      : [...value.matchAll(/(\d+(?:\.\d+)?)px\b/gi)].slice(0, 1);
    for (const pixelValue of pixelValues) {
      const size = Number(pixelValue[1]);
      if (size > 0 && size < 12) {
        const line = withoutComments.slice(0, match.index).split("\n").length;
        violations.push(`${path}:${line} ${property}: ${value.trim()}`);
      }
    }
  }
  return violations;
}

function cssBlock(source, opening) {
  const openingIndex = source.indexOf(opening);
  assert.notEqual(openingIndex, -1, `Missing CSS block: ${opening}`);
  const braceIndex = source.indexOf("{", openingIndex);
  let depth = 0;
  for (let index = braceIndex; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    if (depth === 0) return source.slice(braceIndex + 1, index);
  }
  assert.fail(`Unclosed CSS block: ${opening}`);
}

test("landing cover keeps a transparent three-column path dock", () => {
  assert.match(
    themeSource,
    /\.overview-primary-paths \{ grid-template-columns: repeat\(3, minmax\(0, 1fr\)\); background: transparent; \}/,
  );
  const dock = appSource.match(/<div class="overview-dock overview-primary-paths">([\s\S]*?)<\/div>/)?.[1] || "";
  const entries = [...dock.matchAll(
    /<button[^>]+@click="selectView\('([^']+)'\)"[^>]*><span>(\d+)<\/span><strong>([^<]+)<\/strong>/g,
  )].map((match) => ({ view: match[1], order: match[2], label: match[3] }));
  assert.deepEqual(entries, [
    { view: "preparation", order: "01", label: "数据与融合" },
    { view: "layerpulse", order: "02", label: "LayerPulse 多模态融合基础模型" },
    { view: "prediction", order: "03", label: "单任务推理模型（共享井震融合基座）" },
  ]);
  assert.doesNotMatch(appSource, /原始数据只读|来源全程记录|模型组件可替换/);
  assert.doesNotMatch(appSource, /overview-trust-line/);
  assert.match(appSource, /2026年度中国青年科技创新“揭榜挂帅”擂台赛作品/);
  assert.match(appSource, /首屏_井震智能解释中心\.jpg/);
  assert.match(
    appSource,
    /<div class="landing-brand">[\s\S]*?<div class="landing-brand-copy">\s*<strong>地层慧眼<\/strong>\s*<small>STRATA VISION<\/small>\s*<\/div>\s*<div\s+class="landing-event-title"/,
  );
  assert.doesNotMatch(appSource, /landing-brand-title/);
  assert.match(appSource, /class="landing-event-innovation">科技创新<\/strong>/);
  assert.match(appSource, /class="landing-event-command">揭榜挂帅<\/strong>/);
  assert.match(themeSource, /\.landing-brand \.brand-mark \{ width: 42px; height: 42px; \}/);
  assert.match(themeSource, /\.landing-brand > \.landing-event-title \{[^}]*font-size: clamp\(17px, 1\.25vw, 22px\)/s);
  assert.match(
    themeSource,
    /\.landing-event-title > span,\s*\.landing-event-title > strong,\s*\.landing-event-title > i \{[^}]*font-family: inherit;[^}]*font-size: inherit;[^}]*font-weight: inherit;/s,
  );
  assert.match(
    themeSource,
    /\.landing-event-title > span:first-child \{ margin-right: calc\(-1 \* var\(--landing-event-gap\)\); \}/,
  );
  assert.match(
    themeSource,
    /\.landing-brand > \.landing-event-title \{[^}]*flex: 0 1 auto;[^}]*justify-content: flex-start;[^}]*font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;[^}]*text-align: left;/s,
  );
  assert.doesNotMatch(themeSource, /\.landing-brand > \.landing-event-title \{[^}]*font-family: "STKaiti"/s);
});

test("primary theme surfaces no longer use micro text for controls and summaries", () => {
  assert.match(themeSource, /\.overview-dock strong \{[^}]*font-size: 16px/);
  assert.match(themeSource, /\.overview-dock small \{[^}]*font-size: 12px/);
  assert.match(themeSource, /\.prediction-entry-viewer > header span \{[^}]*font-size: 12px/);
  assert.match(themeSource, /\.prediction-control-panel \.form-select \{[^}]*font-size: 13px/);
  assert.match(themeSource, /\.seismic-dataset-list button strong \{[^}]*font-size: 13px/);
});

test("visible frontend text declarations use at least 12px", () => {
  const violations = readableStyleSources.flatMap(({ path, source }) =>
    microFontDeclarations(path, source));
  assert.deepEqual(
    violations,
    [],
    `Found visible text smaller than 12px:\n${violations.join("\n")}`,
  );
});

test("narrow and short covers keep copy and navigation in normal flow", () => {
  const constrainedCoverSource = cssBlock(
    themeSource,
    "@media (max-width: 900px), (max-height: 760px) {",
  );
  const narrowCoverSource = cssBlock(themeSource, "@media (max-width: 900px) {");
  assert.match(
    constrainedCoverSource,
    /html:has\(\.landing-shell\),\s*body:has\(\.landing-shell\),\s*#app:has\(\.landing-shell\) \{ min-width: 0; \}/,
  );
  assert.match(
    constrainedCoverSource,
    /\.landing-header \{[^}]*position: relative;[^}]*inset: auto;[^}]*height: 64px;[^}]*padding: 0;/s,
  );
  assert.match(
    constrainedCoverSource,
    /\.overview-copy \{[^}]*position: relative;[^}]*inset: auto;[^}]*width: min\(610px, 100%\);[^}]*transform: none;/s,
  );
  assert.match(
    constrainedCoverSource,
    /\.overview-dock \{[^}]*position: relative;[^}]*inset: auto;[^}]*width: 100%;[^}]*margin: 0;/s,
  );
  assert.match(narrowCoverSource, /\.overview-primary-paths \{ grid-template-columns: minmax\(0, 1fr\); \}/);
});

test("small covers place the independent event title on its own responsive row", () => {
  const mobileCoverSource = cssBlock(themeSource, "@media (max-width: 620px) {");
  assert.match(
    mobileCoverSource,
    /\.landing-brand > \.landing-event-title \{[^}]*grid-row: 2;[^}]*grid-column: 1 \/ -1;[^}]*flex-wrap: wrap;[^}]*white-space: normal;/s,
  );
});

test("medium-width covers place the event title below the unchanged brand copy", () => {
  assert.match(
    themeSource,
    /@media \(min-width: 621px\) and \(max-width: 780px\) \{[\s\S]*?\.landing-brand > \.landing-event-title \{[^}]*grid-row:\s*2;[^}]*grid-column:\s*1 \/ -1;/,
  );
});
