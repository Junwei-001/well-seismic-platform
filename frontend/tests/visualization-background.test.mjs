import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const sharedSource = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
const productThemeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");
const adapterSource = await readFile(new URL("../../src/well_seismic/cigvis_adapter.py", import.meta.url), "utf8");
const standardResultsSource = await readFile(new URL("../../src/well_seismic/standard_results.py", import.meta.url), "utf8");

test("legacy and acceptance result canvases use the seismic blue-gray surround", () => {
  assert.match(
    sharedSource,
    /\.acceptance-cigvis \{[^}]*radial-gradient\(circle at 48% 42%, #f8fbfd 0%, #eef3f7 58%, #e3eaf0 100%\)/s,
  );
  assert.match(
    sharedSource,
    /\.plan-view-canvas svg \{[^}]*radial-gradient\(circle at 48% 42%, #f8fbfd 0%, #eef3f7 58%, #e3eaf0 100%\)/s,
  );
  assert.match(
    sharedSource,
    /\.sequence-chart svg \{[^}]*#f7fafc[^}]*#dfe8ef 80px/s,
  );
  assert.match(
    sharedSource,
    /\.fracture-interval-track > svg \{[^}]*#f7fafc[^}]*#dfe8ef 80px/s,
  );
});

test("prediction viewers and generated seismic workbenches share the blue-gray scene background", () => {
  assert.match(
    productThemeSource,
    /--prediction-scene-bg:\s*radial-gradient\(circle at 48% 42%, #f8fbfd 0%, #eef3f7 58%, #e3eaf0 100%\)/,
  );
  assert.match(productThemeSource, /\.prediction-live-frame \{[^}]*background:\s*var\(--prediction-scene-bg\)/s);
  assert.match(productThemeSource, /\.prediction-live-frame iframe \{[^}]*background:\s*var\(--prediction-scene-bg\)/s);
  assert.match(productThemeSource, /\.prediction-overview img \{[^}]*#eef3f7/s);
  assert.match(adapterSource, /--scene-bg:radial-gradient\(circle at 48% 42%,#f8fbfd 0%,#eef3f7 58%,#e3eaf0 100%\)/);
  assert.match(adapterSource, /\.volume-render-view\{\{[^}]*background:var\(--scene-bg\)/s);
  assert.match(adapterSource, /\.orthogonal-stage\{[^}]*background:var\(--scene-bg\)/s);
  assert.match(adapterSource, /context\.fillStyle='#eef3f7';context\.fillRect\(0,0,cw,ch\)/);
  assert.match(adapterSource, /colorbar_axis\.set_facecolor\("#eef3f7"\)/);
  assert.doesNotMatch(adapterSource, /ax\.set_facecolor\("#ffffff"\)/);
  assert.doesNotMatch(adapterSource, /context\.fillStyle='#fff';context\.fillRect\(0,0,cw,ch\)/);
  assert.match(
    standardResultsSource,
    /body\{\{margin:0;background:radial-gradient\(circle at 48% 42%,#f8fbfd 0%,#eef3f7 58%,#e3eaf0 100%\)/,
  );
  assert.doesNotMatch(standardResultsSource, /background:#08111f|background:#0c1b2d/);
});
