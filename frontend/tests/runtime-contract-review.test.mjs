import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const apiSource = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const platformSource = await readFile(new URL("../src/domain/platform.ts", import.meta.url), "utf8");
const stylesSource = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
const productThemeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");

test("runtime defaults are reviewed in a native modal after the actual read completes", () => {
  assert.match(appSource, /<dialog[\s\S]*?class="runtime-contract-dialog"/);
  assert.match(appSource, /offerRuntimeContractReview\(result, dataSnapshotTaskId\.value \|\| completedTaskId\)/);
  assert.match(appSource, /已按本次资产填入推荐参数/);
  assert.match(appSource, /runtimeContractReview\.time_depth_asset_count/);
  assert.match(appSource, /@cancel\.prevent="returnFromRuntimeContractReview"/);
  assert.match(appSource, /@keydown\.esc\.prevent\.stop="returnFromRuntimeContractReview"/);
  assert.match(appSource, /<form[\s\S]*?@submit\.prevent="confirmRuntimeContractReview"/);
  assert.match(appSource, /runtimeContractReviewPending[\s\S]*?确认运行参数/);
  assert.match(appSource, /本次确认声明[\s\S]*?runtimeContractAttestationText/);
});

test("confirmation seals and restores the derived snapshot before choosing the next view", () => {
  assert.match(apiSource, /\/api\/v1\/registration\/runtime-contract/);
  assert.match(apiSource, /confirmation: "CONFIRM_RUNTIME_CONTRACT"/);
  assert.match(apiSource, /attestation,/);
  assert.match(appSource, /const derivedTask = await getTask\(confirmed\.derived_snapshot_id\)/);
  assert.match(appSource, /restoreCompletedDataPreparationTask\(derivedTask/);
  assert.match(appSource, /sealedWellSeismicWorkflowReady\.value/);
  assert.match(appSource, /canEnterFusion \? "fusion" : "pipeline"/);
  assert.match(appSource, /type="submit"[\s\S]*?确认上述参数并继续/);
});

test("confirmation carries the explicit human survey declaration", () => {
  assert.match(appSource, /declaration_text: surveyAttestationDeclaration\(srdElevationM\)/);
  assert.match(appSource, /confirmed_at: new Date\(\)\.toISOString\(\)/);
  assert.match(appSource, /confirmRuntimeContract\([\s\S]*?runtimeContractDraft\.value,[\s\S]*?attestation/);
  assert.match(appSource, /submission\.fingerprint !== fingerprint/);
  assert.match(appSource, /if \(!submission\.confirmation\)/);
  assert.match(appSource, /runtimeContractSourceSnapshotId\.value === sourceSnapshotId/);
  assert.match(appSource, /loadStoredRuntimeContractSubmission\(sourceSnapshotId, fingerprint\)/);
  assert.match(appSource, /storeRuntimeContractSubmission\(sourceSnapshotId, submission\)/);
  assert.match(appSource, /clearStoredRuntimeContractSubmission\(sourceSnapshotId\)/);
});

test("review uses a commercial modal surface instead of a dense text panel", () => {
  assert.match(stylesSource, /\.runtime-contract-dialog::backdrop[\s\S]*?backdrop-filter: blur/);
  assert.match(stylesSource, /\.runtime-contract-shell[\s\S]*?border-radius: 24px/);
  assert.match(stylesSource, /\.runtime-contract-fields[\s\S]*?grid-template-columns: repeat\(2/);
  assert.match(stylesSource, /\.runtime-contract-control input,[\s\S]*?background: transparent;[\s\S]*?border: 0/);
});

test("legacy survey and time-depth paths stay compatible without separate registration rows", () => {
  assert.match(platformSource, /key: "timeDepth"[\s\S]*?时深 \/ Checkshot \/ VSP/);
  assert.match(platformSource, /key: "auxiliary"[\s\S]*?测区坐标等/);
  assert.match(appSource, /HIDDEN_REGISTRATION_PATH_GROUPS[\s\S]*?"survey"[\s\S]*?"timeDepth"/);
  assert.match(appSource, /SURVEY_AUXILIARY_PATH_HINTS[\s\S]*?"测区"[\s\S]*?"坐标"[\s\S]*?"网格"/);
  assert.match(appSource, /survey_paths:\s*surveyPaths/);
  assert.match(appSource, /auxiliaryPaths\.filter\(isLikelySurveyAuxiliaryPath\)/);
  assert.match(appSource, /auxiliary_paths:\s*auxiliaryPaths/);
  assert.match(appSource, /time_depth_paths: values\("timeDepth"\)/);
  assert.match(appSource, /survey: Array\.isArray\(source\.survey_paths\)/);
  assert.match(appSource, /timeDepth: Array\.isArray\(source\.time_depth_paths\)/);
  assert.match(appSource, /survey: demo\.value\.survey_paths \|\| \[\]/);
  assert.match(appSource, /timeDepth: demo\.value\.time_depth_paths \|\| \[\]/);
  assert.match(appSource, /'auxiliary-path-group': group\.key === 'auxiliary'/);
  assert.match(productThemeSource, /\.path-group\.auxiliary-path-group[\s\S]*?padding-top:\s*11px/);
  assert.match(productThemeSource, /\.auxiliary-path-group \.empty-path\s*\{\s*display:\s*none/);
  assert.doesNotMatch(appSource, /const timeDepthPaths/);
  assert.doesNotMatch(appSource, /time_depth_paths:\s*\[\]/);
});

test("completed registration and PreparedView override stale parent-stage labels", () => {
  assert.match(appSource, /function stageCompletedByDownstream\(stageId: string\)/);
  assert.match(appSource, /vertical_alignment[\s\S]*?formalRegistrationReady\.value \|\| preparedViewReady\.value/);
  assert.match(appSource, /sample_building[\s\S]*?preparedViewReady\.value/);
  assert.match(appSource, /stageDisplayReady\(stage\) \? "✓" : index \+ 1/);
  assert.match(appSource, /PreparedView 已就绪/);
  assert.match(appSource, /标定已完成/);
});
