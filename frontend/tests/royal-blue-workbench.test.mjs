import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
const css=await readFile(new URL('../src/product-theme.css',import.meta.url),'utf8');
const assistant=await readFile(new URL('../src/components/assistant/FloatingAssistant.vue',import.meta.url),'utf8');
test('the three-pane workbench shares blue/slate tokens and hides diagnostic chrome',()=>{
  for(const token of ['--primary: #1668dc','--canvas: #f8fafc','--ink: #0f172a','--muted: #64748b','--line: #e2e8f0'])assert.ok(css.includes(token),token);
  assert.match(css,/\.brand span\s*\{\s*display: none !important/);
  assert.match(css,/\.app-shell:not\(\.landing-shell\) \.topbar\s*\{\s*display: none !important/);
  for(const teal of ['#167778','#eaf6f4','#eef8f6','#176e78','#327277','#208677','#5ca5ad','#478b8d'])assert.ok(!assistant.includes(teal),teal);
  assert.ok(assistant.includes('background:#1668dc'));
});
