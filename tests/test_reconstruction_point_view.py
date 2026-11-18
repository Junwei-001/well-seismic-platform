import json
import re
from html.parser import HTMLParser

import numpy as np
import pytest

from well_seismic.reconstruction_point_view import render_query_points


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.views = {}
        self.has_scope = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'button' and 'data-point-view' in attrs:
            self.views[attrs['data-point-view']] = attrs.get('aria-pressed')
        if tag == 'summary' and attrs.get('aria-label') == '查看数据范围说明':
            self.has_scope = True


def test_viewer_preserves_data_and_discloses_real_scope(tmp_path):
    xyz = np.array([[0.0, 1.0, 2.0], [0.5, 0.2, 0.1]])
    values = np.array([0.221, 0.243])
    path = tmp_path / 'points.npz'
    np.savez(path, query_coordinate=xyz, prediction=values)
    html = render_query_points(path)
    match = re.search(r'const data=(\[.*?\]), canvas=', html)
    assert match and json.loads(match.group(1)) == np.column_stack((xyz, values)).tolist()
    controls = Controls()
    controls.feed(html)
    assert controls.views == {'3d': 'true', 'xy': 'false', 'xz': 'false', 'yz': 'false'}
    assert controls.has_scope
    assert 'data-point-viewer' in html
    assert html.count("document.dispatchEvent(new CustomEvent('huiyan:point-viewer-ready'))") == 1
    assert '训练支撑尺度归一化' in html and '不代表完整三维体或新工区验证' in html
    assert 'DVC 已归档' not in html  # No archive receipt is passed to this renderer.
    assert 'DVC 已归档' in render_query_points(path, archived=True)


@pytest.mark.parametrize('coordinates,values', [
    (np.ones((2, 2)), np.ones(2)),
    (np.ones((2, 3)), np.ones(3)),
    (np.array([[0., 0., np.nan]]), np.ones(1)),
    (np.ones((1, 3)), np.array([np.inf])),
    (np.ones((2049, 3)), np.ones(2049)),
])
def test_invalid_coordinate_or_prediction_arrays_rejected(tmp_path, coordinates, values):
    path = tmp_path / 'invalid.npz'
    np.savez(path, query_coordinate=coordinates, prediction=values)
    with pytest.raises(ValueError):
        render_query_points(path)


def test_unexpected_archive_member_is_rejected(tmp_path):
    path = tmp_path / 'extra.npz'
    np.savez(path, query_coordinate=np.zeros((1, 3)), prediction=np.zeros(1), extra=np.zeros(1))
    with pytest.raises(ValueError):
        render_query_points(path)


def test_ready_event_only_after_successful_draw_and_only_once(tmp_path):
    import subprocess

    path = tmp_path / 'points.npz'
    np.savez(path, query_coordinate=np.array([[0., 0., 0.], [1., 1., 1.]]),
             prediction=np.array([0.2, 0.3]))
    html = render_query_points(path)
    script = html.split('<script>', 1)[1].split('</script>', 1)[0]
    # Execute the actual inline script against a tiny canvas/DOM stub. A failed
    # drawing primitive must not tell the embedding application it is ready.
    runner = r'''
const vm = require('node:vm');
const script = JSON.parse(process.argv[1]);
function execute(fail) {
  let events = 0, onresize;
  const noop = () => {};
  const ctx = {clearRect:noop, beginPath:noop, moveTo:noop, lineTo:noop,
    stroke:noop, fillText:noop, strokeText:noop, fill:noop, setTransform:noop,
    arc:() => {if (fail) throw Error('draw failed');}};
  const scope = {dataset:{}, addEventListener:noop};
  const root = {querySelectorAll:()=>[], querySelector:()=>scope};
  const canvas = {getContext:()=>ctx, closest:()=>root, addEventListener:noop,
    getBoundingClientRect:()=>({width:900,height:520})};
  const elements = {'point-canvas':canvas};
  const doc = {getElementById:(id)=>elements[id] || (elements[id]={addEventListener:noop}),
    dispatchEvent:(event)=>{if(event.type==='huiyan:point-viewer-ready')events++;}};
  const window = {devicePixelRatio:1, addEventListener:(n,f)=>{if(n==='resize')onresize=f;},
    matchMedia:()=>({addEventListener:noop})};
  const context = {document:doc,window,CustomEvent:class {constructor(type){this.type=type;}},
    getComputedStyle:()=>({getPropertyValue:()=> '#123456'})};
  let threw = false;
  try {vm.runInNewContext(script,context); if(onresize)onresize();} catch(e){threw=true;}
  return {events,threw};
}
const ok=execute(false), failed=execute(true);
if(ok.events!==1 || ok.threw || failed.events!==0 || !failed.threw) process.exit(1);
'''
    result = subprocess.run(['node', '-e', runner, json.dumps(script)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
