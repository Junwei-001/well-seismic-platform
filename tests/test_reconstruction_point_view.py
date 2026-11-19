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
        self.has_view_group = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'div' and attrs.get('id') == 'point-view' and attrs.get('role') == 'group':
            self.has_view_group = True
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
    assert controls.has_view_group
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
    stroke:noop, fillText:noop, strokeText:noop, fill:noop, setTransform:noop, measureText:(s)=>({width:String(s).length*7}),
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


def test_projected_frame_fits_rotations_planar_views_and_narrow_canvas(tmp_path):
    import subprocess

    path = tmp_path / 'anisotropic.npz'
    xyz = np.array([[-3., -.2, -20.], [5., .5, 40.], [1., .1, 8.]])
    np.savez(path, query_coordinate=xyz, prediction=np.array([.2, .3, .25]))
    script = render_query_points(path).split('<script>', 1)[1].split('</script>', 1)[0]
    runner = r'''
const vm = require('node:vm'), script=JSON.parse(process.argv[1]);
let width=900,height=350, frame=[],labels=[],circles=[],last=null,ready=0;
const noop=()=>{};
const textWidth=s=>String(s).length*7;
const ctx={clearRect:()=>{frame=[];labels=[];circles=[];},beginPath:noop,
 moveTo:(x,y)=>{last=[x,y];},lineTo:(x,y)=>{frame.push([last,[x,y]]);},stroke:noop,
 fillText:(s,x,y)=>labels.push([String(s),x,y,textWidth(s)]),strokeText:noop,fill:noop,
 setTransform:noop,measureText:s=>({width:textWidth(s)}),arc:(x,y,r)=>circles.push([x,y,r])};
const handlers={}, canvas={getContext:()=>ctx, closest:()=>root,
 addEventListener:(name,fn)=>handlers[name]=fn,setPointerCapture:noop,
 getBoundingClientRect:()=>({width,height,left:0,top:0})};
const scope={dataset:{},addEventListener:noop};
const views=['3d','xy','xz','yz'].map(view=>({dataset:{pointView:view},
 setAttribute:noop,addEventListener:(event,fn)=>handlers['view:'+view]=fn}));
const root={querySelectorAll:()=>views,querySelector:()=>scope};
const elements={'point-canvas':canvas,'point-reset':{addEventListener:(event,fn)=>handlers.reset=fn}};
const doc={getElementById:id=>elements[id]||(elements[id]={addEventListener:noop}),
 dispatchEvent:event=>{if(event.type==='huiyan:point-viewer-ready')ready++;}};
const window={devicePixelRatio:1,addEventListener:(n,f)=>{if(n==='resize')handlers.resize=f;},
 matchMedia:()=>({addEventListener:noop})};
vm.runInNewContext(script,{document:doc,window,CustomEvent:class {constructor(type){this.type=type;}},
 getComputedStyle:()=>({getPropertyValue:()=> '#123456'})});
function check(name){
 if(frame.length!==12 || labels.length<8 || circles.length!==3 || ready!==1) throw Error(name+': missing draw');
 for(const [a,b] of frame)for(const [x,y] of [a,b])
   if(!Number.isFinite(x)||!Number.isFinite(y)||x<0||x>width||y<0||y>height)throw Error(name+': frame clipped');
 for(const [s,x,y,t] of labels)
   if(!Number.isFinite(x)||!Number.isFinite(y)||x<0||x+t>width||y<12||y>height)throw Error(name+': label clipped '+s);
 for(const [x,y,r] of circles)
   if(x-r<0||x+r>width||y-r<0||y+r>height)throw Error(name+': point clipped');
 if(!['X','Y','Z'].every(axis=>labels.some(row=>row[0]===axis)))throw Error(name+': missing axis');
}
check('initial');
for (const size of [[240,320],[540,170],[900,350]]) {
 [width,height]=size;handlers.resize();check('resize');
 for (const view of ['xy','xz','yz','3d']) {handlers['view:'+view]();check(view);
  if(view==='xy') {
   const horizontal=frame.map(([a,b])=>Math.abs(b[0]-a[0])).filter(x=>x>1e-8);
   const vertical=frame.map(([a,b])=>Math.abs(b[1]-a[1])).filter(x=>x>1e-8);
   const observed=horizontal[0]/vertical[0];
   if(Math.abs(observed-(8/.7))>1e-7)throw Error('physical aspect ratio changed');
  }
 }
 handlers.pointerdown({clientX:0,clientY:0,pointerId:1});
 for(const p of [[130,80],[-210,230],[450,-270],[-450,-500],[0,0]]) {
  handlers.pointermove({clientX:p[0],clientY:p[1]});check('rotation');
 }
 handlers.pointerup();
}
'''
    result = subprocess.run(['node', '-e', runner, json.dumps(script)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_nine_ordered_query_strips_render_continuous_surfaces_and_sections(tmp_path):
    import subprocess

    xy = np.array([(x, y) for y in np.linspace(0, 1, 8) for x in np.linspace(0, 1, 8)])
    xyz = np.concatenate([np.column_stack((xy, k * .15 + .04 * np.sin(xy[:, 0] * 4))) for k in range(9)])
    values = .22 + xyz[:, 0] * .01 + xyz[:, 2] * .005
    path = tmp_path / 'nine-layers.npz'
    np.savez(path, query_coordinate=xyz, prediction=values)
    html = render_query_points(path)
    strata = json.loads(re.search(r'const strata=(.*?);', html).group(1))
    assert len(strata['x']) == len(strata['y']) == 20
    assert np.asarray(strata['z']).shape == (9, 20, 20)
    z = np.asarray(strata['z'])
    assert np.isfinite(z).all()
    assert np.all(np.diff(z, axis=0) > 0)
    assert np.min(z[-1] - z[0]) > .4 * np.ptp(xyz[:, 2])
    assert z.min() >= xyz[:, 2].min() - 1e-5
    assert z.max() <= xyz[:, 2].max() + 1e-5
    assert strata['geometry_kind'] == 'illustrative_corner_grid'
    assert values.min() - 1e-5 <= np.min(strata['v']) <= np.max(strata['v']) <= values.max() + 1e-5
    assert '查询点插值展示' in html and '并非新增地层识别结果' in html
    script = html.split('<script>', 1)[1].split('</script>', 1)[0]
    runner = r'''
const vm=require('node:vm'),script=require('node:fs').readFileSync(0,'utf8');
let fills=0,arcs=0,ready=0,coordinates=[];const handlers={},noop=()=>{};
const ctx={clearRect:()=>{fills=0;arcs=0;coordinates=[];},beginPath:noop,closePath:noop,
 moveTo:(x,y)=>{if(!Number.isFinite(x+y))throw Error('nonfinite');coordinates.push([x,y]);},lineTo:noop,
 fill:()=>fills++,arc:()=>arcs++,stroke:noop,setTransform:noop,
 measureText:s=>({width:String(s).length*7}),fillText:noop,strokeText:noop};
const scope={dataset:{},addEventListener:noop};
const buttons=['3d','xy','xz','yz'].map(view=>({dataset:{pointView:view},setAttribute:noop,
 addEventListener:(e,f)=>handlers[view]=f}));
const orbit=['up','down','left','right'].map(v=>({dataset:{orbit:v},addEventListener:(e,f)=>handlers['orbit:'+v]=f}));
const zoom=['in','out'].map(v=>({dataset:{zoom:v},addEventListener:(e,f)=>handlers['zoom:'+v]=f}));
const root={querySelectorAll:s=>s==='[data-orbit]'?orbit:s==='[data-zoom]'?zoom:buttons,querySelector:()=>scope};
const canvas={getContext:()=>ctx,closest:()=>root,addEventListener:(n,f)=>handlers[n]=f,setPointerCapture:noop,
 getBoundingClientRect:()=>({width:800,height:500})};
const elements={'point-canvas':canvas};
const document={getElementById:id=>elements[id]||(elements[id]={addEventListener:noop}),
 dispatchEvent:e=>{if(e.type==='huiyan:point-viewer-ready')ready++;}};
const window={addEventListener:noop,matchMedia:()=>({addEventListener:noop})};
vm.runInNewContext(script,{document,window,CustomEvent:class{constructor(type){this.type=type;}},
 getComputedStyle:()=>({getPropertyValue:()=> '#123456'})});
if(fills<1000||fills>1804||arcs!==3||ready!==1)throw Error('missing cutaway block '+fills);
const blockFills=fills;
for(const [view,expected] of [['xy',362],['xz',153],['yz',153],['3d',blockFills]]){
 handlers[view]();if(fills!==expected||arcs!==(view==='3d'?3:1)||ready!==1)throw Error('wrongsection '+view+':'+fills);
}
const baseline=JSON.stringify(coordinates);
handlers.wheel({deltaY:-200,preventDefault:noop});
if(JSON.stringify(coordinates)===baseline)throw Error('wheel did not zoom');
handlers.dblclick();if(JSON.stringify(coordinates)!==baseline)throw Error('reset did not restore');
handlers.pointerdown({clientX:0,clientY:0,pointerId:1});handlers.pointermove({clientX:70,clientY:30});handlers.pointerup();
if(JSON.stringify(coordinates)===baseline)throw Error('drag did not rotate');
handlers.dblclick();if(JSON.stringify(coordinates)!==baseline)throw Error('doubleclick did not reset');
for(const control of ['orbit:up','orbit:down','orbit:left','orbit:right','zoom:in','zoom:out']){
 handlers[control]();if(JSON.stringify(coordinates)===baseline)throw Error('button did not change camera '+control);handlers.dblclick();
}
const tickIds=['point-max','point-q3','point-mid','point-q1','point-min'];
const ticks=tickIds.map(id=>{if(!/^\d+\.\d{4}$/.test(elements[id].textContent))throw Error('tickformat');return Number(elements[id].textContent);});
if(!ticks.every((v,i)=>i===0||v<=ticks[i-1]))throw Error('legend order');
'''
    result = subprocess.run(['node', '-e', runner], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_large_unlayered_query_cloud_keeps_point_fallback(tmp_path):
    path = tmp_path / 'unlayered.npz'
    xyz = np.column_stack((np.linspace(0, 1, 80), np.linspace(0, 1, 80), np.zeros(80)))
    np.savez(path, query_coordinate=xyz, prediction=np.linspace(.2, .3, 80))
    assert 'const strata=null;' in render_query_points(path)


def test_offline_grid_export_is_self_contained_and_discloses_schematic_geometry(tmp_path):
    from well_seismic.reconstruction_point_view import export_query_grid_html

    path = tmp_path / 'offline.npz'
    np.savez(path, query_coordinate=np.array([[0., 0., 0.], [1., 1., 1.]]), prediction=np.array([.2, .3]))
    html = export_query_grid_html(path, archived=True)
    assert html.startswith('<!doctype html>') and html.endswith('</body></html>')
    assert '<script>' in html and 'const data=' in html
    assert not re.search(r'<(?:script|link)[^>]*(?:src|href)=', html)
    assert '不是新增预测或实测井' in html
    assert str(path) not in html
    assert 'DVC 已归档' in html
    assert '.p21-eyebrow,.p21-subtitle,.p21-kpis,.p21-scope,.p21-caption,.p21-detail{display:none!important}' in html
