"""Bounded offline query-point preview; no geographic or volume-grid claim."""
from pathlib import Path
import json
import zipfile

import numpy as np


def render_query_points(path: Path, *, archived: bool = False) -> str:
    # Check the decompressed member sizes before NumPy allocates arrays.
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if {item.filename for item in members} != {"query_coordinate.npy", "prediction.npy"} or len(members) != 2:
            raise ValueError("三维重建查询点预览必须仅包含坐标与预测值")
        if any(item.file_size > 256_000 for item in members):
            raise ValueError("三维重建查询点预览的数组超过大小限制")
        for item in members:
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError("三维重建查询点预览需要符合大小限制的NPY v1文件头")
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                expected_rank = 2 if item.filename == "query_coordinate.npy" else 1
                if (len(shape) != expected_rank or not 1 <= shape[0] <= 2048
                        or (expected_rank == 2 and shape[1] != 3)
                        or not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating)
                        or np.prod(shape) * dtype.itemsize > item.file_size):
                    raise ValueError("三维重建查询点预览的文件头形状或数据类型无效")
    with np.load(path, allow_pickle=False) as arrays:
        xyz, prediction = arrays["query_coordinate"], arrays["prediction"]
    if prediction.ndim != 1 or not 1 <= len(prediction) <= 2048 or xyz.shape != (len(prediction), 3):
        raise ValueError("三维重建查询点预览需要1至2048个坐标与预测值对齐的查询点")
    if not all(np.issubdtype(value.dtype, np.number) and not np.iscomplexobj(value) and np.isfinite(value).all()
               for value in (xyz, prediction)):
        raise ValueError("三维重建查询点预览必须使用有限的实数值")
    payload = json.dumps(np.column_stack([xyz, prediction]).tolist(), allow_nan=False)
    return """<section class="p21-viewer" data-point-viewer aria-labelledby="p21-title">
<style>
.p21-viewer{--p21-ink:#19334b;--p21-muted:#566f83;--p21-surface:#fff;--p21-canvas:#f4f9fc;--p21-edge:#d8e6ef;--p21-accent:#14669a;--p21-axis:#627e93;--p21-shadow:0 12px 36px rgba(33,70,94,.08);color:var(--p21-ink);font:14px/1.5 system-ui,sans-serif;background:var(--p21-surface);border-radius:18px;padding:clamp(16px,2vw,26px);box-shadow:var(--p21-shadow)}
.p21-viewer *{box-sizing:border-box}.p21-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}.p21-eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;font-weight:750;color:var(--p21-accent);margin:0 0 4px}.p21-viewer h2{font-size:clamp(19px,2.2vw,25px);line-height:1.3;margin:0;color:var(--p21-ink)}.p21-subtitle{color:var(--p21-muted);margin:5px 0 0}.p21-kpis{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.p21-kpi{display:inline-flex;gap:6px;align-items:baseline;padding:6px 11px;background:var(--p21-canvas);border-radius:9px;color:var(--p21-muted);font-size:12px}.p21-kpi strong{color:var(--p21-ink);font-size:13px;font-variant-numeric:tabular-nums}.p21-scope{position:relative;color:var(--p21-accent)}.p21-scope summary{list-style:none;cursor:pointer;display:grid;place-items:center;width:30px;height:30px;border:1px solid var(--p21-edge);border-radius:50%;font-weight:700}.p21-scope summary::-webkit-details-marker{display:none}.p21-scope summary:focus-visible,.p21-viewer button:focus-visible{outline:3px solid #489fce;outline-offset:3px}.p21-scope-body{position:absolute;z-index:3;right:0;top:36px;width:min(320px,80vw);padding:12px 14px;background:var(--p21-surface);border:1px solid var(--p21-edge);border-radius:10px;box-shadow:var(--p21-shadow);font-size:12px;line-height:1.6;color:var(--p21-ink)}.p21-scope:not([data-dismissed]):hover:not([open]) .p21-scope-preview,.p21-scope:not([data-dismissed]):focus-within:not([open]) .p21-scope-preview{display:block}.p21-scope-preview{display:none}
.p21-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px}.p21-controls{display:flex;gap:3px;overflow-x:auto;max-width:100%;padding:4px;background:var(--p21-canvas);border-radius:11px}.p21-viewer button{font:inherit;color:var(--p21-muted);cursor:pointer;border:0;background:transparent;border-radius:8px;padding:7px 10px;margin:0;white-space:nowrap}.p21-controls button[aria-pressed="true"]{color:var(--p21-accent);background:var(--p21-surface);box-shadow:0 1px 5px rgba(21,61,85,.14);font-weight:700}.p21-reset{border:1px solid var(--p21-edge)!important;color:var(--p21-ink)!important}.p21-viewer button:hover{color:var(--p21-accent)}
.p21-plot{display:flex;align-items:stretch;gap:10px;min-width:0;background:var(--p21-canvas);border-radius:12px;padding:12px}.p21-canvas-wrap{min-width:0;flex:1}.p21-viewer canvas{display:block;width:100%;height:clamp(240px,48vh,500px);touch-action:none;cursor:grab}.p21-viewer canvas:active{cursor:grabbing}.p21-legend{display:flex;align-items:stretch;gap:7px;padding:14px 5px 14px 0;font-size:11px;color:var(--p21-muted);font-variant-numeric:tabular-nums}.p21-colorbar{width:11px;min-height:120px;border-radius:8px;background:linear-gradient(to top,hsl(240 75% 43%),hsl(180 75% 43%),hsl(120 75% 43%),hsl(60 75% 43%),hsl(0 75% 43%))}.p21-ticks{display:flex;flex-direction:column;justify-content:space-between}.p21-caption{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;color:var(--p21-muted);font-size:12px;margin:12px 0 0}.p21-detail{min-height:1.5em;margin:8px 0 0;color:var(--p21-muted);font-size:12px}
@media(max-width:580px){.p21-viewer{padding:14px}.p21-plot{padding:6px}.p21-legend{font-size:10px}.p21-colorbar{width:8px;min-height:80px}.p21-viewer button{padding:7px 8px}.p21-kpis{margin:12px 0}}
@media(prefers-color-scheme:dark){.p21-viewer{--p21-ink:#e5f2f9;--p21-muted:#aec3d1;--p21-surface:#172a38;--p21-canvas:#10222e;--p21-edge:#3d5868;--p21-accent:#80c5ee;--p21-axis:#9ab6c9;--p21-shadow:0 12px 36px rgba(0,0,0,.2)}}
</style>
<header class="p21-head"><div><p class="p21-eyebrow">HUIYAN · 3D RECONSTRUCTION</p><h2 id="p21-title">P21 三维条件属性重建</h2><p class="p21-subtitle">查询点预测结果 · 拖动图形旋转视角</p></div>
<details class="p21-scope"><summary aria-label="查看数据范围说明">ⓘ<span class="p21-scope-body p21-scope-preview" aria-hidden="true">仅显示原开发折查询点；坐标按训练支撑尺度归一化，无量纲。颜色表示预测值，目标单位沿用原研究档案。</span></summary><div class="p21-scope-body">仅显示原开发折查询点，不代表完整三维体或新工区验证。X、Y、Z 为按训练支撑尺度归一化的坐标，无量纲；不是地理坐标或米制深度。颜色表示预测值，目标单位沿用原研究档案。</div></details></header>
<div class="p21-kpis"><span class="p21-kpi">查询点数 <strong id="point-count"></strong></span><span class="p21-kpi">预测区间 <strong id="point-range"></strong></span>__ARCHIVE_BADGE__</div>
<div class="p21-toolbar"><div class="p21-controls" id="point-view" role="group" aria-label="选择点云投影视角">
<button type="button" data-point-view="3d" aria-pressed="true">三维斜视</button><button type="button" data-point-view="xy" aria-pressed="false">X–Y 平面</button><button type="button" data-point-view="xz" aria-pressed="false">X–Z 平面</button><button type="button" data-point-view="yz" aria-pressed="false">Y–Z 平面</button></div><button type="button" class="p21-reset" id="point-reset">重置视角</button></div>
<div class="p21-plot"><div class="p21-canvas-wrap"><canvas id="point-canvas" width="900" height="520" aria-label="标准化坐标查询点彩色投影；可拖动旋转、悬停查询数据" role="img"></canvas></div>
<div class="p21-legend" aria-label="预测值颜色图例，蓝色为低值，红色为高值"><div class="p21-colorbar" aria-hidden="true"></div><div class="p21-ticks"><span id="point-max"></span><span id="point-mid"></span><span id="point-min"></span></div></div></div>
<p class="p21-caption"><span>坐标 X / Y / Z · 训练支撑尺度归一化</span><span>颜色表示预测值</span></p><p id="point-detail" class="p21-detail" role="status" aria-live="polite">悬停查询点可查看数值。</p>
<script>
(() => {
'use strict';
const data=__POINT_DATA__, canvas=document.getElementById('point-canvas'),ctx=canvas.getContext('2d'),
 root=canvas.closest('.p21-viewer'),detail=document.getElementById('point-detail'),buttons=[...root.querySelectorAll('[data-point-view]')];
const number=v=>Number(v).toPrecision(6),count=document.getElementById('point-count');
count.textContent=new Intl.NumberFormat('zh-CN').format(data.length);
const low=Math.min(...data.map(p=>p[3])),high=Math.max(...data.map(p=>p[3])),
 lo=[0,1,2].map(k=>Math.min(...data.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...data.map(p=>p[k]))),
 unitScale=Math.max(...lo.map(Math.abs),...hi.map(Math.abs),1),
 normLo=lo.map(v=>v/unitScale),normHi=hi.map(v=>v/unitScale),
 center=normLo.map((v,k)=>v/2+normHi[k]/2),span=Math.max(...normHi.map((v,k)=>v-normLo[k]),1e-300),
 valueScale=Math.max(Math.abs(low),Math.abs(high),1),valueLow=low/valueScale,valueHigh=high/valueScale;
document.getElementById('point-range').textContent=`${number(low)} – ${number(high)}`;
document.getElementById('point-max').textContent=number(high);document.getElementById('point-mid').textContent=number(low/2+high/2);document.getElementById('point-min').textContent=number(low);
let yaw=.6,pitch=-.4,view='3d',projected=[],drag=null,w=900,h=520,readyNotified=false;
function setView(next){view=next;buttons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.pointView===view)));draw();}
let fittedScale=1,fittedX=0,fittedY=0;
function rawProject(point){
 const [x,y,z]=point.slice(0,3).map((v,k)=>(v/unitScale-center[k])/span);
 if(view==='xy')return{x,y,depth:z};
 if(view==='xz')return{x,y:z,depth:y};
 if(view==='yz')return{x:y,y:z,depth:x};
 const rx=x*Math.cos(yaw)+z*Math.sin(yaw),rz=-x*Math.sin(yaw)+z*Math.cos(yaw);
 return{x:rx,y:y*Math.cos(pitch)-rz*Math.sin(pitch),depth:y*Math.sin(pitch)+rz*Math.cos(pitch)};
}
function fitFrame(){
 const corners=[];
 for(let i=0;i<8;i++)corners.push(rawProject([0,1,2].map(k=>(i&(1<<k))?hi[k]:lo[k])));
 const xmin=Math.min(...corners.map(p=>p.x)),xmax=Math.max(...corners.map(p=>p.x)),
       ymin=Math.min(...corners.map(p=>p.y)),ymax=Math.max(...corners.map(p=>p.y));
 ctx.font='12px system-ui,sans-serif';
 const longest=Math.max(...lo.concat(hi).map(v=>ctx.measureText(Number(v).toPrecision(3)).width));
 const padX=Math.min(w*.3,Math.max(30,longest+14)),padY=Math.min(h*.3,30),
       dx=xmax-xmin,dy=ymax-ymin,
       sx=dx>0?(w-2*padX)/dx:Infinity,sy=dy>0?(h-2*padY)/dy:Infinity;
 fittedScale=Math.min(sx,sy);if(!Number.isFinite(fittedScale))fittedScale=1;
 fittedX=w/2-(xmin+xmax)/2*fittedScale;fittedY=h/2+(ymin+ymax)/2*fittedScale;
}
function project(point){const p=rawProject(point);return{x:fittedX+p.x*fittedScale,y:fittedY-p.y*fittedScale,depth:p.depth};}
function labelPosition(text,x,y){
 const width=ctx.measureText(text).width;
 return{x:Math.max(2,Math.min(Math.max(2,w-width-2),x)),y:Math.max(15,Math.min(h-3,y))};
}
function line(a,b){const x=project(a),y=project(b);ctx.moveTo(x.x,x.y);ctx.lineTo(y.x,y.y);}
function frame(){
 ctx.strokeStyle=getComputedStyle(root).getPropertyValue('--p21-axis').trim();ctx.globalAlpha=.42;ctx.lineWidth=1;ctx.beginPath();
 for(let axis=0;axis<3;axis++)for(let a=0;a<2;a++)for(let b=0;b<2;b++){
  const other=[0,1,2].filter(k=>k!==axis),from=lo.slice(),to=lo.slice();
  from[other[0]]=a?hi[other[0]]:lo[other[0]];from[other[1]]=b?hi[other[1]]:lo[other[1]];to.splice(0,3,...from);to[axis]=hi[axis];line(from,to);
 }ctx.stroke();ctx.globalAlpha=1;ctx.fillStyle=getComputedStyle(root).getPropertyValue('--p21-axis').trim();ctx.font='12px system-ui,sans-serif';
 for(let axis=0;axis<3;axis++){
  for(const t of [0,.5,1]){if(t===0 && axis>0)continue;const point=lo.slice();point[axis]=lo[axis]*(1-t)+hi[axis]*t;const pos=project(point);const label=Number(point[axis]).toPrecision(3),at=labelPosition(label,pos.x+5,pos.y-6);ctx.strokeStyle=getComputedStyle(root).getPropertyValue('--p21-canvas').trim();ctx.lineWidth=4;ctx.strokeText(label,at.x,at.y);ctx.fillText(label,at.x,at.y);}
  const end=lo.slice();end[axis]=hi[axis];const pos=project(end);ctx.font='bold 14px system-ui,sans-serif';const at=labelPosition('XYZ'[axis],pos.x+8,pos.y+13);ctx.fillText('XYZ'[axis],at.x,at.y);ctx.font='12px system-ui,sans-serif';
 }
}
function draw(){
 if(!ctx)return;ctx.clearRect(0,0,w,h);fitFrame();projected=data.map((p,index)=>({...project(p),index}));
 projected.sort((a,b)=>a.depth-b.depth).forEach(p=>{const t=high>low?(data[p.index][3]/valueScale-valueLow)/(valueHigh-valueLow):.5;
  ctx.fillStyle=`hsl(${240*(1-t)} 75% 43%)`;ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2,Math.min(3.5,w/240)),0,2*Math.PI);ctx.fill();});frame();
 if(!readyNotified){readyNotified=true;document.dispatchEvent(new CustomEvent('huiyan:point-viewer-ready'));}
}
function resize(){const bounds=canvas.getBoundingClientRect();if(!bounds.width||!bounds.height)return;w=bounds.width;h=bounds.height;const ratio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);draw();}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
canvas.addEventListener('pointermove',e=>{
 if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];setView('3d');return;}
 const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let nearest=null,distance=100;
 projected.forEach(p=>{const d=(p.x-x)**2+(p.y-y)**2;if(d<distance){nearest=p;distance=d;}});
 if(nearest){const p=data[nearest.index];detail.textContent=`查询点 ${nearest.index+1}：X=${number(p[0])}，Y=${number(p[1])}，Z=${number(p[2])}；预测=${number(p[3])}`;}
 else detail.textContent='悬停查询点可查看数值。';
});
buttons.forEach(button=>button.addEventListener('click',()=>setView(button.dataset.pointView)));
document.getElementById('point-reset').addEventListener('click',()=>{yaw=.6;pitch=-.4;setView('3d');});
const scope=root.querySelector('.p21-scope');
scope.addEventListener('keydown',e=>{if(e.key==='Escape'){scope.open=false;scope.dataset.dismissed='true';}});
scope.addEventListener('focusout',()=>delete scope.dataset.dismissed);scope.addEventListener('pointerenter',()=>delete scope.dataset.dismissed);
scope.addEventListener('toggle',()=>{if(scope.open)delete scope.dataset.dismissed;});
window.addEventListener('resize',resize);
const scheme=window.matchMedia('(prefers-color-scheme: dark)');if(scheme.addEventListener)scheme.addEventListener('change',draw);
if(window.ResizeObserver)new ResizeObserver(resize).observe(canvas);resize();
})();
</script></section>""".replace("__POINT_DATA__", payload).replace(
        "__ARCHIVE_BADGE__", '<span class="p21-kpi">DVC 已归档</span>' if archived else ""
    )
