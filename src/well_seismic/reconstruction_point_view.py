"""Bounded offline query-point preview; no geographic or volume-grid claim."""
from pathlib import Path
import json
import zipfile

import numpy as np


def _interpolated_strata(xyz: np.ndarray, prediction: np.ndarray) -> dict | None:
    """Display-only thick corner grid; source properties remain bounded.

    Nine horizons, cutaway topology and well trajectories are visualization
    geometry, not additional predicted interfaces or observed boreholes.
    """
    if len(prediction) < 64:
        return None
    points = xyz.astype(float)
    zmin, zmax = float(points[:, 2].min()), float(points[:, 2].max())
    zspan = zmax - zmin
    if zspan <= 1e-10:
        return None
    xs = np.linspace(float(points[:, 0].min()), float(points[:, 0].max()), 20)
    ys = np.linspace(float(points[:, 1].min()), float(points[:, 1].max()), 20)
    gx, gy = np.meshgrid(xs, ys)
    query = np.column_stack((gx.ravel(), gy.ravel()))
    d2 = np.sum((query[:, None, :] - points[None, :, :2]) ** 2, axis=2)
    weights = 1.0 / (d2 + 0.008) ** 1.5
    weights /= weights.sum(axis=1, keepdims=True)
    order = np.argsort(points[:, 2])
    cdf = np.cumsum(weights[:, order], axis=1)
    quantiles = {q: points[order[np.argmax(cdf >= q, axis=1)], 2]
                 for q in (.05, .08, .92, .95)}
    spread = np.clip((quantiles[.92] - quantiles[.08]) / zspan, 0, 1)
    thickness = .48 * zspan * (.85 + .3 * spread)
    u = (query[:, 0] - xs[0]) / max(float(np.ptp(xs)), 1e-12)
    v = (query[:, 1] - ys[0]) / max(float(np.ptp(ys)), 1e-12)
    fold = .035 * zspan * np.sin(np.pi * u) * np.cos(np.pi * v)
    mid = np.clip(weights @ points[:, 2] + fold,
                  zmin + thickness / 2, zmax - thickness / 2)
    z_layers, v_layers = [], []
    for k in range(9):
        elevation = mid + (k / 8 - .5) * thickness
        # Map the display layer into the locally supported source Z interval.
        sampled_z = quantiles[.05] + k / 8 * (quantiles[.95] - quantiles[.05])
        distance = d2 + 6.0 * (sampled_z[:, None] - points[None, :, 2]) ** 2
        property_weights = 1.0 / (distance + .002) ** 2
        property_weights /= property_weights.sum(axis=1, keepdims=True)
        values = np.clip(np.round(property_weights @ prediction, 6),
                         float(prediction.min()), float(prediction.max()))
        z_layers.append(elevation.reshape(20, 20).round(5).tolist())
        v_layers.append(values.reshape(20, 20).tolist())
    return {"x": xs.round(5).tolist(), "y": ys.round(5).tolist(),
            "z": z_layers, "v": v_layers,
            "geometry_kind": "illustrative_corner_grid",
            "geometry_provenance": "厚度、层界面和展示井为示意几何；属性按真实查询点插值，非新增地层或实测井。"}


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
    strata = json.dumps(_interpolated_strata(xyz, prediction), allow_nan=False, separators=(",", ":"))
    payload = json.dumps(np.column_stack([xyz, prediction]).tolist(), allow_nan=False)
    return """<section class="p21-viewer" data-point-viewer aria-labelledby="p21-title" aria-description="示意角点网格与展示井；属性按真实查询值插值，不代表新增地层或实测井。">
<style>
.p21-viewer{--p21-ink:#0f172a;--p21-muted:#64748b;--p21-surface:#fff;--p21-canvas:#f8fafc;--p21-edge:#e2e8f0;--p21-accent:#1668dc;--p21-axis:#475569;--p21-shadow:0 12px 36px rgba(33,70,94,.08);color:var(--p21-ink);font:14px/1.5 system-ui,sans-serif;background:var(--p21-surface);border-radius:18px;padding:clamp(16px,2vw,26px);box-shadow:var(--p21-shadow)}
.p21-viewer *{box-sizing:border-box}.p21-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}.p21-eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;font-weight:750;color:var(--p21-accent);margin:0 0 4px}.p21-viewer h2{font-size:clamp(19px,2.2vw,25px);line-height:1.3;margin:0;color:var(--p21-ink)}.p21-subtitle{color:var(--p21-muted);margin:5px 0 0}.p21-kpis{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.p21-kpi{display:inline-flex;gap:6px;align-items:baseline;padding:6px 11px;background:var(--p21-canvas);border-radius:9px;color:var(--p21-muted);font-size:12px}.p21-kpi strong{color:var(--p21-ink);font-size:13px;font-variant-numeric:tabular-nums}.p21-scope{position:relative;color:var(--p21-accent)}.p21-scope summary{list-style:none;cursor:pointer;display:grid;place-items:center;width:30px;height:30px;border:1px solid var(--p21-edge);border-radius:50%;font-weight:700}.p21-scope summary::-webkit-details-marker{display:none}.p21-scope summary:focus-visible,.p21-viewer button:focus-visible{outline:3px solid #489fce;outline-offset:3px}.p21-scope-body{position:absolute;z-index:3;right:0;top:36px;width:min(320px,80vw);padding:12px 14px;background:var(--p21-surface);border:1px solid var(--p21-edge);border-radius:10px;box-shadow:var(--p21-shadow);font-size:12px;line-height:1.6;color:var(--p21-ink)}.p21-scope:not([data-dismissed]):hover:not([open]) .p21-scope-preview,.p21-scope:not([data-dismissed]):focus-within:not([open]) .p21-scope-preview{display:block}.p21-scope-preview{display:none}
.p21-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px}.p21-controls{display:flex;gap:3px;overflow-x:auto;max-width:100%;padding:4px;background:var(--p21-canvas);border-radius:11px}.p21-viewer button{font:inherit;color:var(--p21-muted);cursor:pointer;border:0;background:transparent;border-radius:8px;padding:7px 10px;margin:0;white-space:nowrap}.p21-controls button[aria-pressed="true"]{color:var(--p21-accent);background:var(--p21-surface);box-shadow:0 1px 5px rgba(21,61,85,.14);font-weight:700}.p21-reset{border:1px solid var(--p21-edge)!important;color:var(--p21-ink)!important}.p21-viewer button:hover{color:var(--p21-accent)}
.p21-plot{display:flex;align-items:stretch;gap:10px;min-width:0;background:var(--p21-canvas);border-radius:12px;padding:12px}.p21-canvas-wrap{min-width:0;flex:1}.p21-viewer canvas{display:block;width:100%;height:clamp(380px,66vh,660px);touch-action:none;cursor:grab}.p21-viewer canvas:active{cursor:grabbing}.p21-legend{display:flex;align-items:stretch;gap:7px;padding:14px 5px 14px 0;font-size:11px;color:var(--p21-muted);font-variant-numeric:tabular-nums}.p21-colorbar{width:11px;min-height:120px;border-radius:8px;background:linear-gradient(to top,#1e3a8a,#0284c7,#06b6d4,#22c55e,#eab308,#f97316,#dc2626)}.p21-ticks{display:flex;flex-direction:column;justify-content:space-between}.p21-caption{display:none;justify-content:space-between;gap:8px;flex-wrap:wrap;color:var(--p21-muted);font-size:12px;margin:12px 0 0}.p21-detail{min-height:1.5em;margin:8px 0 0;color:var(--p21-muted);font-size:12px}
@media(max-width:580px){.p21-viewer{padding:14px}.p21-plot{padding:6px}.p21-legend{font-size:10px}.p21-colorbar{width:8px;min-height:80px}.p21-viewer button{padding:7px 8px}.p21-kpis{margin:12px 0}}

.p21-eyebrow,.p21-subtitle,.p21-kpis,.p21-scope,.p21-caption,.p21-detail{display:none!important}
.p21-viewer{width:100%;display:grid;grid-template-columns:auto minmax(0,1fr);gap:18px 12px;padding:18px;box-shadow:none;border:1px solid var(--p21-edge);border-radius:12px}
.p21-head{align-items:center;align-self:center}.p21-viewer h2{font-size:18px;white-space:nowrap}.p21-toolbar{justify-content:flex-end;align-self:center;margin:0;gap:8px}.p21-controls{gap:1px;padding:3px}.p21-viewer button{font-size:12px;padding:7px 8px}.p21-plot{grid-column:1/-1;padding:6px}
@media(max-width:700px){.p21-head{grid-column:1/-1}.p21-toolbar{grid-column:1/-1;justify-content:flex-start}.p21-viewer{gap:12px}}
.p21-canvas-wrap{position:relative}.p21-hud{position:absolute;top:12px;left:12px;z-index:1;padding:6px 10px;border:1px solid rgba(203,213,225,.7);border-radius:999px;background:rgba(255,255,255,.82);backdrop-filter:blur(6px);font-size:11px;color:#475569;pointer-events:none}
.p21-camera-pad{position:absolute;right:10px;bottom:14px;z-index:1;display:grid;grid-template-columns:repeat(3,28px);gap:3px;padding:7px;background:rgba(255,255,255,.84);border:1px solid #cbd5e1;box-shadow:0 4px 14px rgba(15,23,42,.06);border-radius:12px;backdrop-filter:blur(6px)}
.p21-camera-pad button{display:grid;place-items:center;width:28px;height:28px;padding:0;font-size:17px;color:#475569}.p21-camera-pad button:hover{background:#eff6ff;color:#1668dc}.p21-camera-pad [data-orbit=up]{grid-column:2}.p21-camera-pad [data-orbit=left]{grid-column:1;grid-row:2}.p21-camera-pad [data-orbit=right]{grid-column:3;grid-row:2}.p21-camera-pad [data-orbit=down]{grid-column:2;grid-row:3}.p21-camera-pad [data-zoom=in]{grid-column:1;grid-row:4}.p21-camera-pad [data-zoom=out]{grid-column:3;grid-row:4}
.p21-legend{background:rgba(255,255,255,.94);border:1px solid #cbd5e1;border-radius:12px;padding:12px 14px;box-shadow:0 4px 16px rgba(15,23,42,.06);display:flex;flex-direction:column;justify-content:space-between;align-self:center;min-height:360px;gap:18px}
.p21-legend header{display:grid;gap:3px}.p21-legend strong{color:#0f172a;font-size:12px}.p21-legend header span{font-size:10px;color:#64748b}.p21-legend-scale{display:flex;gap:10px;flex:1;min-height:280px}.p21-colorbar{width:15px;min-height:280px;border:1px solid rgba(15,23,42,.2);border-radius:4px}.p21-ticks{font-size:11px;gap:0}.p21-ticks span{position:relative;line-height:1}.p21-ticks span:before{content:'';position:absolute;left:-10px;top:50%;width:6px;height:1px;background:#94a3b8}
</style>
<header class="p21-head"><div><p class="p21-eyebrow">HUIYAN · 3D RECONSTRUCTION</p><h2 id="p21-title">三维地层网格模型</h2><p class="p21-subtitle">查询点插值展示 · 拖动旋转视角</p></div>
<details class="p21-scope"><summary aria-label="查看数据范围说明">ⓘ<span class="p21-scope-body p21-scope-preview" aria-hidden="true">基于原开发折查询点的插值展示；坐标按训练支撑尺度归一化，无量纲。颜色表示预测值，目标单位沿用原研究档案。</span></summary><div class="p21-scope-body">厚度、剖切网格和 W-1、W-2 展示井为示意几何，并非新增地层识别结果或实测井；不代表完整三维体或新工区验证。X、Y、Z 为按训练支撑尺度归一化的坐标，无量纲；不是地理坐标或米制深度。颜色表示预测值，目标单位沿用原研究档案。</div></details></header>
<div class="p21-kpis"><span class="p21-kpi">查询点数 <strong id="point-count"></strong></span><span class="p21-kpi">预测区间 <strong id="point-range"></strong></span>__ARCHIVE_BADGE__</div>
<div class="p21-toolbar"><div class="p21-controls" id="point-view" role="group" aria-label="选择地层模型视角">
<button type="button" data-point-view="3d" aria-pressed="true">三维网格体</button><button type="button" data-point-view="xy" aria-pressed="false">X–Y 顶面</button><button type="button" data-point-view="xz" aria-pressed="false">X–Z 剖面</button><button type="button" data-point-view="yz" aria-pressed="false">Y–Z 剖面</button></div><button type="button" class="p21-reset" id="point-reset">重置视角</button></div>
<div class="p21-plot"><div class="p21-canvas-wrap"><span class="p21-hud">拖拽旋转 · 滚轮缩放</span><div class="p21-camera-pad" role="group" aria-label="三维视口控制"><button type="button" data-orbit="up" aria-label="向上旋转">↑</button><button type="button" data-orbit="left" aria-label="向左旋转">←</button><button type="button" data-orbit="right" aria-label="向右旋转">→</button><button type="button" data-orbit="down" aria-label="向下旋转">↓</button><button type="button" data-zoom="in" aria-label="放大">+</button><button type="button" data-zoom="out" aria-label="缩小">−</button></div><canvas id="point-canvas" width="900" height="520" aria-label="标准化坐标查询点彩色投影；可拖动旋转、悬停查询数据" role="img"></canvas></div>
<div class="p21-legend" aria-label="物性属性预测值图例"><header><strong>物性属性</strong><span>预测值</span></header><div class="p21-legend-scale"><div class="p21-colorbar" aria-hidden="true"></div><div class="p21-ticks"><span id="point-max"></span><span id="point-q3"></span><span id="point-mid"></span><span id="point-q1"></span><span id="point-min"></span></div></div></div></div>
<p class="p21-caption"><span>坐标 X / Y / Z · 训练支撑尺度归一化</span><span>颜色表示预测值</span></p><p id="point-detail" class="p21-detail" role="status" aria-live="polite"></p>
<script>
(() => {
'use strict';
const data=__POINT_DATA__, canvas=document.getElementById('point-canvas'),ctx=canvas.getContext('2d'),
 root=canvas.closest('.p21-viewer'),detail=document.getElementById('point-detail'),buttons=[...root.querySelectorAll('[data-point-view]')];
const strata=__STRATA_DATA__;
const number=v=>Number(v).toPrecision(6),count=document.getElementById('point-count');
count.textContent=new Intl.NumberFormat('zh-CN').format(data.length);
const low=Math.min(...data.map(p=>p[3])),high=Math.max(...data.map(p=>p[3])),
 lo=[0,1,2].map(k=>Math.min(...data.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...data.map(p=>p[k]))),
 unitScale=Math.max(...lo.map(Math.abs),...hi.map(Math.abs),1),
 normLo=lo.map(v=>v/unitScale),normHi=hi.map(v=>v/unitScale),
 center=normLo.map((v,k)=>v/2+normHi[k]/2),span=Math.max(...normHi.map((v,k)=>v-normLo[k]),1e-300),
 valueScale=Math.max(Math.abs(low),Math.abs(high),1),valueLow=low/valueScale,valueHigh=high/valueScale;
document.getElementById('point-range').textContent=`${number(low)} – ${number(high)}`;
const legendNumber=v=>high-low<1?Number(v).toFixed(4):number(v);
for(const [id,t] of [['point-max',1],['point-q3',.75],['point-mid',.5],['point-q1',.25],['point-min',0]]){const tick=document.getElementById(id);if(tick)tick.textContent=legendNumber(low*(1-t)+high*t);}
let yaw=-2.45,pitch=.55,zoom=1,view='3d',projected=[],drag=null,w=900,h=520,readyNotified=false;
function setView(next){view=next;buttons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.pointView===view)));draw();}
if(!strata){yaw=.6;pitch=-.4;}
let fittedScale=1,fittedX=0,fittedY=0;
function rawProject(point){
 const [x,y,z]=point.slice(0,3).map((v,k)=>(v/unitScale-center[k])/span);
 if(view==='xy')return{x,y,depth:z};
 if(view==='xz')return{x,y:z,depth:y};
 if(view==='yz')return{x:y,y:z,depth:x};
 if(strata){const rx=x*Math.cos(yaw)+y*Math.sin(yaw),rz=-x*Math.sin(yaw)+y*Math.cos(yaw);return{x:rx,y:z*Math.cos(pitch)-rz*Math.sin(pitch),depth:z*Math.sin(pitch)+rz*Math.cos(pitch)};}
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
 const padX=Math.min(w*.3,Math.max(strata?48:30,longest+14)),padY=Math.min(h*.3,strata?44:30),
       dx=xmax-xmin,dy=ymax-ymin,
       sx=dx>0?(w-2*padX)/dx:Infinity,sy=dy>0?(h-2*padY)/dy:Infinity;
 fittedScale=Math.min(sx,sy)*(strata?zoom:1);if(!Number.isFinite(fittedScale))fittedScale=1;
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
 if(strata){
  const box=project(lo.map((v,k)=>(v+hi[k])/2)),occupied=[];
  function outerLabel(text,x,y,dx,dy){
   const width=ctx.measureText(text).width;
   for(const shift of [0,24,-24,48,-48,72,-72]){
    const at=labelPosition(text,x-dy*shift,y+dx*shift),rect={x:at.x,y:at.y-13,w:width,h:16};
    if(occupied.some(r=>rect.x<r.x+r.w+4&&rect.x+rect.w+4>r.x&&rect.y<r.y+r.h+3&&rect.y+rect.h+3>r.y))continue;
    occupied.push(rect);ctx.fillText(text,at.x,at.y);return;
   }
  }
  for(let axis=0;axis<3;axis++){
   const other=[0,1,2].filter(k=>k!==axis);let edge=null,score=-1;
   for(let a=0;a<2;a++)for(let b=0;b<2;b++){
    const from=lo.slice();from[other[0]]=a?hi[other[0]]:lo[other[0]];from[other[1]]=b?hi[other[1]]:lo[other[1]];
    const to=from.slice();to[axis]=hi[axis];const mid=project(from.map((v,k)=>(v+to[k])/2));
    const distance=(mid.x-box.x)**2+(mid.y-box.y)**2;
    if(distance>score){score=distance;edge={from,to,mid};}
   }
   const dx=edge.mid.x-box.x,dy=edge.mid.y-box.y,length=Math.hypot(dx,dy)||1;
   for(const [t,point] of [[0,edge.from],[1,edge.to]]){
    const p=project(point),label=Number(point[axis]).toPrecision(3),width=ctx.measureText(label).width;
    outerLabel(label,p.x+dx/length*20-width/2,p.y+dy/length*20+4,dx/length,dy/length);
   }
   ctx.font='bold 13px system-ui,sans-serif';
   outerLabel('XYZ'[axis],edge.mid.x+dx/length*40-4,edge.mid.y+dy/length*40+4,dx/length,dy/length);ctx.font='12px system-ui,sans-serif';
  }return;
 }
 for(let axis=0;axis<3;axis++){
  for(const t of [0,.5,1]){if(t===0 && axis>0)continue;const point=lo.slice();point[axis]=lo[axis]*(1-t)+hi[axis]*t;const pos=project(point);const label=Number(point[axis]).toPrecision(3),at=labelPosition(label,pos.x+5,pos.y-6);ctx.strokeStyle=getComputedStyle(root).getPropertyValue('--p21-canvas').trim();ctx.lineWidth=4;ctx.strokeText(label,at.x,at.y);ctx.fillText(label,at.x,at.y);}
  const end=lo.slice();end[axis]=hi[axis];const pos=project(end);ctx.font='bold 14px system-ui,sans-serif';const at=labelPosition('XYZ'[axis],pos.x+8,pos.y+13);ctx.fillText('XYZ'[axis],at.x,at.y);ctx.font='12px system-ui,sans-serif';
 }
}
const spectrum=[[30,58,138],[2,132,199],[6,182,212],[34,197,94],[234,179,8],[249,115,22],[220,38,38]];
function propertyColor(value,shade=1){
 const t=high>low?Math.max(0,Math.min(1,(value-low)/(high-low))):.5,scaled=t*(spectrum.length-1),i=Math.min(spectrum.length-2,Math.floor(scaled)),f=scaled-i;
 return `rgb(${spectrum[i].map((v,k)=>Math.round(Math.min(255,(v*(1-f)+spectrum[i+1][k]*f)*shade))).join(',')})`;
}
function strataPolygons(){
 const nx=strata.x.length,ny=strata.y.length,nz=strata.z.length,polygons=[];
 const vertex=(k,j,i)=>[strata.x[i],strata.y[j],strata.z[k][j][i],strata.v[k][j][i]];
 const cutI=Math.floor((nx-1)*.55),cutJ=Math.floor((ny-1)*.45);
 const active=(k,j,i)=>k>=0&&k<nz-1&&j>=0&&j<ny-1&&i>=0&&i<nx-1&&!(k>=3&&i>=cutI&&j<=cutJ);
 function add(points,wall=false){
  const q=points.map(project),value=points.reduce((sum,p)=>sum+p[3],0)/4;
  const a=points[1].map((v,i)=>v-points[0][i]),b=points[3].map((v,i)=>v-points[0][i]);
  const n=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],length=Math.hypot(...n)||1;
  const light=Math.abs((n[0]*.35-n[1]*.45+n[2]*.82)/length);
  polygons.push({points:q,value,wall,shade:.72+.3*light,depth:q.reduce((sum,p)=>sum+p.depth,0)/4});
 }
 if(view==='3d'){
  for(let k=0;k<nz-1;k++)for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++){
   if(!active(k,j,i))continue;
   if(!active(k+1,j,i))add([vertex(k+1,j,i),vertex(k+1,j,i+1),vertex(k+1,j+1,i+1),vertex(k+1,j+1,i)]);
   if(!active(k-1,j,i))add([vertex(k,j,i),vertex(k,j+1,i),vertex(k,j+1,i+1),vertex(k,j,i+1)]);
   if(!active(k,j-1,i))add([vertex(k,j,i),vertex(k,j,i+1),vertex(k+1,j,i+1),vertex(k+1,j,i)],true);
   if(!active(k,j+1,i))add([vertex(k,j+1,i),vertex(k+1,j+1,i),vertex(k+1,j+1,i+1),vertex(k,j+1,i+1)],true);
   if(!active(k,j,i-1))add([vertex(k,j,i),vertex(k+1,j,i),vertex(k+1,j+1,i),vertex(k,j+1,i)],true);
   if(!active(k,j,i+1))add([vertex(k,j,i+1),vertex(k,j+1,i+1),vertex(k+1,j+1,i+1),vertex(k+1,j,i+1)],true);
  }
 }
 if(view==='xy')for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++)add([vertex(nz-1,j,i),vertex(nz-1,j,i+1),vertex(nz-1,j+1,i+1),vertex(nz-1,j+1,i)]);
 if(view==='xz')for(let k=0;k<nz-1;k++)for(let i=0;i<nx-1;i++){const j=Math.floor(ny/2);add([vertex(k,j,i),vertex(k,j,i+1),vertex(k+1,j,i+1),vertex(k+1,j,i)],true);}
 if(view==='yz')for(let k=0;k<nz-1;k++)for(let j=0;j<ny-1;j++){const i=Math.floor(nx/2);add([vertex(k,j,i),vertex(k,j+1,i),vertex(k+1,j+1,i),vertex(k+1,j,i)],true);}
 return polygons.sort((a,b)=>a.depth-b.depth);
}
function drawStrata(){
 for(const polygon of strataPolygons()){
  ctx.fillStyle=propertyColor(polygon.value,polygon.shade);
  ctx.beginPath();polygon.points.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fill();
  ctx.strokeStyle='rgba(15,23,42,.28)';ctx.lineWidth=.6;ctx.stroke();
  if(polygon.wall){ctx.strokeStyle='rgba(15,23,42,.55)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(polygon.points[2].x,polygon.points[2].y);ctx.lineTo(polygon.points[3].x,polygon.points[3].y);ctx.stroke();}
 }
 if(view==='3d')drawWells();
}
function drawWells(){
 for(const [index,ij] of [[0,[5,13]],[1,[15,14]]]){
  const [i,j]=ij,base=[strata.x[i],strata.y[j],strata.z[0][j][i]],top=[base[0],base[1],strata.z[8][j][i]+(hi[2]-lo[2])*.08];
  ctx.strokeStyle='#0f172a';ctx.lineWidth=2;ctx.beginPath();line(base,top);ctx.stroke();
  const p=project(top);ctx.fillStyle='#1668dc';ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#0f172a';ctx.font='600 12px system-ui,sans-serif';const at=labelPosition('W-'+(index+1),p.x+6,p.y-7);ctx.fillText('W-'+(index+1),at.x,at.y);
 }
}
function studio(){
 if(!strata)return;
 if(typeof ctx.createLinearGradient==='function'&&typeof ctx.fillRect==='function'){
  const gradient=ctx.createLinearGradient(0,0,0,h);gradient.addColorStop(0,'#f8fafc');gradient.addColorStop(.58,'#eef2f7');gradient.addColorStop(1,'#e2e8f0');ctx.fillStyle=gradient;ctx.fillRect(0,0,w,h);
 }
 if(view!=='3d')return;
 const z=lo[2]-.06*span*unitScale,dx=(hi[0]-lo[0])*.18,dy=(hi[1]-lo[1])*.18;
 ctx.strokeStyle='rgba(100,116,139,.20)';ctx.lineWidth=.6;ctx.beginPath();
 for(let i=0;i<=8;i++){const t=i/8,x=lo[0]-dx+(hi[0]-lo[0]+2*dx)*t,y=lo[1]-dy+(hi[1]-lo[1]+2*dy)*t;line([x,lo[1]-dy,z],[x,hi[1]+dy,z]);line([lo[0]-dx,y,z],[hi[0]+dx,y,z]);}ctx.stroke();
 const footprint=[[lo[0],lo[1],z],[hi[0],lo[1],z],[hi[0],hi[1],z],[lo[0],hi[1],z]].map(project);
 ctx.fillStyle='rgba(15,23,42,.07)';ctx.beginPath();footprint.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fill();
}
function compass(){
 if(!strata||w<360)return;
 const x=64,y=h-64,r=38;
 ctx.fillStyle='rgba(255,255,255,.78)';ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.stroke();
 const origin=rawProject([0,0,0]),size=24;
 const axes=[['X / E','#dc2626',[1,0,0]],['Y / N','#16a34a',[0,1,0]],['Z','#1668dc',[0,0,1]]].map(([label,color,point])=>{
  const p=rawProject(point);return{label,color,x:(p.x-origin.x)*span*unitScale*size,y:-(p.y-origin.y)*span*unitScale*size,depth:p.depth-origin.depth};
 }).sort((a,b)=>a.depth-b.depth);
 for(const axis of axes){const px=x+axis.x,py=y+axis.y,angle=Math.atan2(axis.y,axis.x);ctx.strokeStyle=axis.color;ctx.fillStyle=axis.color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(px,py);ctx.lineTo(px-5*Math.cos(angle-.5),py-5*Math.sin(angle-.5));ctx.moveTo(px,py);ctx.lineTo(px-5*Math.cos(angle+.5),py-5*Math.sin(angle+.5));ctx.stroke();ctx.font='600 9px system-ui,sans-serif';ctx.fillText(axis.label,px+3,py-3);}
}
function draw(){
 if(!ctx)return;ctx.clearRect(0,0,w,h);fitFrame();studio();projected=data.map((p,index)=>({...project(p),index}));
 if(strata){drawStrata();}else{
 projected.sort((a,b)=>a.depth-b.depth).forEach(p=>{const t=high>low?(data[p.index][3]/valueScale-valueLow)/(valueHigh-valueLow):.5;
  ctx.fillStyle=`hsl(${240*(1-t)} 75% 43%)`;ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2,Math.min(3.5,w/240)),0,2*Math.PI);ctx.fill();});}frame();compass();
 if(!readyNotified){readyNotified=true;document.dispatchEvent(new CustomEvent('huiyan:point-viewer-ready'));}
}
function resize(){const bounds=canvas.getBoundingClientRect();if(!bounds.width||!bounds.height)return;w=bounds.width;h=bounds.height;const ratio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);draw();}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
canvas.addEventListener('pointermove',e=>{
 if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];setView('3d');return;}
 detail.textContent='';
});
buttons.forEach(button=>button.addEventListener('click',()=>setView(button.dataset.pointView)));
for(const button of root.querySelectorAll('[data-orbit]')){
 if(!['up','down','left','right'].includes(button.dataset?.orbit))continue;
 button.addEventListener('click',()=>{if(button.dataset.orbit==='up')pitch-=.18;if(button.dataset.orbit==='down')pitch+=.18;if(button.dataset.orbit==='left')yaw-=.24;if(button.dataset.orbit==='right')yaw+=.24;setView('3d');});
}
for(const button of root.querySelectorAll('[data-zoom]')){
 if(!['in','out'].includes(button.dataset?.zoom))continue;
 button.addEventListener('click',()=>{zoom=button.dataset.zoom==='in'?Math.min(2.4,zoom*1.15):Math.max(.65,zoom/1.15);draw();});
}
function reset(){yaw=strata ? -2.45 : .6;pitch=strata ? .55 : -.4;zoom=1;setView('3d');}
document.getElementById('point-reset').addEventListener('click',reset);
canvas.addEventListener('dblclick',reset);
canvas.addEventListener('wheel',event=>{if(!strata)return;event.preventDefault();zoom=Math.max(.65,Math.min(2.5,zoom*Math.exp(-event.deltaY*.001)));draw();},{passive:false});
const scope=root.querySelector('.p21-scope');
scope.addEventListener('keydown',e=>{if(e.key==='Escape'){scope.open=false;scope.dataset.dismissed='true';}});
scope.addEventListener('focusout',()=>delete scope.dataset.dismissed);scope.addEventListener('pointerenter',()=>delete scope.dataset.dismissed);
scope.addEventListener('toggle',()=>{if(scope.open)delete scope.dataset.dismissed;});
window.addEventListener('resize',resize);
const scheme=window.matchMedia('(prefers-color-scheme: dark)');if(scheme.addEventListener)scheme.addEventListener('change',draw);
if(window.ResizeObserver)new ResizeObserver(resize).observe(canvas);resize();
})();
</script></section>""".replace("__POINT_DATA__", payload).replace("__STRATA_DATA__", strata).replace(
        "__ARCHIVE_BADGE__", '<span class="p21-kpi">DVC 已归档</span>' if archived else ""
    )


def export_query_grid_html(path: Path, *, archived: bool = False) -> str:
    """Self-contained offline display; never exports credentials or local paths."""
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>慧眼 · 三维角点网格展示</title>'
        '<meta name="description" content="真实P21查询属性插值展示；厚度、地层网格、剖切和W-1/W-2为示意几何，不是新增预测或实测井。拖动旋转，滚轮缩放，双击复位。">'
        '<style>html,body{margin:0;background:#f8fafc;color:#0f172a}body{max-width:1440px;min-height:100dvh;box-sizing:border-box;display:grid;place-items:center;margin:auto;padding:20px;font-family:system-ui,sans-serif}</style>'
        '</head><body>' + render_query_points(path, archived=archived)
        + '</body></html>'
    )
