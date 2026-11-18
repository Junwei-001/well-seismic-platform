"""Bounded offline query-point preview; no geographic or volume-grid claim."""
from pathlib import Path
import json
import zipfile

import numpy as np


def render_query_points(path: Path) -> str:
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
    return """<section aria-label="三维重建查询点预览">
<p>仅显示原开发折查询点，不代表完整三维体或新工区验证。X、Y、Z 为按训练支撑尺度归一化的坐标，无量纲；不是地理坐标或米制深度。颜色表示预测值，目标单位沿用原研究档案。</p>
<p>查询点：<strong id="point-count"></strong> · 拖动旋转，悬停查询点值。</p>
<label>投影视角 <select id="point-view"><option value="3d">三维斜视</option><option value="xy">X–Y 平面</option><option value="xz">X–Z 平面</option><option value="yz">Y–Z 平面</option></select></label>
<button id="point-reset">重置视角</button>
<p id="point-range"></p><canvas id="point-canvas" width="900" height="520" style="width:100%;max-width:900px;background:#f8fbfd;border:1px solid #adc2ce;touch-action:none" aria-label="标准化坐标查询点彩色投影"></canvas>
<p id="point-detail" role="status">悬停查询一个点的标准化坐标和预测值。</p>
<script>
(() => {
const data=__POINT_DATA__, canvas=document.getElementById('point-canvas'),ctx=canvas.getContext('2d'),
 view=document.getElementById('point-view'),detail=document.getElementById('point-detail');
document.getElementById('point-count').textContent=data.length;
const low=Math.min(...data.map(p=>p[3])),high=Math.max(...data.map(p=>p[3])),
 lo=[0,1,2].map(k=>Math.min(...data.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...data.map(p=>p[k]))),
 center=lo.map((v,k)=>(v+hi[k])/2),span=Math.max(...hi.map((v,k)=>v-lo[k]),1e-12);
document.getElementById('point-range').textContent=`预测范围：${low.toPrecision(6)} 至 ${high.toPrecision(6)}；颜色由蓝（低）到红（高）。`;
let yaw=.6,pitch=-.4,projected=[],drag=null;
function draw(){
 ctx.clearRect(0,0,900,520);projected=[];
 data.forEach((p,index)=>{
  let [x,y,z]=p.slice(0,3).map((v,k)=>(v-center[k])/span);
  let px,py,depth=0;
  if(view.value==='xy'){px=x;py=y;depth=z;}
  else if(view.value==='xz'){px=x;py=z;depth=y;}
  else if(view.value==='yz'){px=y;py=z;depth=x;}
  else{const rx=x*Math.cos(yaw)+z*Math.sin(yaw),rz=-x*Math.sin(yaw)+z*Math.cos(yaw);
   px=rx;py=y*Math.cos(pitch)-rz*Math.sin(pitch);depth=y*Math.sin(pitch)+rz*Math.cos(pitch);}
  projected.push({x:450+px*390,y:260-py*390,depth,index});
 });
 projected.sort((a,b)=>a.depth-b.depth).forEach(p=>{
  const t=high>low?(data[p.index][3]-low)/(high-low):.5;
  ctx.fillStyle=`hsl(${240*(1-t)} 75% 43%)`;ctx.beginPath();ctx.arc(p.x,p.y,3,0,2*Math.PI);ctx.fill();
 });
 ctx.fillStyle='#254656';ctx.font='15px system-ui';ctx.fillText(view.value==='3d'?'三维斜视 · 标准化坐标':view.options[view.selectedIndex].text,18,25);
}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
canvas.addEventListener('pointermove',e=>{
 if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];view.value='3d';draw();return;}
 const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*900/r.width,y=(e.clientY-r.top)*520/r.height;
 let nearest=null,distance=100;
 projected.forEach(p=>{const d=(p.x-x)**2+(p.y-y)**2;if(d<distance){nearest=p;distance=d;}});
 if(nearest){const p=data[nearest.index];detail.textContent=`查询行 ${nearest.index+1}：X=${p[0].toPrecision(6)}，Y=${p[1].toPrecision(6)}，Z=${p[2].toPrecision(6)}；预测=${p[3].toPrecision(6)}`;}
 else detail.textContent='悬停查询一个点的标准化坐标和预测值。';
});
view.addEventListener('change',draw);document.getElementById('point-reset').addEventListener('click',()=>{yaw=.6;pitch=-.4;view.value='3d';draw();});
draw();
})();
</script></section>""".replace("__POINT_DATA__", payload)
