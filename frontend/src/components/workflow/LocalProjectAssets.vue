<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from "vue";
import { localWorkbenchProjectId as selected } from "../../localWorkbenchState";
import { resetConversation } from "../assistant/conversationState";
const emit = defineEmits<{ projectSelected: [] }>();
const projects = ref<Array<{id:string;name:string}>>([]), name = ref("");
const busy = ref(false), status = ref(""), error = ref("");
const file = ref<File | null>(null), input = ref<HTMLInputElement | null>(null);
const controller = new AbortController();
let pending: {name:string;id:string} | undefined;
onBeforeUnmount(()=>controller.abort());
async function action(operation:string, arguments_:Record<string,unknown>, id: string=crypto.randomUUID()) {
 const response=await fetch('/api/v1/workbench/actions',{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,body:JSON.stringify({operation,arguments:arguments_,operation_id:id})});
 if(!response.ok) throw Error('request failed');const receipt=await response.json();
 if(receipt.status!=='completed'||receipt.operation!==operation)throw Error('incomplete');return receipt.result;
}
async function load(){
 try { const result=await action('project_list',{});if(!controller.signal.aborted)projects.value=result.projects; }
 catch { if(!controller.signal.aborted)error.value='暂时无法读取工区，请重试。'; }
}
function select(){resetConversation();status.value='';error.value='';emit('projectSelected');}
async function create(){
 const text=name.value.trim();if(!text||busy.value)return;busy.value=true;error.value='';
 if(pending?.name!==text)pending={name:text,id:crypto.randomUUID()};
 try {const result=await action('project_create',{name:text},pending.id);if(controller.signal.aborted)return;await load();selected.value=result.project_id;select();name.value='';pending=undefined;}
 catch {if(!controller.signal.aborted)error.value='创建未完成，请重试。';}
 finally {busy.value=false;}
}
async function upload(){
 const chosen=file.value,project=selected.value;if(!chosen||!project||busy.value)return;
 if(!/\.las$/i.test(chosen.name)||!chosen.size||chosen.size>120*1024*1024){error.value='请选择非空 LAS 文件，最大 120 MiB。';return;}
 busy.value=true;status.value='正在上传并封存…';error.value='';
 try {
  const response=await fetch(`/api/v1/workbench/projects/${encodeURIComponent(project)}/assets/log?filename=${encodeURIComponent(chosen.name)}`,{method:'POST',body:chosen,headers:{'Content-Type':'application/octet-stream'},signal:controller.signal});
  if(!response.ok)throw Error('upload failed');const receipt=await response.json();
  if(receipt.asset?.project_id!==project||receipt.asset?.role!=='log')throw Error('scope mismatch');
  if(controller.signal.aborted)return;
  const archived=receipt.versioning?.status==='archived'&&receipt.revision_id===receipt.versioning.revision_id;
  status.value=archived?'测井数据已封存 · DVC 已归档':'数据已上传，归档未完成，可请助手重试归档';
  file.value=null;if(input.value)input.value.value='';
 } catch {if(!controller.signal.aborted){status.value='';error.value='上传未完成，请检查工区与文件后重试。';}}
 finally {busy.value=false;}
}
onMounted(load);
</script>
<template>
 <div class="local-project-assets">
  <strong>工区与测井数据</strong>
  <label>选择工区<select v-model="selected" :disabled="busy" aria-label="选择工区" @change="select"><option value="">请选择工区</option><option v-for="project in projects" :key="project.id" :value="project.id">{{project.name}}</option></select></label>
  <div class="project-create"><input v-model="name" :disabled="busy" aria-label="新工区名称" placeholder="新工区名称" maxlength="120" /><button type="button" :disabled="busy||!name.trim()" @click="create">创建工区</button></div>
  <input ref="input" type="file" accept=".las" hidden aria-label="选择 LAS 测井文件" @change="file=($event.target as HTMLInputElement).files?.[0]||null" />
  <button type="button" :disabled="busy||!selected" @click="input?.click()">＋ 选择 LAS 文件</button>
  <span v-if="file">{{file.name}}</span><button type="button" :disabled="busy||!file||!selected" @click="upload">{{busy?'处理中…':'上传并归档'}}</button>
  <p v-if="status" role="status">{{status}}</p><p v-if="error" role="alert">{{error}}</p>
 </div>
</template>
<style scoped>
.local-project-assets{display:grid;gap:9px;font-size:12px;color:#475569}.local-project-assets strong{color:#0f172a}.local-project-assets label{display:grid;gap:5px}.local-project-assets select,.local-project-assets input{min-width:0;width:100%;padding:7px;border:1px solid #e2e8f0;border-radius:7px;background:#fff;font:inherit}.project-create{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px}.local-project-assets button{padding:7px 9px;color:#1668dc;background:#eff6ff;border:1px solid #dbeafe;border-radius:7px;cursor:pointer}.local-project-assets button:disabled{opacity:.5;cursor:default}.local-project-assets p{margin:0;overflow-wrap:anywhere}
</style>
