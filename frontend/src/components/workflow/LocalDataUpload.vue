<script setup lang="ts">
import { onBeforeUnmount, ref } from "vue";
import { uploadRawWellFile } from "../../api";
const emit = defineEmits<{ uploaded: [path: string] }>();
const input = ref<HTMLInputElement | null>(null);
const busy = ref(false), name = ref(""), error = ref("");
let active = true;
onBeforeUnmount(() => { active = false; });
async function upload(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file || busy.value) return;
  if (!/\.(las|csv|txt)$/i.test(file.name) || file.size > 128 * 1024 * 1024) {
    error.value = "请选择不超过 128 MiB 的 LAS、CSV 或 TXT 文件。"; return;
  }
  busy.value = true; error.value = "";
  try {
    const result = await uploadRawWellFile(file);
    if (active) { name.value = result.name; emit("uploaded", result.path); }
  } catch { if (active) error.value = "上传未完成，请检查文件后重试。"; }
  finally { if (active) busy.value = false; }
}
</script>
<template>
  <div class="local-data-upload">
    <input ref="input" type="file" accept=".las,.csv,.txt" hidden aria-label="选择测井文件" @change="upload" />
    <button type="button" class="primary-button" :disabled="busy" @click="input?.click()">{{ busy ? '正在上传…' : '＋ 选择测井文件' }}</button>
    <p v-if="name" role="status">{{ name }} · 已上传</p>
    <p v-if="error" role="alert">{{ error }}</p>
  </div>
</template>
<style scoped>
.local-data-upload { display: grid; justify-items: start; gap: 12px; margin-top: 24px; }
.local-data-upload p { margin: 0; color: var(--muted); overflow-wrap: anywhere; }
</style>
