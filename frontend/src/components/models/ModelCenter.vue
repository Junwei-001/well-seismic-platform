<script setup lang="ts">
import type { Capabilities } from "../../api";

defineProps<{ capabilities: Capabilities | null }>();
</script>

<template>
  <section class="section-panel">
    <div class="section-heading">
      <div><h2>模型组件</h2><p>每个组件拥有独立 ID、输入输出契约、版本和实现状态，可单独替换。</p></div>
      <span class="count-badge">{{ capabilities?.models.length || 0 }} 个组件</span>
    </div>
    <p v-if="capabilities?.plugin_load_errors.length" class="error-message">
      {{ capabilities.plugin_load_errors.length }} 个外部模型插件加载失败；内置数据工作流不受影响，请检查插件版本。
    </p>
    <p v-if="capabilities?.runtime_plugin_contract.plugin_load_errors.length" class="error-message">
      {{ capabilities.runtime_plugin_contract.plugin_load_errors.length }} 个输入适配器、运行器或融合策略插件加载失败，请检查插件入口与契约版本。
    </p>
    <div class="model-grid">
      <article v-for="model in capabilities?.models" :key="model.id" class="model-card">
        <div class="model-card-head"><span>{{ model.category }}</span><b :class="{ builtin: model.status === '内置基线' }">{{ model.status }}</b></div>
        <h2>{{ model.name }}</h2><p>{{ model.description }}</p>
        <dl><dt>输入</dt><dd>{{ model.inputs.join("、") }}</dd><dt>输出</dt><dd>{{ model.outputs.join("、") }}</dd></dl>
        <footer><code>{{ model.id }}</code><span>{{ model.version }}</span></footer>
      </article>
    </div>
  </section>

  <section class="section-panel">
    <div class="section-heading">
      <div><h2>井震融合方案</h2><p>融合算法独立于下游解释任务；先形成带质量门控的统一表征，再由不同任务模型消费。</p></div>
      <span class="count-badge">{{ capabilities?.fusion_strategies.length || 0 }} 种策略</span>
    </div>
    <div class="model-data-lineage">
      <article class="ready"><span>01</span><div><strong>标准数据</strong><small>地震 / 测井 / 轨迹 / 掩码</small></div></article>
      <b>→</b>
      <article class="ready"><span>02</span><div><strong>井震对齐</strong><small>空间邻域 + 时深标定 + 不确定性</small></div></article>
      <b>→</b>
      <article class="ready"><span>03</span><div><strong>统一表征</strong><small>固定基线或可学习融合</small></div></article>
      <b>→</b>
      <article><span>04</span><div><strong>下游任务</strong><small>共享编码或任务专属头</small></div></article>
    </div>
    <div class="task-model-grid">
      <article v-for="strategy in capabilities?.fusion_strategies" :key="strategy.id">
        <div><span>{{ strategy.stage }}</span><b>{{ strategy.status }}</b></div>
        <h2>{{ strategy.name }}</h2>
        <p>{{ strategy.description }}</p>
        <dl class="capability-details">
          <div><dt>输入</dt><dd>{{ strategy.inputs.join(" / ") }}</dd></div>
          <div><dt>输出</dt><dd>{{ strategy.output }}</dd></div>
          <div><dt>建议用途</dt><dd>{{ strategy.recommended_for || "按任务配置" }}</dd></div>
        </dl>
        <code>{{ strategy.id }}</code>
      </article>
    </div>
  </section>

  <section class="section-panel">
    <div class="section-heading">
      <div><h2>下游解释任务契约</h2><p>任务定义业务语义与成果类型，模型负责输入张量和运行实现；新增任务不再修改页面中的固定清单。</p></div>
      <span class="count-badge">{{ capabilities?.prediction_tasks.length || 0 }} 个任务</span>
    </div>
    <div class="task-model-grid">
      <article v-for="task in capabilities?.prediction_tasks" :key="task.id">
        <div><span>契约 {{ task.contract_version }}</span><b>{{ task.status }}</b></div>
        <h2>{{ task.name }}</h2>
        <p>{{ task.description }}</p>
        <dl class="capability-details">
          <div><dt>输入模态</dt><dd>{{ task.required_modalities.join(" / ") }}</dd></div>
          <div><dt>标准输出</dt><dd>{{ task.outputs.join(" / ") }}</dd></div>
          <div><dt>评价指标</dt><dd>{{ task.evaluation_metrics.join(" / ") }}</dd></div>
        </dl>
        <code>{{ task.id }} · {{ task.runnable_model_ids.length }} 个可运行模型</code>
      </article>
    </div>
  </section>

  <section class="plugin-contract">
    <div><span>模型插件入口</span><code>{{ capabilities?.plugin_contract.entry_point_group }}</code></div>
    <p>任务入口 {{ capabilities?.interpretation_task_contract.entry_point_group }}；输入适配器入口 {{ capabilities?.runtime_plugin_contract.input_adapter_entry_point_group }}；运行器入口 {{ capabilities?.runtime_plugin_contract.prediction_runner_entry_point_group }}。</p>
  </section>
</template>
