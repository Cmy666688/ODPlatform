/* DeepSeek 助手。凭据由服务端读取,浏览器只保存当前页内会话。 */
"use strict";
(() => {
  let configured = false, sending = false, controller = null, turns = [], contextId = "";
  const conversation = $("#assistant-messages"), form = $("#assistant-form"), question = $("#assistant-question");
  function reset() {
    controller?.abort(); controller = null; turns = [];
    conversation.replaceChildren(); $("#assistant-feedback").textContent = "";
  }
  function controls() {
    $("#assistant-send").disabled = !configured || sending;
    $("#assistant-job").disabled = sending;
    $("#assistant-clear").disabled = sending;
    question.disabled = sending;
  }
  async function load() {
    if (location.hash !== "#assistant" || sending) return;
    try {
      const config = await api("/api/assistant/config"), changed = configured !== config.configured; configured = config.configured;
      $("#assistant-status").textContent = configured ? `DeepSeek · ${config.model} · 已配置` : "DeepSeek · 未配置";
      $("#assistant-status").className = `badge ${configured ? "success" : "warning"}`;
      if (changed || !$("#assistant-feedback").textContent) $("#assistant-feedback").textContent = configured ? "发送问题后调用 API，按 DeepSeek 账户计费。" : "智能助手尚未启用，请由管理员完成服务配置。";
      const options = state.jobs.filter((job) => job.config?.data === state.datasetId);
      const select = $("#assistant-job"), previous = select.value;
      select.innerHTML = '<option value="">仅使用数据质检信息</option>' + options.map((job) => `<option value="${esc(job.id)}">${esc(job.name)} · ${esc(statusNames[job.status] || job.status)}</option>`).join("");
      if (options.some((job) => job.id === previous)) select.value = previous;
      $("#assistant-dataset").textContent = state.overview?.dataset?.name || state.datasetId;
      const nextId = `${state.datasetId}/${select.value}`;
      if (contextId && contextId !== nextId) reset(); contextId = nextId;
    } catch (error) {
      configured = false; $("#assistant-status").textContent = "配置读取失败"; $("#assistant-feedback").textContent = error.message;
    } finally { controls(); }
  }
  function append(role, content) {
    const block = document.createElement("article"); block.className = `assistant-turn ${role}`;
    const label = document.createElement("strong"); label.textContent = role === "user" ? "你" : "DeepSeek";
    const text = document.createElement("p"); text.textContent = content; block.append(label, text); conversation.append(block);
    block.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault(); const message = question.value.trim(); if (!message || sending || !configured) return;
    sending = true; controls(); controller = new AbortController();
    const dataset = state.datasetId, job = $("#assistant-job").value;
    append("user", message); $("#assistant-feedback").textContent = "DeepSeek 正在解读…";
    try {
      const result = await api("/api/assistant/chat", { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ message, dataset, job_id: job, history: turns.slice(-12) }) });
      if (dataset !== state.datasetId || job !== $("#assistant-job").value) return;
      append("assistant", result.answer); turns.push({ role: "user", content: message }, { role: "assistant", content: result.answer.slice(0, 6000) });
      question.value = ""; $("#assistant-feedback").textContent = `${result.model}${result.usage.total_tokens ? ` · ${result.usage.total_tokens} tokens` : ""}`;
    } catch (error) { if (error.name !== "AbortError") $("#assistant-feedback").textContent = error.message; }
    finally { sending = false; controller = null; controls(); }
  });
  $("#assistant-check").addEventListener("click", load);
  $("#assistant-clear").addEventListener("click", reset);
  $("#assistant-job").addEventListener("change", () => { reset(); contextId = `${state.datasetId}/${$("#assistant-job").value}`; });
  $$('[data-question]').forEach((button) => button.addEventListener("click", () => { question.value = button.dataset.question; question.focus(); }));
  window.addEventListener("hashchange", load);
  window.addEventListener("odp:overview", () => {
    if (contextId && !contextId.startsWith(`${state.datasetId}/`)) reset();
    if (!sending) load();
  });
  load();
})();
