/* ODPlatform 工作台。图表与状态只使用 API 返回的真实数据。 */
"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { overview: null, defaults: null, jobs: [], monitorId: "", detailId: "", chart: "loss", uploadId: null, gallery: [], source: "reference", busy: false };
const statusNames = { queued: "排队中", running: "运行中", cancelling: "正在停止", cancelled: "已停止", completed: "已完成", failed: "失败" };
const phaseNames = { validating: "数据质检", starting: "准备模型", training: "训练中", validating_epoch: "验证中", epoch_complete: "本轮完成", testing: "独立测试", syncing: "同步结果", inference: "推理中", completed: "已完成", failed: "失败", cancelled: "已停止" };
const checkNames = { mask_pair_existence: "图像与掩膜配对", mask_decode_and_format: "解码与标注格式", mask_dimensions: "图像与掩膜尺寸", mask_alpha_channels: "Alpha 通道处理", mask_foreground_coverage: "前景覆盖率", mask_split_uniqueness: "跨集合像素重复", mask_source_group_overlap: "同源分组风险", mask_manifest_lineage: "图像与掩膜身份" };
const active = (job) => ["queued", "running", "cancelling"].includes(job.status);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const number = (value, places = 4) => Number.isFinite(value) ? value.toFixed(places) : "—";
const percent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const badge = (label, kind = "neutral") => `<span class="badge ${kind}">${esc(label)}</span>`;
const statusBadge = (job) => badge(job.cancel_requested && active(job) ? "正在停止" : statusNames[job.status] || job.status, job.status === "completed" ? "success" : job.status === "failed" ? "danger" : active(job) ? "warning" : "neutral");
const empty = (title, message) => `<div class="empty"><strong>${esc(title)}</strong><span>${esc(message)}</span></div>`;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { const body = await response.json(); message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch (_) { /* 保留 HTTP 状态 */ }
    throw new Error(message);
  }
  return response.json();
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message; element.classList.toggle("error", error); element.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { element.hidden = true; }, 4500);
}

function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${Math.max(0, Math.floor(seconds))} 秒`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒`;
  return `${Math.floor(seconds / 3600)} 小时 ${Math.floor(seconds % 3600 / 60)} 分`;
}

function route() {
  const name = location.hash.slice(1) || "overview";
  const valid = ["overview", "data", "experiments", "results"].includes(name) ? name : "overview";
  $$(".page").forEach((page) => { page.hidden = page.id !== `page-${valid}`; });
  $$(".nav-link").forEach((link) => link.classList.toggle("active", link.dataset.page === valid));
  $("#breadcrumb-title").textContent = { overview: "监控概览", data: "数据质检", experiments: "实验记录", results: "结果可视化" }[valid];
  window.scrollTo(0, 0);
  if (valid === "results") loadGallery().catch((error) => toast(error.message, true));
}

function drawChart(history) {
  const series = state.chart === "loss"
    ? [{ label: "训练损失", color: "#198c70", values: history.map((row) => row.train_loss) }, { label: "验证损失", color: "#dfad57", values: history.map((row) => row.validation.loss) }]
    : [{ label: "F1", color: "#198c70", values: history.map((row) => row.validation.F1) }, { label: "前景 IoU", color: "#7595b7", values: history.map((row) => row.validation.foreground_IoU) }, { label: "mIoU", color: "#dfad57", values: history.map((row) => row.validation.mIoU) }];
  $("#chart-legend").innerHTML = series.map((s) => `<span style="--series:${s.color}">${s.label}</span>`).join("");
  const width = 700, height = 225, left = 40, right = 18, top = 13, bottom = 25;
  const values = series.flatMap((s) => s.values).filter(Number.isFinite);
  const max = state.chart === "metrics" ? 1 : Math.max(0.1, ...values) * 1.12;
  let content = "";
  for (let i = 0; i <= 4; i++) {
    const y = top + (height - top - bottom) * i / 4;
    content += `<line x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" stroke="#edf1f4" stroke-dasharray="3 4"/><text x="${left - 10}" y="${y + 3}" text-anchor="end" font-size="9" fill="#a6b1bb">${(max * (4 - i) / 4).toFixed(2)}</text>`;
  }
  if (history.length) {
    const x = (i) => left + (width - left - right) * i / Math.max(1, history.length - 1);
    const y = (value) => top + (height - top - bottom) * (1 - value / max);
    for (const s of series) {
      const points = s.values.map((value, i) => `${x(i)},${y(value)}`).join(" ");
      content += `<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
      for (let i = 0; i < s.values.length; i++) content += `<circle cx="${x(i)}" cy="${y(s.values[i])}" r="${history.length < 8 ? 3 : 2}" fill="${s.color}"><title>Epoch ${history[i].epoch} · ${s.label}: ${number(s.values[i])}</title></circle>`;
    }
    const labels = [...new Set([0, Math.floor((history.length - 1) / 2), history.length - 1])];
    for (const i of labels) content += `<text x="${x(i)}" y="${height - 5}" text-anchor="middle" font-size="9" fill="#a6b1bb">${history[i].epoch}</text>`;
  }
  $("#training-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${state.chart === "loss" ? "训练与验证损失曲线" : "验证集指标曲线"}">${content}</svg>${history.length ? "" : '<div class="chart-empty"><strong>⌁</strong><span>训练开始后，指标曲线将在这里更新</span></div>'}`;
}

function jobTable(jobs, limit) {
  if (!jobs.length) return empty("还没有实验记录", "GPU 就绪后提交完整训练，或前往结果页体验图片推理。");
  return `<div class="table-wrap"><table><thead><tr><th>实验名称</th><th>类型</th><th>状态</th><th>创建时间</th><th></th></tr></thead><tbody>${jobs.slice(0, limit).map((job) => `<tr><td>${esc(job.name)}</td><td>${job.kind === "training" ? "完整训练" : "图片推理"}</td><td>${statusBadge(job)}</td><td>${esc(new Date(job.created_at).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }))}</td><td><button data-detail="${job.id}">查看 →</button></td></tr>`).join("")}</tbody></table></div>`;
}

function renderOverview() {
  const data = state.overview, split = data.dataset.splits;
  const training = state.jobs.filter((job) => job.kind === "training");
  if (!state.monitorChosen && training.length) {
    state.monitorId = (training.find(active) || training[0]).id;
    state.monitorChosen = true;
  }
  const monitored = state.jobs.find((job) => job.id === state.monitorId);
  const hardware = monitored?.remote_hardware || data.hardware;
  $("#dataset-counts").textContent = `${split.train + split.val + split.test} 张图像　·　${split.train} 训练 / ${split.val} 验证 / ${split.test} 测试`;
  $("#device-name").textContent = hardware.cuda_available ? hardware.device : "CPU / CUDA 未就绪";
  $("#device-note").textContent = `${monitored?.execution === "remote" ? "远程 GPU · " : ""}PyTorch ${hardware.torch}${monitored?.execution === "remote" && !active(monitored) ? " · 实验结束时采样" : ""}`;
  $("#gpu-badge").textContent = hardware.cuda_available ? "GPU 就绪" : "等待 GPU";
  $("#gpu-badge").className = `badge ${hardware.cuda_available ? "success" : "neutral"}`;
  $("#cpu-value").textContent = `${number(hardware.cpu_percent, 0)}%`;
  $("#memory-value").textContent = `${number(hardware.memory_percent, 0)}%`;
  $("#cpu-meter").style.width = `${hardware.cpu_percent}%`;
  $("#memory-meter").style.width = `${hardware.memory_percent}%`;
  $("#memory-caption").textContent = `${hardware.memory_used_gb} / ${hardware.memory_gb} GB`;
  $("#gpu-resources").hidden = !hardware.cuda_available;
  $("#gpu-utilization").textContent = hardware.gpu_percent === null || hardware.gpu_percent === undefined ? "暂不可用" : `${number(hardware.gpu_percent, 0)}%`;
  $("#gpu-memory").textContent = hardware.gpu_memory_gb ? `${hardware.gpu_memory_used_gb} / ${hardware.gpu_memory_gb} GB` : "暂不可用";
  const readiness = data.hardware.cuda_available ? "CUDA 已就绪，可提交完整实验。" : "本机 CUDA GPU 尚未就绪，不能从本机启动新训练。";
  $("#config-readiness").textContent = readiness;
  $("#config-readiness").classList.toggle("ready", data.hardware.cuda_available);
  const remoteNote = monitored?.status === "completed" ? "远程实验已结束，权重、指标和可视化已同步到本机。" : !active(monitored || {}) ? "远程实验已结束，可在实验记录中查看日志。" : Date.now() - Date.parse(monitored.synced_at) > 20000 ? "远程进度同步暂时中断，正在等待连接恢复。" : "当前实验在远程 GPU 执行，进度与产物自动同步到本机。";
  $("#environment-message").textContent = monitored?.execution === "remote" ? remoteNote : readiness;
  $("#environment-message").classList.toggle("ready", hardware.cuda_available);
  $("#start-training").disabled = !data.hardware.cuda_available || state.jobs.some(active);
  $("#start-training").textContent = data.hardware.cuda_available ? "开始完整训练" : "GPU 就绪后开始训练";
  const select = $("#monitor-job");
  const options = '<option value="">新实验 / 待启动</option>' + training.map((job) => `<option value="${job.id}">${esc(job.name)} · ${statusNames[job.status]}</option>`).join("");
  if (select.innerHTML !== options) { select.innerHTML = options; select.value = state.monitorId; }
  renderMonitor();
  $("#recent-jobs").innerHTML = jobTable(state.jobs, 3);
  $("#all-jobs").innerHTML = jobTable(state.jobs, 100);
  $("#job-count").textContent = `${state.jobs.length} 个任务`;
  renderQuality(data.quality);
  const previous = $("#result-source").value;
  const sources = '<option value="reference">官方权重参考结果</option>' + state.jobs.filter((job) => job.status === "completed").map((job) => `<option value="${job.id}">${esc(job.name)} · ${job.kind === "training" ? "训练验收" : "推理结果"}</option>`).join("");
  if ($("#result-source").innerHTML !== sources) { $("#result-source").innerHTML = sources; $("#result-source").value = previous || "reference"; }
  const modelValue = $("#infer-model").value;
  const models = '<option value="official">官方 CrackMap 参考权重</option>' + state.jobs.filter((job) => job.kind === "training" && job.status === "completed").map((job) => `<option value="trained:${job.id}">${esc(job.name)} · best</option>`).join("");
  if ($("#infer-model").innerHTML !== models) { $("#infer-model").innerHTML = models; $("#infer-model").value = modelValue; }
  $("#run-inference").disabled = !state.uploadId || state.jobs.some(active);
  const inferJob = state.jobs.find((job) => job.id === state.inferId);
  if (inferJob) {
    $("#infer-feedback").textContent = `${statusNames[inferJob.status]}${inferJob.error ? " · " + inferJob.error : ""}`;
    if (inferJob.status === "completed" && state.loadedInfer !== inferJob.id) {
      state.loadedInfer = inferJob.id; $("#result-source").value = inferJob.id;
      loadGallery().catch((error) => toast(error.message, true));
    }
  }
}

function renderMonitor() {
  const job = state.jobs.find((row) => row.id === state.monitorId);
  const history = job?.history || [], last = history.at(-1), epochs = job?.config?.epochs || state.draft?.config?.epochs || state.defaults?.epochs || 50;
  $("#training-name").textContent = job?.name || "等待新实验";
  $("#training-status").outerHTML = `<span id="training-status" class="badge ${job ? job.status === "completed" ? "success" : job.status === "failed" ? "danger" : "neutral" : "neutral"}">${job ? esc(statusNames[job.status]) : "待训练"}</span>`;
  $("#training-dot").classList.toggle("running", Boolean(job && active(job)));
  $("#metric-epoch").innerHTML = `${history.length || "—"} <small>/ ${epochs}</small>`;
  $("#metric-loss").textContent = number(last?.train_loss);
  $("#metric-f1").textContent = percent(last?.validation?.F1);
  $("#metric-time").textContent = duration(job?.elapsed_seconds);
  $("#metric-eta").textContent = job?.status === "completed" ? "实验已结束" : job?.eta_seconds ? `预计剩余 ${duration(job.eta_seconds)}` : "启动后显示预计剩余时间";
  $("#metric-epoch-note").textContent = job ? `${phaseNames[job.phase] || job.phase}${job.epoch ? ` · 第 ${job.epoch} 轮` : ""}` : "完整训练";
  $("#chart-update").textContent = last ? `已记录 ${history.length} 轮` : "等待训练开始";
  drawChart(history);
  const progress = Math.max(0, Math.min(1, job?.progress || 0));
  $("#progress-label").textContent = percent(progress); $("#progress-bar").style.width = `${progress * 100}%`;
  $("#step-data").className = state.overview?.dataset?.available && state.overview?.quality?.overall_severity !== "ERROR" ? "done" : "";
  $("#step-train").className = job?.phase === "testing" || job?.status === "completed" ? "done" : job && active(job) ? "active" : "";
  $("#step-test").className = job?.status === "completed" ? "done" : job?.phase === "testing" ? "active" : "";
  $("#step-results").className = job?.status === "completed" ? "done" : "";
  $("#batch-progress").textContent = job?.error || (job ? `${phaseNames[job.phase] || statusNames[job.status]}${job.batch ? ` · ${job.batch} / ${job.batches || "—"} 批` : ""}${job.cancel_requested ? " · 已请求停止，等待当前计算结束" : ""}` : "训练尚未启动。GPU 就绪后可提交完整实验。");
}

function renderQuality(quality) {
  const checks = quality?.results || [], warning = checks.filter((row) => row.severity === "WARNING"), errors = checks.filter((row) => row.severity === "ERROR");
  const severity = quality?.overall_severity || "未检查";
  $("#quality-badge").textContent = errors.length ? "需修复" : warning.length ? "风险提示" : quality ? "通过" : "未检查";
  $("#quality-badge").className = `badge ${errors.length ? "danger" : warning.length ? "warning" : "success"}`;
  $("#quality-summary").innerHTML = quality ? `<div class="summary-line"><strong>${checks.length - warning.length - errors.length}</strong> 项检查通过 / ${checks.length} 项</div><div class="summary-copy ${warning.length ? "risk" : ""}">${errors.length ? "存在错误，需处理后训练。" : warning.length ? "存在潜在同源跨集风险，详见质检报告。" : "未发现阻断问题。"}</div>` : empty("暂无质检记录", "前往数据页运行检查。");
  $("#data-dot").hidden = !warning.length && !errors.length;
  const split = state.overview.dataset.splits;
  $("#data-metrics").innerHTML = [["训练集", split.train, "用于模型参数学习"], ["验证集", split.val, "用于选择 best 权重"], ["测试集", split.test, "独立效果验收"], ["质检结果", severity, `${checks.length} 项语义数据检查`]].map(([label, value, note]) => `<article class="metric-card"><span>${label}</span><strong>${esc(value)}</strong><div>${note}</div></article>`).join("");
  $("#quality-checks").innerHTML = checks.length ? checks.map((row) => `<div class="check-row"><div><strong>${esc(checkNames[row.name] || row.name)}</strong><p>${esc(row.summary)}</p></div>${badge({ PASS: "通过", INFO: "信息", WARNING: "风险提示", ERROR: "错误" }[row.severity], row.severity === "ERROR" ? "danger" : row.severity === "WARNING" ? "warning" : "success")}</div>`).join("") : empty("尚未检查", "点击重新质检检查当前数据集。");
  const groups = checks.find((row) => row.name === "mask_source_group_overlap")?.details?.groups || {};
  $("#source-groups").innerHTML = Object.entries(groups).map(([name, splits]) => `<span>${esc(name)} · ${esc(splits.join(" / "))}</span>`).join("") || '<span>无分组风险记录</span>';
}

async function refresh(silent = false) {
  if (state.busy) return;
  state.busy = true;
  try {
    state.overview = await api("/api/overview"); state.jobs = state.overview.jobs;
    $("#connection-dot").classList.add("connected"); $("#connection-text").textContent = "服务已连接";
    renderOverview();
    if (state.detailId) await showDetail(state.detailId, false);
  } catch (error) {
    $("#connection-dot").classList.remove("connected"); $("#connection-text").textContent = "服务连接中断";
    if (!silent) toast(error.message, true);
  } finally { state.busy = false; }
}

async function showDetail(id, navigate = true) {
  const previousLog = state.detailId === id ? $("#job-log")?.innerHTML || "" : "";
  state.detailId = id;
  const job = state.jobs.find((row) => row.id === id) || await api(`/api/jobs/${id}`);
  const container = $("#job-detail"); container.hidden = false;
  container.innerHTML = `<div class="card-heading"><h2>${esc(job.name)}</h2>${statusBadge(job)}</div><div class="detail-grid"><div>任务 ID<strong>${esc(job.id.slice(0, 12))}</strong></div><div>当前阶段<strong>${esc(phaseNames[job.phase] || job.phase)}</strong></div><div>运行时长<strong>${duration(job.elapsed_seconds)}</strong></div></div>${job.error ? `<p class="environment-message">${esc(job.error)}</p>` : ""}<div class="detail-actions"><button class="button secondary" data-log="${id}">查看日志</button>${job.kind === "training" ? `<button class="button secondary" data-monitor="${id}">监控此实验</button>` : ""}${active(job) ? `<button class="button secondary" data-cancel="${id}" ${job.cancel_requested ? "disabled" : ""}>${job.cancel_requested ? "已请求停止" : "停止任务"}</button>` : ""}${job.status === "completed" ? `<button class="button primary" data-results="${id}">查看结果</button>` : ""}</div><div id="job-log"></div>`;
  $("#job-log").innerHTML = previousLog;
  if (navigate) location.hash = "experiments";
}

async function loadGallery() {
  const source = $("#result-source").value || "reference";
  state.source = source; $("#result-panel-label").textContent = source === "reference" || state.jobs.find((job) => job.id === source)?.kind === "training" ? "原图 · GT · 预测 · 误差" : "原图 · 概率 · 预测"; $("#evaluation-metrics").innerHTML = ""; $("#result-downloads").innerHTML = "";
  if (source === "reference") {
    const data = await api("/api/reference"); state.gallery = data.images;
    $("#gallery-description").textContent = "官方权重参考结果 · 24 张独立测试图 · 不代表本次完整训练效果";
  } else {
    const artifacts = await api(`/api/jobs/${source}/artifacts`);
    state.gallery = artifacts.filter((entry) => entry.path.includes("comparisons/") && entry.name.endsWith(".png"));
    const job = state.jobs.find((row) => row.id === source);
    $("#gallery-description").textContent = job?.kind === "training" ? "本次训练最佳权重 · 独立 test 集验收" : "图片推理结果 · 无 GT 时展示概率图与预测叠加";
    if (job?.metrics) $("#evaluation-metrics").innerHTML = [["F1", job.metrics.F1], ["前景 IoU", job.metrics.foreground_IoU], ["mIoU", job.metrics.mIoU], ["精确率", job.metrics.precision], ["召回率", job.metrics.recall]].map(([label, value]) => `<div>${label}<strong>${percent(value)}</strong></div>`).join("");
    $("#result-downloads").innerHTML = artifacts.filter((entry) => ["report.json", "report.md", "per_image.csv", "training_curves.png", "best.pth"].includes(entry.name)).map((entry) => `<a href="${encodeURI(entry.url)}?download=true" download>${esc(entry.name)} ↓</a>`).join("");
    if (job?.kind === "training" && job.quality_status === "needs_review") $("#gallery-description").textContent += " · 未设精度目标，效果待人工评审";
    if (job?.kind === "training" && job.quality_status === "failed") $("#gallery-description").textContent += " · 未达到配置的 F1 验收目标";
    if (job?.kind === "training" && job.quality_status === "passed") $("#gallery-description").textContent += " · 达到配置的 F1 验收目标";
  }
  $("#gallery-tools").hidden = !state.gallery.length;
  $("#gallery-image").innerHTML = state.gallery.map((entry, index) => `<option value="${index}">${esc(entry.name)}</option>`).join("");
  $("#gallery-thumbnails").innerHTML = state.gallery.map((entry, index) => `<button class="thumbnail ${index === 0 ? "selected" : ""}" data-image="${index}" aria-label="查看第 ${index + 1} 张结果"><img loading="lazy" src="${encodeURI(entry.url)}" alt="${esc(entry.name)}"></button>`).join("");
  displayImage(0);
}

function displayImage(index) {
  const entry = state.gallery[index];
  $("#gallery-preview").innerHTML = entry ? `<a href="${encodeURI(entry.url)}" target="_blank" rel="noreferrer" title="打开原尺寸图片"><img src="${encodeURI(entry.url)}" alt="分割对比结果 ${esc(entry.name)}，点击查看原尺寸"></a>` : empty("暂无可视化产物", "完整训练或图片推理完成后，可在这里查看结果。");
  $("#gallery-image").value = String(index);
  if (entry) { $("#download-result").href = encodeURI(entry.url); $("#download-result").download = entry.name; }
  $$(".thumbnail").forEach((button) => button.classList.toggle("selected", Number(button.dataset.image) === index));
}

async function uploadImage(file) {
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) return toast("图片不能超过 20 MB", true);
  $("#upload-description").textContent = "正在上传…";
  try {
    const form = new FormData(); form.append("file", file);
    const data = await api("/api/uploads", { method: "POST", body: form }); state.uploadId = data.id;
    $("#upload-description").textContent = `${file.name} · ${data.width} × ${data.height}`;
    $("#run-inference").disabled = state.jobs.some(active);
  } catch (error) { state.uploadId = null; $("#upload-description").textContent = "上传失败，请重新选择图片"; $("#run-inference").disabled = true; toast(error.message, true); }
}

function draft() {
  const values = new FormData($("#training-form"));
  return { name: values.get("name"), config: { ...state.defaults, epochs: Number(values.get("epochs")), batch_size_train: Number(values.get("batch_size_train")), lr: Number(values.get("lr")), seed: Number(values.get("seed")) }, min_f1: values.get("min_f1") === "" ? null : Number(values.get("min_f1")) };
}

function openConfig() {
  let saved;
  try { saved = JSON.parse(localStorage.getItem("odp-training-draft")); } catch (_) { /* 无合法草稿时采用默认配方 */ }
  if (saved) {
    const form = $("#training-form"); form.elements.name.value = saved.name;
    for (const field of ["epochs", "batch_size_train", "lr", "seed"]) form.elements[field].value = saved.config[field];
    form.elements.min_f1.value = saved.min_f1 ?? "";
  }
  $("#config-dialog").showModal();
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("button"); if (!button) return;
  try {
    if (button.hasAttribute("data-open-config")) openConfig();
    if (button.dataset.chart) { state.chart = button.dataset.chart; $$("[data-chart]").forEach((item) => item.classList.toggle("selected", item === button)); renderMonitor(); }
    if (button.dataset.detail) await showDetail(button.dataset.detail);
    if (button.dataset.monitor) { state.monitorId = button.dataset.monitor; state.monitorChosen = true; $("#monitor-job").value = state.monitorId; location.hash = "overview"; renderOverview(); }
    if (button.dataset.results) { $("#result-source").value = button.dataset.results; location.hash = "results"; await loadGallery(); }
    if (button.dataset.cancel) { await api(`/api/jobs/${button.dataset.cancel}/cancel`, { method: "POST" }); toast("已请求停止，等待当前计算结束"); await refresh(); }
    if (button.dataset.log) {
      const response = await fetch(`/api/jobs/${button.dataset.log}/files/worker.log`);
      if (!response.ok) throw new Error("日志尚未生成");
      $("#job-log").innerHTML = `<pre class="log">${esc((await response.text()).slice(-14000))}</pre>`;
    }
    if (button.dataset.image !== undefined) displayImage(Number(button.dataset.image));
  } catch (error) { toast(error.message, true); }
});

$("#refresh").addEventListener("click", () => refresh());
$("#monitor-job").addEventListener("change", (event) => { state.monitorId = event.target.value; state.monitorChosen = true; renderOverview(); });
$("#close-config").addEventListener("click", () => $("#config-dialog").close());
$("#save-draft").addEventListener("click", () => { if (!$("#training-form").reportValidity()) return; state.draft = draft(); localStorage.setItem("odp-training-draft", JSON.stringify(state.draft)); $("#config-dialog").close(); renderMonitor(); toast("训练配置已保存，未启动训练"); });
$("#training-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const button = $("#start-training"); button.disabled = true;
  try {
    const request = { ...draft(), kind: "training", require_cuda: true };
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    state.monitorId = job.id; $("#config-dialog").close(); await refresh(); location.hash = "overview"; toast("完整训练任务已提交");
  } catch (error) { toast(error.message, true); } finally { renderOverview(); }
});
$("#validate-data").addEventListener("click", async (event) => {
  const button = event.target; button.disabled = true; button.textContent = "检查中…";
  try { await api("/api/datasets/crackmap/validate", { method: "POST" }); await refresh(); toast("数据质检已完成"); } catch (error) { toast(error.message, true); } finally { button.disabled = false; button.textContent = "重新质检"; }
});
$("#image-upload").addEventListener("change", (event) => uploadImage(event.target.files[0]));
$("#upload-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("#upload-zone").classList.add("dragging"); });
$("#upload-zone").addEventListener("dragleave", () => $("#upload-zone").classList.remove("dragging"));
$("#upload-zone").addEventListener("drop", (event) => { event.preventDefault(); $("#upload-zone").classList.remove("dragging"); uploadImage(event.dataTransfer.files[0]); });
$("#infer-threshold").addEventListener("input", (event) => { $("#threshold-value").textContent = Number(event.target.value).toFixed(2); });
$("#run-inference").addEventListener("click", async () => {
  $("#run-inference").disabled = true;
  try {
    const request = { kind: "inference", name: "单图裂缝分割", upload_id: state.uploadId, model_id: $("#infer-model").value, config: { threshold: Number($("#infer-threshold").value) } };
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    state.inferId = job.id; $("#infer-feedback").textContent = "推理任务已提交…"; await refresh(); toast("正在生成分割结果");
  } catch (error) { toast(error.message, true); $("#run-inference").disabled = !state.uploadId; }
});
$("#result-source").addEventListener("change", () => loadGallery().catch((error) => toast(error.message, true)));
$("#gallery-image").addEventListener("change", (event) => displayImage(Number(event.target.value)));
window.addEventListener("hashchange", route);
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(true); });

async function initialize() {
  try { state.defaults = (await api("/api/config")).defaults; } catch (error) { toast(error.message, true); }
  try { state.draft = JSON.parse(localStorage.getItem("odp-training-draft")); } catch (_) { /* 默认配置 */ }
  await refresh(); route();
  setInterval(() => { if (!document.hidden) refresh(true); }, 4000);
}
initialize();
