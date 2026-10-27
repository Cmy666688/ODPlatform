/* 项目介绍、数据理解、保留策略和视频分析。沿用现有工作台 API。 */
"use strict";
(() => {
  let analysisId = "", analysisBusy = false, videoUpload = "", videoLoading = false, loadedVideo = "";
  const jsonPost = (path, value) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) });
  async function analysis() {
    if (analysisBusy || !state.overview || analysisId === state.datasetId) return;
    analysisBusy = true; const id = state.datasetId;
    try {
      const data = await api(`/api/datasets/${id}/statistics`); if (id !== state.datasetId) return;
      analysisId = id;
      const names = Array.isArray(data.names) ? data.names : Object.values(data.names);
      $("#analysis-summary").innerHTML = [["当前数据集", data.dataset], ["图像总数", data.sample_count], ["检测类别", names.length], ["采集分组", Object.keys(data.groups).length]].map(([label, value]) => `<article class="metric-card"><span>${esc(label)}</span><strong class="analysis-value">${esc(value)}</strong></article>`).join("");
      const splits = ["train", "val", "test"], labels = { train: "训练", val: "验证", test: "测试" };
      const counts = data.class_counts, max = Math.max(1, ...names.flatMap((name) => splits.map((split) => counts[split]?.[name] || 0)));
      $("#class-distribution").innerHTML = `<div class="table-scroll"><table class="data-table"><thead><tr><th>类别</th>${splits.map((split) => `<th>${labels[split]}</th>`).join("")}</tr></thead><tbody>${names.map((name) => `<tr><td>${esc(name)}</td>${splits.map((split) => { const count = counts[split]?.[name]; return `<td><div class="distribution-bar"><i class="${split}" style="width:${(count || 0)/max*100}%"></i></div>${count === undefined ? "待质检" : count}${count === 0 ? ' <span class="risk">缺类</span>' : ""}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`;
      $("#analysis-groups").innerHTML = `<div class="table-scroll"><table class="data-table"><thead><tr><th>分组</th><th>训练</th><th>验证</th><th>测试</th></tr></thead><tbody>${Object.entries(data.groups).map(([group, values]) => `<tr><td>${esc(group)}</td>${splits.map((split) => `<td>${values[split] || 0}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
      const policy = data.split_policy === "provided" ? "保留上传数据的训练 / 验证 / 测试划分" : data.split_policy.startsWith("camera_") ? "按相机与采集日期分组，相同图像合并分组" : "按文件名前缀分组，固定种子划分，需核对采集记录";
      $("#analysis-provenance").innerHTML = `<p class="body-copy">来源格式：${esc(data.source_format)}<br>划分策略：${policy}<br>图像划分：${splits.map((split) => `${labels[split]} ${data.splits[split]}`).join(" · ")}</p><p class="caption">分辨率：${Object.entries(data.resolutions).map(([size, count]) => `${esc(size)}（${count} 张）`).join("、")}</p><p class="caption fingerprint">数据指纹：${esc(data.fingerprint)}</p>`;
    } catch (error) { $("#class-distribution").textContent = error.message; }
    finally { analysisBusy = false; }
  }
  async function maintenance() {
    const data = await api("/api/retention"), form = $("#retention-form");
    for (const [key, value] of Object.entries(data.policy)) if (form.elements[key]) {
      if (key === "enabled") form.elements[key].checked = value; else form.elements[key].value = value;
    }
    $("#retention-plan").innerHTML = `<p class="body-copy">候选 ${data.candidates.length} 个目录 · ${number(data.reclaimable_bytes/1024**2, 1)} MB<br>受保护 ${data.protected.length} 个目录</p>${data.candidates.map((row) => `<div class="check-row"><strong>${esc(row.path)}</strong><span>${number(row.bytes/1024**2,1)} MB</span></div>`).join("") || '<p class="caption">当前无符合策略的候选目录。</p>'}<details><summary>查看引用保护</summary>${data.protected.map((row) => `<p class="caption">${esc(row.path)} · ${esc(row.protected)}</p>`).join("")}</details>`;
    $("#retention-trash").innerHTML = data.trash.map((row) => `<div class="check-row"><div><strong>${esc(row.path)}</strong><p>${new Date(row.moved_at*1000).toLocaleString("zh-CN")} · ${number(row.bytes/1024**2,1)} MB</p></div><button class="button secondary" data-restore-run="${esc(row.id)}">恢复</button></div>`).join("") || '<p class="caption">回收区为空。</p>';
  }
  function videoControls() {
    const models = state.jobs.filter((job) => job.kind === "training" && job.status === "completed" && job.config?.backend === "yolo" && job.config?.data === state.datasetId);
    const select = $("#video-model"), old = select.value;
    const options = '<option value="">等待该数据集模型归档完成</option>' + models.map((job) => `<option value="trained:${job.id}">${esc(job.name)}</option>`).join("");
    if (select.innerHTML !== options) { select.innerHTML = options; select.value = old; if (!select.value && models.length) select.value = `trained:${models[0].id}`; }
    const local = $("#video-device").value === "local";
    $("#video-submit").textContent = local ? "本机 CPU 检测" : "远程 GPU 检测";
    $("#video-submit").disabled = !videoUpload || !select.value || (!local && !state.overview?.hardware?.cuda_available) || state.jobs.some(active);
    if (local) $("#video-feedback").textContent = "使用本机 CPU 检测，无需上传权重；视频处理可能较慢。";
    else if (!state.overview?.hardware?.cuda_available) $("#video-feedback").textContent = "等待远程 GPU 连接。可先上传视频或查看已有结果。";
    else $("#video-feedback").textContent = "远程 GPU 已就绪，上传视频后即可检测。";
    const jobs = state.jobs.filter((job) => job.media_type === "video" && job.config?.data === state.datasetId), task = $("#video-job"), value = task.value;
    const tasks = '<option value="">请选择视频任务</option>' + jobs.map((job) => `<option value="${job.id}">${esc(job.name)} · ${esc(statusNames[job.status])}</option>`).join("");
    if (task.innerHTML !== tasks) { task.innerHTML = tasks; task.value = jobs.some((job) => job.id === value) ? value : jobs[0]?.id || ""; }
  }
  async function videoResult() {
    if (videoLoading) return; const id = $("#video-job").value; if (!id) return;
    const job = state.jobs.find((value) => value.id === id); if (!job) return;
    videoLoading = true;
    try {
      $("#video-job-status").textContent = `${statusNames[job.status]} · ${phaseNames[job.phase] || job.phase}${job.processed_frames ? ` · ${job.processed_frames}/${job.total_frames} 帧` : ""}${job.error ? " · " + job.error : ""}`;
      $("#video-explain").disabled = job.status !== "completed";
      if (job.status !== "completed") return;
      if (loadedVideo !== id) {
        const files = await api(`/api/jobs/${id}/artifacts`); if (id !== $("#video-job").value) return;
        const video = files.find((file) => file.path === "run/annotated.mp4"), summary = files.find((file) => file.path === "run/video_summary.json");
        $("#video-player").hidden = !video; if (video) $("#video-player").src = video.url;
        $("#video-downloads").innerHTML = files.filter((file) => ["run/annotated.mp4", "run/frames.jsonl", "run/video_summary.json"].includes(file.path)).map((file) => `<a class="text-link" href="${artifactUrl(file.url)}?download=true" download>${esc(file.name)} ↓</a>`).join("");
        if (summary) {
          const value = await api(summary.url);
          $("#video-summary").innerHTML = `<p class="caption">${value.frames} 帧 · ${number(value.duration_seconds,1)} 秒</p><div class="table-scroll"><table class="data-table"><thead><tr><th>类别</th><th>出现帧数</th><th>同帧最多</th><th>最高置信度</th></tr></thead><tbody>${Object.entries(value.class_summary).map(([name, row]) => `<tr><td>${esc(name)}</td><td>${row.frame_occurrences}</td><td>${row.max_simultaneous}</td><td>${percent(row.max_confidence)}</td></tr>`).join("")}</tbody></table></div>`;
        } loadedVideo = id;
      }
      const explanation = await api(`/api/jobs/${id}/explanation`); if (id === $("#video-job").value) renderAssistantText($("#video-explanation"), explanation.answer || explanation.error || (explanation.status === "running" ? "DeepSeek 正在解释…" : "可点击生成视频解释。"));
    } catch (error) { $("#video-job-status").textContent = error.message; }
    finally { videoLoading = false; }
  }
  $("#video-upload").addEventListener("change", async (event) => {
    const file = event.target.files[0]; if (!file) return;
    videoUpload = ""; videoControls();
    if (file.size > 200*1024**2) return toast("视频最大 200 MB", true);
    const form = new FormData(); form.append("file", file); $("#video-upload-track").hidden = false;
    try {
      const value = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/uploads/video"); xhr.setRequestHeader("X-CSRFToken", csrfToken());
        xhr.upload.onprogress = (item) => { if (item.lengthComputable) { $("#video-upload-bar").style.width = `${item.loaded/item.total*100}%`; $("#video-upload-note").textContent = `上传 ${Math.round(item.loaded/item.total*100)}%`; } };
        xhr.onerror = () => reject(new Error("视频上传连接中断"));
        xhr.onload = () => { try { const data = JSON.parse(xhr.responseText); if (xhr.status >= 400) reject(new Error(data.detail)); else resolve(data); } catch (_) { reject(new Error("视频上传响应异常")); } };
        xhr.send(form);
      }); videoUpload = value.id; $("#video-upload-note").textContent = `${file.name} · ${value.width}×${value.height} · ${number(value.duration_seconds,1)} 秒 · ${value.frames} 帧`;
    } catch (error) { $("#video-upload-note").textContent = error.message; toast(error.message, true); }
    finally { $("#video-upload-track").hidden = true; videoControls(); }
  });
  $("#video-model").addEventListener("change", videoControls);
  $("#video-device").addEventListener("change", videoControls);
  $("#video-submit").addEventListener("click", async () => {
    $("#video-submit").disabled = true;
    try {
      const local = $("#video-device").value === "local";
      const value = await jsonPost("/api/jobs", { kind: "inference", name: `${state.overview.dataset.name} · 视频目标检测`, upload_id: videoUpload, media_type: "video", model_id: $("#video-model").value, execution: local ? "local" : "remote", config: { threshold: Number($("#video-threshold").value) } });
      await refresh(); $("#video-job").value = value.id; await videoResult(); toast(local ? "视频任务已提交本机 CPU" : "视频任务已提交远程 GPU");
    } catch (error) { toast(error.message, true); } finally { videoControls(); }
  });
  $("#video-job").addEventListener("change", () => { loadedVideo = ""; $("#video-player").hidden = true; $("#video-summary").replaceChildren(); $("#video-downloads").replaceChildren(); $("#video-explanation").textContent = "等待所选任务解释"; videoResult(); });
  $("#video-explain").addEventListener("click", async () => {
    try { await jsonPost(`/api/jobs/${$("#video-job").value}/explanation`, {}); $("#video-explanation").textContent = "DeepSeek 正在解释…"; } catch (error) { toast(error.message, true); }
  });
  $("#analysis-dataset").addEventListener("change", async (event) => { analysisId = ""; await selectDataset(event.target.value); await analysis(); });
  $("#retention-preview").addEventListener("click", () => maintenance().catch((error) => toast(error.message, true)));
  $("#retention-form").addEventListener("submit", async (event) => {
    event.preventDefault(); const form = event.target, values = {};
    for (const key of ["keep_days", "keep_latest", "max_gb", "trash_days"]) values[key] = Number(form.elements[key].value);
    values.enabled = form.elements.enabled.checked;
    try { await jsonPost("/api/retention", values); await maintenance(); $("#retention-feedback").textContent = "策略已保存。后台每天检查；可先核对预览。"; } catch (error) { toast(error.message, true); }
  });
  $("#retention-apply").addEventListener("click", async () => {
    try { const result = await jsonPost("/api/retention/apply", {}); $("#retention-feedback").textContent = result.disabled ? "策略尚未启用，未清理任何目录。" : `移入回收区 ${result.moved.length} 个，清理过期回收记录 ${result.purged.length} 个。`; await maintenance(); } catch (error) { toast(error.message, true); }
  });
  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-restore-run]"); if (!button) return;
    try { await jsonPost(`/api/retention/restore/${button.dataset.restoreRun}`, {}); await maintenance(); toast("运行目录已恢复"); } catch (error) { toast(error.message, true); }
  });
  function routeExtra() {
    if (location.hash === "#understanding") { analysisId = ""; analysis(); }
    if (location.hash === "#maintenance") maintenance().catch((error) => toast(error.message, true));
    if (location.hash === "#video") { videoControls(); videoResult(); }
  }
  window.addEventListener("hashchange", routeExtra);
  window.addEventListener("odp:overview", () => {
    const data = state.overview.datasets || [], select = $("#analysis-dataset");
    select.innerHTML = data.map((row) => `<option value="${row.id}">${esc(row.name)}</option>`).join(""); select.value = state.datasetId;
    if (location.hash === "#understanding") analysis();
    if (location.hash === "#video") { videoControls(); videoResult(); }
  });
  phaseNames.encoding = "视频编码中";
  routeExtra();
})();
