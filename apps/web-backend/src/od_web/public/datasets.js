/* Vue 数据集管理。上传完成后由独立导入进程转换和质检。 */
"use strict";

const datasetManager = Vue.createApp({
  data() {
    return { rows: [], name: "", mode: "archive", files: [], busy: false, progress: 0,
      message: "", error: false, previews: [], previewName: "",
      labels: { ready: "已就绪", importing: "导入中", blocked: "质检需修复", failed: "导入失败" } };
  },
  computed: {
    selection() { return this.mode === "archive" ? this.files[0]?.name : `${this.files.length} 个文件 · ${(this.files.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024).toFixed(1)} MB`; }
  },
  methods: {
    switchMode(mode) { if (this.busy) return; this.mode = mode; this.files = []; this.message = ""; },
    choose(event) { this.files = [...event.target.files]; this.message = ""; },
    drop(event) {
      if (this.busy) return;
      if (this.mode !== "archive") { this.message = "文件夹请点击上传区域选择，以保留相对路径。"; return; }
      this.files = [...event.dataTransfer.files];
    },
    async load() { this.rows = await api("/api/datasets"); },
    async upload() {
      this.error = false; this.message = "";
      if (this.files.reduce((sum, file) => sum + file.size, 0) > 512 * 1024 * 1024) { this.error = true; this.message = "单次上传不能超过 512 MB。"; return; }
      const form = new FormData(); form.append("name", this.name.trim()); form.append("mode", this.mode);
      for (const file of this.files) form.append("files", file);
      if (this.mode === "folder") form.append("relative_paths", JSON.stringify(this.files.map((file) => file.webkitRelativePath)));
      this.busy = true; this.progress = 0;
      try {
        await new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/datasets"); xhr.setRequestHeader("X-CSRFToken", csrfToken());
          xhr.upload.onprogress = (event) => { if (event.lengthComputable) this.progress = Math.round(100 * event.loaded / event.total); };
          xhr.onerror = () => reject(new Error("上传连接中断，请重新选择文件后重试。"));
          xhr.onload = () => {
            let body; try { body = JSON.parse(xhr.responseText); } catch (_) { reject(new Error("上传响应异常，请检查文件大小与格式。")); return; }
            if (xhr.status >= 200 && xhr.status < 300) resolve(body);
            else reject(new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)));
          };
          xhr.send(form);
        });
        this.message = "上传完成，正在后台导入与质检。可在下方查看进度。"; this.files = []; this.name = ""; await this.load();
      } catch (error) { this.error = true; this.message = error.message; }
      finally { this.busy = false; }
    },
    async preview(item) {
      try { this.previews = await api(`/api/datasets/${item.id}/previews`); this.previewName = item.name; this.$refs.previewDialog.showModal(); }
      catch (error) { toast(error.message, true); }
    },
    async inspect(item) { await selectDataset(item.id); location.hash = "data"; },
    async train(item) { await selectDataset(item.id); openConfig(); }
  },
  mounted() {
    this.load().catch((error) => toast(error.message, true));
    window.addEventListener("odp:overview", (event) => { this.rows = event.detail.datasets; });
  }
});
datasetManager.config.compilerOptions.delimiters = ["[[", "]]"];
datasetManager.mount("#dataset-manager");
