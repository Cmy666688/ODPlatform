"""D7 Gradio 演示端,所有模型任务复用 Django API 并在远程 GPU 执行。"""
import argparse
from io import BytesIO
import os
from pathlib import Path

import httpx
from PIL import Image

from od_platform.common import paths

API_URL = os.environ.get("ODP_API_URL", "http://127.0.0.1:8000")


def request(path, *, document=None, files=None):
    with httpx.Client(base_url=API_URL, timeout=60, trust_env=False) as client:
        if document is not None or files is not None:
            client.get("/").raise_for_status()
            response = client.post(path, json=document, files=files, headers={"X-CSRFToken": client.cookies.get("csrftoken", "")})
        else: response = client.get(path)
        if response.is_error:
            try: detail = response.json().get("detail", "请求失败")
            except ValueError: detail = "服务暂时不可用"
            raise ValueError(str(detail))
        return response.json()


def submit(file, model, threshold):
    if not file or not model: raise ValueError("请选择文件和已训练模型")
    video = Path(file).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    with open(file, "rb") as stream:
        upload = request("/api/uploads/video" if video else "/api/uploads", files={"file": (Path(file).name, stream)})
    job = request("/api/jobs", document={"kind": "inference", "name": "Gradio · 视频目标检测" if video else "Gradio · 图片目标检测",
        "media_type": "video" if video else "image", "upload_id": upload["id"], "model_id": model, "config": {"threshold": threshold}})
    return job["id"], job


def refresh(identifier):
    if not identifier: return {"status": "等待任务"}, None, None, ""
    state = request("/api/jobs/" + identifier); entries = request("/api/jobs/" + identifier + "/artifacts")
    image = None; video = None; explanation = ""
    for entry in entries:
        if image is None and "comparisons/" in entry["path"] and entry["path"].endswith(".png"):
            response = httpx.get(API_URL + entry["url"], timeout=30, trust_env=False); response.raise_for_status()
            image = Image.open(BytesIO(response.content)).copy()
        if entry["path"] == "run/annotated.mp4":
            video = paths.META_DIR / "gradio-results" / identifier / "annotated.mp4"; video.parent.mkdir(parents=True, exist_ok=True)
            if not video.exists():
                temporary = video.with_suffix(".download")
                with httpx.stream("GET", API_URL + entry["url"], timeout=60, trust_env=False) as response, temporary.open("wb") as stream:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(): stream.write(chunk)
                temporary.replace(video)
    if state.get("media_type") == "video":
        value = request("/api/jobs/" + identifier + "/explanation")
        explanation = value.get("answer") or value.get("error") or value["status"]
    status = {key: state.get(key) for key in ("id", "status", "phase", "progress", "epoch", "processed_frames", "total_frames", "metrics", "connection_status", "sync_error", "error") if state.get(key) is not None}
    return status, image, str(video) if video else None, explanation


def build():
    import gradio as gr
    with gr.Blocks(title="ODPlatform · D7 演示") as demo:
        gr.Markdown("# 港区水域通航目标检测\nD7 演示工作台 · 图片/视频推理在远程 CUDA GPU 执行，复用 Django 数据与实验。")
        with gr.Row():
            dataset = gr.Dropdown(label="数据集", choices=[])
            model = gr.Dropdown(label="已完成训练的模型", choices=[])
            reload = gr.Button("刷新数据与模型")
        source = gr.File(label="图片 / 视频", file_types=[".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov", ".mkv"], type="filepath")
        threshold = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
        run = gr.Button("提交远程 GPU 推理", variant="primary")
        identifier = gr.Textbox(label="任务 ID", info="也可以粘贴已有任务 ID 查看结果")
        status = gr.JSON(label="真实任务状态")
        with gr.Row():
            image = gr.Image(label="检测框可视化", interactive=False)
            video = gr.Video(label="标注视频", interactive=False)
        explanation = gr.Textbox(label="DeepSeek 视频解释", lines=6, interactive=False)
        check = gr.Button("刷新结果"); explain = gr.Button("生成 / 重试视频解释")
        def datasets():
            rows = request("/api/datasets")
            return gr.Dropdown(choices=[(row["name"], row["id"]) for row in rows], value=rows[0]["id"] if rows else None)
        def models(data):
            rows = [row for row in request("/api/jobs") if row["kind"] == "training" and row["status"] == "completed" and row.get("config", {}).get("data") == data]
            return gr.Dropdown(choices=[(row["name"], "trained:" + row["id"]) for row in rows], value="trained:" + rows[0]["id"] if rows else None)
        demo.load(datasets, outputs=dataset); reload.click(datasets, outputs=dataset); dataset.change(models, inputs=dataset, outputs=model)
        run.click(submit, inputs=[source, model, threshold], outputs=[identifier, status])
        check.click(refresh, inputs=identifier, outputs=[status, image, video, explanation])
        explain.click(lambda job: request("/api/jobs/" + job + "/explanation", document={}), inputs=identifier, outputs=status)
        gr.Timer(5).tick(refresh, inputs=identifier, outputs=[status, image, video, explanation])
    return demo


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=7860); args = parser.parse_args()
    build().launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False, max_file_size="200mb", footer_links=[])


if __name__ == "__main__": main()
