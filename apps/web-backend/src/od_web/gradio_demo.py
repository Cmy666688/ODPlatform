"""D7 Gradio 演示端,复用 Django 数据、训练与推理接口。"""
import argparse
from io import BytesIO
import os
from pathlib import Path

import httpx
from PIL import Image

from od_platform.common import paths

API_URL = os.environ.get("ODP_API_URL", "http://127.0.0.1:8000")


def request(path, *, document=None, files=None, data=None, timeout=60):
    with httpx.Client(base_url=API_URL, timeout=timeout, trust_env=False) as client:
        if document is not None or files is not None or data is not None:
            client.get("/").raise_for_status()
            response = client.post(path, json=document, files=files, data=data, headers={"X-CSRFToken": client.cookies.get("csrftoken", "")})
        else: response = client.get(path)
        if response.is_error:
            try: detail = response.json().get("detail", "请求失败")
            except ValueError: detail = "服务暂时不可用"
            raise ValueError(str(detail))
        return response.json()


def submit(file, model, threshold, execution="remote"):
    if not file or not model: raise ValueError("请选择文件和已训练模型")
    video = Path(file).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    with open(file, "rb") as stream:
        upload = request("/api/uploads/video" if video else "/api/uploads", files={"file": (Path(file).name, stream)})
    job = request("/api/jobs", document={"kind": "inference", "name": "Gradio · 视频目标检测" if video else "Gradio · 图片目标检测",
        "media_type": "video" if video else "image", "upload_id": upload["id"], "model_id": model, "execution": execution, "config": {"threshold": threshold}})
    return job["id"], job


def upload_dataset(file, name):
    if not file or not name or not name.strip(): raise ValueError("请填写数据集名称并选择压缩包")
    with open(file, "rb") as stream:
        result = request("/api/datasets", data={"name": name.strip(), "mode": "archive"},
                         files={"files": (Path(file).name, stream)}, timeout=3600)
    return result["id"], result


def dataset_status(identifier):
    if not identifier: return {"status": "等待导入"}
    result = request("/api/datasets/" + identifier)
    return {key: result.get(key) for key in ("id", "name", "status", "phase", "progress", "error", "trainable", "splits")}


def train(dataset, architecture, epochs, batch, imgsz):
    if not dataset: raise ValueError("请选择已就绪的数据集")
    job = request("/api/jobs", document={"kind": "training", "name": "Gradio · YOLO 训练", "execution": "remote",
        "require_cuda": True, "config": {"backend": "yolo", "data": dataset, "architecture": architecture,
        "device": "cuda", "epochs": int(epochs), "batch_size_train": int(batch), "imgsz": int(imgsz)}})
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
        gr.Markdown("# 港区水域通航目标检测\nD7 演示工作台 · 数据集导入、远程 GPU 训练和图片/视频检测。")
        with gr.Accordion("导入目标检测数据集", open=False):
            gr.Markdown("支持 YOLO / VOC 检测数据压缩包（ZIP、TAR、TAR.GZ、TGZ），最大 4 GB。上传后后台转换和质检；完成后刷新数据集。")
            dataset_name = gr.Textbox(label="数据集名称", placeholder="例如：港区水域通航目标数据集")
            archive = gr.File(label="数据集压缩包", file_types=[".zip", ".tar", ".gz", ".tgz"], type="filepath")
            upload_button = gr.Button("上传并导入数据集")
            dataset_id = gr.Textbox(label="导入任务数据集 ID")
            import_status = gr.JSON(label="导入和质检状态")
            import_check = gr.Button("刷新导入状态")
        with gr.Row():
            dataset = gr.Dropdown(label="数据集", choices=[])
            model = gr.Dropdown(label="已完成训练的模型", choices=[])
            reload = gr.Button("刷新数据与模型")
        with gr.Accordion("远程 GPU 训练", open=False):
            gr.Markdown("选择已就绪的数据集；训练在远程 GPU 上执行，使用下方任务状态查看进度。")
            with gr.Row():
                architecture = gr.Dropdown(label="模型", choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], value="yolov8n.pt")
                epochs = gr.Number(label="训练轮次", value=50, precision=0, minimum=1, maximum=1000)
                batch = gr.Number(label="批次大小", value=16, precision=0, minimum=1, maximum=64)
                imgsz = gr.Number(label="图像尺寸", value=640, precision=0, minimum=320, maximum=1280)
            train_button = gr.Button("提交远程 GPU 训练", variant="primary")
        source = gr.File(label="图片 / 视频", file_types=[".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov", ".mkv"], type="filepath")
        threshold = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
        execution = gr.Radio(choices=[("本机 CPU", "local"), ("远程 GPU", "remote")], value="local", label="检测运行位置")
        run = gr.Button("提交图片 / 视频检测", variant="primary")
        identifier = gr.Textbox(label="任务 ID", info="也可以粘贴已有任务 ID 查看结果")
        status = gr.JSON(label="真实任务状态")
        with gr.Row():
            image = gr.Image(label="检测框可视化", interactive=False)
            video = gr.Video(label="标注视频", interactive=False)
        explanation = gr.Textbox(label="DeepSeek 视频解释", lines=6, interactive=False)
        check = gr.Button("刷新结果"); explain = gr.Button("生成 / 重试视频解释")
        def datasets():
            rows = request("/api/datasets")
            rows = [row for row in rows if row.get("task") == "detect" and row.get("trainable")]
            return gr.Dropdown(choices=[(row["name"], row["id"]) for row in rows], value=rows[0]["id"] if rows else None)
        def models(data):
            rows = [row for row in request("/api/jobs") if row["kind"] == "training" and row["status"] == "completed" and row.get("config", {}).get("data") == data]
            return gr.Dropdown(choices=[(row["name"], "trained:" + row["id"]) for row in rows], value="trained:" + rows[0]["id"] if rows else None)
        demo.load(datasets, outputs=dataset); reload.click(datasets, outputs=dataset); dataset.change(models, inputs=dataset, outputs=model)
        upload_button.click(upload_dataset, inputs=[archive, dataset_name], outputs=[dataset_id, import_status])
        import_check.click(dataset_status, inputs=dataset_id, outputs=import_status)
        train_button.click(train, inputs=[dataset, architecture, epochs, batch, imgsz], outputs=[identifier, status])
        run.click(submit, inputs=[source, model, threshold, execution], outputs=[identifier, status])
        check.click(refresh, inputs=identifier, outputs=[status, image, video, explanation])
        explain.click(lambda job: request("/api/jobs/" + job + "/explanation", document={}), inputs=identifier, outputs=status)
        gr.Timer(5).tick(refresh, inputs=identifier, outputs=[status, image, video, explanation])
    return demo


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=7860); args = parser.parse_args()
    build().launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False, max_file_size="4gb", footer_links=[])


if __name__ == "__main__": main()
