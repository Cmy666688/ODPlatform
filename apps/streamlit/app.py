"""算法调试工作台:复用 Web API,不在页面进程启动训练。"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("ODP_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="ODPlatform · 算法工作台", page_icon=":material/monitoring:", layout="wide")
st.title("算法工作台")
st.caption("YOLOv8 · 多数据集水面目标检测、训练监控与独立测试")


@st.cache_resource
def client():
    return httpx.Client(base_url=API_URL, timeout=20, trust_env=False)


def post(path, document):
    connection = client()
    connection.get("/").raise_for_status()
    return connection.post(path, json=document, headers={"X-CSRFToken": connection.cookies.get("csrftoken", "")})


def get(path):
    response = client().get(path); response.raise_for_status()
    return response.json()


if "selected_job" not in st.session_state:
    st.session_state.selected_job = None

try:
    overview = get("/api/overview")
except httpx.HTTPError:
    st.error("后端尚未连接，请先启动 odp-web。", title="服务不可用")
    st.stop()

with st.sidebar:
    st.subheader("ODPlatform")
    st.caption("算法实验与效果检查")
    choices = {row["id"]: row["name"] for row in overview.get("datasets", [])} or {"wsodd": "WSODD"}
    dataset = st.selectbox("当前数据集", list(choices), format_func=lambda key: choices[key], key="active_dataset")
    st.link_button("打开监控面板", API_URL, icon=":material/open_in_new:", width="stretch")
    st.divider()
    st.caption("当前设备")
    st.text(overview["hardware"]["device"])
    if not overview["hardware"]["cuda_available"]:
        st.info("远程 GPU 尚未就绪。可查看既有实验；连接恢复后再提交新任务。")

if dataset != overview.get("dataset", {}).get("id", "wsodd"):
    try: overview = get("/api/overview?dataset=" + dataset)
    except httpx.HTTPError:
        st.error("数据集信息暂时不可用，请刷新后重试。"); st.stop()

view = st.segmented_control("工作视图", ["训练监控", "数据质量", "结果浏览"], default="训练监控")

if view == "训练监控":
    with st.expander("配置完整训练"):
        with st.form("new_experiment"):
            available = {row["id"]: row["name"] for row in overview.get("datasets", []) if row.get("trainable")}
            st.caption("训练数据集：" + choices[dataset])
            name = st.text_input("实验名称", choices[dataset] + " 完整实验", key="experiment_name_" + dataset)
            col1, col2, col3 = st.columns(3)
            epochs = col1.number_input("训练轮次", min_value=1, max_value=1000, value=50)
            batch = col2.number_input("批大小", min_value=1, max_value=64, value=16)
            lr = col3.number_input("学习率", min_value=0.000002, max_value=1.0, value=0.001, format="%.6f")
            submitted = st.form_submit_button("开始完整实验", type="primary", disabled=not overview["hardware"]["cuda_available"] or dataset not in available)
        if submitted:
            response = post("/api/jobs", {"kind": "training", "name": name,
                       "config": {"data": dataset, "device": "cuda", "epochs": epochs, "batch_size_train": batch, "lr": lr}, "require_cuda": True})
            if response.is_success:
                st.session_state.selected_job = response.json()["id"]
                st.success("任务已提交，训练完成后自动生成独立测试报告。")
            else:
                st.error(str(response.json().get("detail", "提交失败")))

    @st.fragment(run_every="4s")
    def monitor():
        try: jobs = get("/api/jobs")
        except httpx.HTTPError:
            st.warning("监控连接暂时中断，稍后自动重试。"); return
        jobs = [job for job in jobs if job["kind"] == "training" and job.get("config", {}).get("data") == dataset]
        if not jobs:
            st.info("尚无完整训练任务。GPU 就绪后可提交实验。")
            return
        mapping = {job["id"]: job for job in jobs}
        selected = st.selectbox("监控任务", list(mapping), format_func=lambda key: mapping[key]["name"], key="monitor_" + dataset)
        job = mapping[selected]; history = job.get("history", [])
        with st.container(horizontal=True):
            st.metric("完成轮次", f"{len(history)} / {job['config']['epochs']}", border=True)
            st.metric("训练损失", f"{history[-1]['train_loss']:.4f}" if history else "—", border=True)
            st.metric("验证 mAP50–95", f"{history[-1]['validation']['mAP50_95']:.1%}" if history else "—", border=True)
        st.progress(min(1.0, max(0.0, job.get("progress", 0))), text=job["phase"])
        if history:
            frame = pd.DataFrame([{"Epoch": row["epoch"], "训练损失": row["train_loss"], "验证损失": row["validation"]["loss"]} for row in history])
            st.line_chart(frame, x="Epoch", y=["训练损失", "验证损失"], color=["#147d68", "#dfad57"], alt="逐轮训练与验证损失")
        if job.get("error"): st.error(job["error"])
    monitor()
elif view == "数据质量":
    quality = overview.get("quality")
    if quality:
        st.dataframe(pd.DataFrame([{"检查项": row["name"], "等级": row["severity"], "说明": row["summary"]} for row in quality["results"]]),
                     hide_index=True, alt=choices[dataset] + " 数据质量检查结果")
        st.caption("按当前数据契约核对划分来源；缺失类别的独立验收指标为空。")
    else: st.info("暂无质检报告，可在监控面板运行检查。")
else:
    try:
        jobs = [job for job in get("/api/jobs") if job["status"] == "completed" and job.get("config", {}).get("data") == dataset]
        options = {job["id"]: job["name"] for job in jobs}
        if not options:
            st.info("当前数据集训练或推理完成后在这里展示检测结果。"); st.stop()
        source = st.selectbox("结果来源", list(options), format_func=lambda key: options[key], key="results_" + dataset)
        if source == "reference":
            images = get("/api/reference")["images"]
            st.caption("这些结果来自官方权重，不代表本次完整训练效果。")
        else:
            images = [row for row in get(f"/api/jobs/{source}/artifacts") if "comparisons/" in row["path"]]
            job = next(job for job in jobs if job["id"] == source)
            if job.get("metrics"):
                with st.container(horizontal=True):
                    for key in ("mAP50", "mAP50_95", "precision", "recall"):
                        st.metric(key, f"{job['metrics'][key]:.1%}", border=True)
        if images:
            image = st.selectbox("图片", range(len(images)), format_func=lambda index: images[index]["name"])
            response = client().get(images[image]["url"]); response.raise_for_status()
            st.image(response.content, width="stretch", alt="原图、目标类别与检测框对比")
            st.download_button("下载对比图", response.content, file_name=images[image]["name"], mime="image/png")
        else: st.info("暂无结果图片。")
    except httpx.HTTPError:
        st.error("结果加载失败，请检查后端连接。")
