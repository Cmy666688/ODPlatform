# 算法工作台

这是与正式监控页面共享后端的 Streamlit 调试入口。读取真实任务、指标和分割结果，
不在页面重跑时创建训练任务。

```powershell
pip install -e "./apps/web-backend[workbench]"
powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1 -Workbench
```

正式面板：<http://127.0.0.1:8000>；算法工作台：<http://127.0.0.1:8501>。

也可以单独启动，需先运行后端：

```powershell
cd apps/streamlit
python -m streamlit run app.py
```

默认后端为 `http://127.0.0.1:8000`，可通过 `ODP_API_URL` 指定。
配置表单必须明确提交，GPU 未就绪时按钮禁用；页面刷新只查询已存在任务。

数据集导入与格式说明使用正式面板的数据集管理页；本入口展示相同任务与产物，统一提交到远程 GPU，训练提交带 Django CSRF 校验。
