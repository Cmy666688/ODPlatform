# apps/web-backend — Django 数据集与训练工作台

采用 Django 5.2 的路由、视图、表单、ORM、数据库迁移、CSRF 与 Admin。
Vue 3 用于数据集管理交互，沿用现有绿色监控界面。Vue 生产文件随包提供，页面不依赖外部 CDN。
训练和推理继续由 `apps/platform` 引擎的独立进程执行。

## 启动

```powershell
pip install -e "./apps/platform[mixercseg,cpu]"
pip install -e "./apps/web-backend[workbench,test]"
powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1 -Workbench
```

`odp-web --port 8000` 自动迁移数据库、收集静态文件并启动 Django ASGI 服务。
面板：<http://127.0.0.1:8000/#datasets>；Streamlit：<http://127.0.0.1:8501>。
启动脚本要求端口上的健康接口返回 `framework: django`，避免把旧服务误认为迁移完成。
服务绑定本机回环地址，本版本定位为本机/实验室工作台。

## 添加数据集并训练

1. 在“数据集管理”填写名称，选择压缩包或文件夹。
2. 上传后由后台进程完成解包、格式转换和 8 项数据质检。
3. 使用“预览”核对原图与掩膜，使用“质检”查看错误及风险提示。
4. 就绪后点击“创建训练”，选择数据集、计算设备、轮次、批大小、学习率与验收目标。
5. 在监控概览查看进度；正常训练结束后自动用 best 权重执行独立 test 验收。
6. 在“结果可视化”选择对应实验，查看指标、掩膜对比并下载曲线、CSV、报告与权重。

当前支持 **MixerCSeg 二值语义分割**。数据必须包含同名图像与二值掩膜，
掩膜值为 0/1 或 0/255，不能直接把 YOLO TXT、分类图片或多类别彩色标注当作二值分割数据。
图像支持 PNG、JPG/JPEG、BMP、TIF/TIFF。上传时会归一化为 RGB 图像和单通道二值 PNG。

支持 ZIP、TAR、TAR.GZ、TGZ，以及浏览器文件夹上传。单次总大小最多 512 MB；
压缩包解压最多 2 GB、20000 个文件，单文件最多 64 MB、1600 万像素。
越界路径、链接、加密 ZIP、重名路径和不完整图像/标注配对会被拒绝。
上传内容中的 YAML、脚本等不会被当作配置执行。

支持三种已有划分布局：

```text
images/{train,val,test}/ + masks/{train,val,test}/
train_img/ + train_lab/，val_img/ + val_lab/，test_img/ + test_lab/
{train,val,test}/images/ + {train,val,test}/masks/
```

也支持未划分布局 `images/ + masks/`。按文件名前缀分组，种子 42，
约 70% / 15% / 15% 分给 train/val/test，至少需要 3 个组。
文件名尾部 ` (数字)` 被视为可能同源的裁剪编号；这是一条分组线索，不能代替原始采集记录。
已有划分原样保留，跨集合重复图像为 ERROR，同前缀跨集合为 WARNING。
页面提供含 9 对合成样本的模板 ZIP，仅用于验证流程。

训练按钮使用 **Django 服务所在机器** 的设备。本机 CUDA 不可用时 GPU 训练按钮禁用；
可显式选择“CPU 验证”，需要安装 platform 的 `cpu` extra，建议使用少量轮次检查流程。
CPU 不会在未选择时替代 GPU。远程 GPU 的既有实验仍可展示同步状态，
本次数据集管理不包含新建 SSH 连接或页面一键提交远程训练的功能。

## 数据和实验追溯

- SQLite 台账：`.odp-meta/web/platform.sqlite3`，记录导入来源、状态、统计及质检报告。
- 上传原件和导入日志：`.odp-meta/dataset_uploads/<id>/`。
- 标准化数据：`data/processed/ds_<id>/`；运行时配置：`apps/platform/configs/datasets/ds_<id>.yaml`。
- 契约：`runs/data_pipeline/<run_id>/semantic_manifest.json`，包含图像、掩膜和划分身份。
- 训练与验收：`runs/training/`、`runs/web_jobs/`，保留原有完整训练记录。

训练提交前重做质检，ERROR 阻止开训。每批更新进度，每轮保存 best/last，
按 val mIoU 选 best，test 不参与选模，固定概率阈值并按原图尺寸统计指标。
未填写业务 F1 目标时标为“待人工评审”。

同一时刻执行一个训练/推理任务；关闭页面不主动停止进程。
停止采用协作信号，保留完成轮次的权重，不承诺断点续训。
远程任务以 `execution: remote` 标记，其 PID 不会被本机进程表误判；同步完成后才显示完成。

## Django 管理命令

```powershell
python apps/web-backend/manage.py check
python apps/web-backend/manage.py migrate
python apps/web-backend/manage.py createsuperuser
```

Admin：<http://127.0.0.1:8000/admin/>。管理员由使用者自行创建；没有内置默认账号。
后台展示数据集台账，导入与质检通过业务页面操作。

## API

| 接口 | 用途 |
|---|---|
| GET /api/health | Django 标识与本机设备 |
| GET /api/overview?dataset=<id> | 所选数据、质检、硬件和实验 |
| GET/POST /api/datasets | 列表 / multipart 导入 |
| GET /api/datasets/template | 下载合成样本 ZIP |
| GET /api/datasets/<id> | 导入状态与统计 |
| GET /api/datasets/<id>/previews | 原图与掩膜预览 |
| POST /api/datasets/<id>/validate | 重做质检 |
| GET /api/config | 默认训练配方和参数 Schema |
| GET/POST /api/jobs | 实验列表 / 创建任务 |
| POST /api/jobs/<id>/cancel | 协作停止 |
| POST /api/uploads | 单图推理上传 |
| GET /api/jobs/<id>/artifacts | 验收报告、曲线、图片和权重 |

写接口需要 Django CSRF cookie 和 `X-CSRFToken`；浏览器端及 Streamlit 已接入。

## 验证与参考

```powershell
python scripts/check_project.py
python -m build apps/web-backend
```

验收覆盖 ZIP、TAR.GZ、文件夹、已有划分、错误标注、路径穿越、链接、大小限制及 CSRF。
已通过真实 HTTP 上传 9 对合成样本，完成 1 轮 CPU 训练和独立测试，生成曲线、掩膜及对比图；
该记录仅验证功能链路，不代表业务精度。此前 CrackMap 50 轮实验见
[完整训练报告](../../docs/crackmap-training-20261026.md)。

实现参考 [Django 文件上传](https://docs.djangoproject.com/en/5.2/topics/http/file-uploads/)、
[Django CSRF](https://docs.djangoproject.com/en/5.2/howto/csrf/) 与
[Vue 无构建工具接入](https://vuejs.org/guide/quick-start.html#using-vue-from-cdn)。
