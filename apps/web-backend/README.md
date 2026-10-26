# apps/web-backend — Django 数据集与训练工作台

当前默认数据与模型为 **WSODD / YOLOv8n 水面目标检测**，具体数据划分和完整训练配方见 [WSODD 实验说明](../../docs/wsodd-training.md)。网页现已支持多类别 VOC / YOLO 压缩包和文件夹上传。

采用 Django 5.2 的路由、视图、表单、ORM、数据库迁移、CSRF 与 Admin。
Vue 3 用于数据集管理交互，沿用现有绿色监控界面。Vue 生产文件随包提供，页面不依赖外部 CDN。
训练、独立测试和图片推理由远程 CUDA GPU 上的 `apps/platform` 独立进程执行。
本机负责 Django 界面、数据导入与质检、结果归档；远程不可用时不回退到本机 CPU。

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

## 添加目标检测数据集并训练

网页上传入口采用多类别目标检测,不再把新数据集转换为二值掩膜。
支持 ZIP、TAR、TAR.GZ、TGZ 和浏览器文件夹上传。单次最大 4 GiB,后台解压最大 12 GiB、100000 个文件,
单文件最大 64 MiB、图像最大 1600 万像素。Django 将大文件落到临时文件,接收与解包均分块读取。
上传进度与后台解包、转换、质检进度分别显示。网页上传当前不提供断点续传。

- VOC: `JPEGImages/ + Annotations/`,也支持 WSODD 的 `image/ + annotation/` 和 `images/ + annotations/`。
  从 XML 的 object/name 生成固定且有序的类别表,检查标注尺寸与绝对像素 xyxy 框。
- YOLO: `images/ + labels/` 配合根目录 `data.yaml`（names 列表或从零连续编号的字典）或 `classes.txt`。
  TXT 每行仅接受 `class cx cy w h` 五列归一化坐标;空 TXT 是无目标样本,缺少 TXT 会阻断导入。
  分割多边形、掩膜与分类目录不会当作目标检测标注接收。
- 已划分数据支持 `images/{train,val,test}/ + labels/{train,val,test}/`、
  `{train,val,test}/images/ + labels/`,VOC 可提供 `ImageSets/Main/{train,val,test}.txt`。
  原划分保持不变,三个集合均须非空。当前不解析 YOLO 配置中的任意路径或图片列表。
- 未划分时,按相机采集日或去掉末尾帧号的文件名前缀分组;相同原图合并来源组。
  固定种子 42 搜索约 70/15/15 的组划分,至少三个组。推断分组须结合采集记录核实。

图像完整解码后转为 RGB JPEG（quality 95）;框不裁剪、不合并类别。Manifest 同时记录原件与派生文件摘要、
类别表、划分与转换策略。相同派生图像跨集合、同组跨集合、非法框、训练类别缺失为 ERROR;
验证或测试缺类为 WARNING,缺失类不伪造 AP。原始上传包保留。

选择数据集后,训练配方、类别数、监控和推理权重按当前数据集切换。训练与独立测试始终在远程 CUDA GPU 执行,
采用 YOLOv8n,完成后回传 mAP50、mAP50–95、逐类 CSV、PR 曲线、混淆矩阵、检测框对比图及 best/last 权重。
网页提供港区目标检测合成模板,仅验证流程,不代表业务数据或模型精度。
历史分割数据和实验保留,不列入当前目标检测数据集列表。

## 配置 DeepSeek 智能助手

参照仓库根目录 `.env.example`，复制为 `.env`，填写 `DEEPSEEK_API_KEY`。
`DEEPSEEK_BASE_URL=https://api.deepseek.com`，`DEEPSEEK_MODEL=deepseek-flash`；
也可选择 `deepseek-v4-pro`。调用格式依据 [DeepSeek 官方文档](https://api-docs.deepseek.com/)。
系统环境变量优先；服务按请求重新读取 `.env`，无需重启正在上传或训练的服务。
密钥留空时，网页“智能助手”显示待配置，不产生 API 请求；不会伪造模型回答。
网页仅返回配置状态，API Key 不进入返回内容、日志或 Git，也不会上传远程 GPU。
后端使用固定的 DeepSeek 官方 HTTPS 地址；不接受网页自定义代理地址。

助手发送当前会话、所选数据集的类别/划分/质检摘要、关联实验最近五轮指标及独立测试报告。
不发送原图、视频、权重、SSH 信息和训练日志。回答为纯文本，按 DeepSeek 账户实际调用计费。
当前完成的是数据与实验问答；视频检测后自动解释仍需接通视频任务与摘要，不能将此入口视为视频链路已验收。

## 配置远程 GPU

远端先准备好 CUDA PyTorch、Ultralytics 与 Web 工作进程依赖。YOLO 检测不要求扫描内核。历史 MixerCSeg 实验才需要相应依赖和 `selective_scan_cuda_oflex`。本地后端不要求 CUDA。
当前 GPU 现场使用 Ultralytics 8.4.183、NumPy 1.26.3、opencv-python 4.11.0.86、faster-coco-eval 1.8.0。
远端无法访问 GitHub 时,应预先缓存 YOLOv8n 训练初始权重、YOLO26n AMP 检查权重及 Ultralytics 配置目录中的 Arial.ttf;中文类别还需相应 Unicode 字体。使用已核验主机指纹的 known_hosts：

```powershell
python -m od_web.remote_setup --host GPU_HOST --port SSH_PORT --username root --root /absolute/ODPlatform-web --python /absolute/venv/bin/python --extra-pythonpath /absolute/kernel-source --known-hosts /absolute/known_hosts
```

首次在隐藏口令提示中输入已有 SSH 密码。安装专用 SSH 密钥后，口令不写入文件。
连接配置与私钥保存在 Git 忽略的 `.odp-meta/web/remote.json`、`remote_gpu_key`；
主机密钥必须已在 `known_hosts` 中，未知主机不会自动信任。这些本地文件不随仓库复制。
更新后运行 `python -m od_web.remote_setup --deploy-only` 同步源码；请在没有活动任务时更新。

新检测任务自动打包上传图像、检测标签、配置和 manifest，远端复核指纹与质检后开训。
推理自动上传所选权重和图片。训练与独立测试结束后，权重、报告、曲线、检测框和对比图回传本机，
同步完成才标记“已完成”。监控面板显示远程设备的实时资源数据，历史实验采样不冒充当前连接。

任务通过 SSH 在远端独立运行。连接中断时显示同步异常并自动重连，使用同一任务 ID 和启动标记
防止重复执行；本机同步进程意外退出后，访问实验列表会恢复同步。停止请求保存在本地，重连后补发。
远程大包通过连续 SSH 字节流传输,依据完整内容摘要接续临时文件;完成后在远端再次计算 SHA256,校验通过才解包。
这属于传输与监控重连，不会在远程训练进程自身崩溃后自动重训或断点续训。

## 数据和实验追溯

- SQLite 台账：`.odp-meta/web/platform.sqlite3`，记录导入来源、状态、统计及质检报告。
- 上传原件和导入日志：`.odp-meta/dataset_uploads/<id>/`。
- 标准化数据：`data/processed/ds_<id>/`；运行时配置：`apps/platform/configs/datasets/ds_<id>.yaml`。
- 检测契约：`data/processed/ds_<id>/manifest.json`，包含原件和派生图像/标签摘要、类别与划分身份。
- 训练与验收：`runs/training/`、`runs/web_jobs/`，保留原有完整训练记录。
- 检测质检历史：`runs/data_validation/<id>/history/`，每次报告独立快照，比较新增、修复、变化与移除的检查项。
  网页重新质检在独立进程运行，显示已检查样本数与总数；失败后可重试。比较粒度为检查项，不承诺逐个目标的问题追踪。

训练提交前重做质检，ERROR 阻止开训。每批更新进度，每轮保存 best/last，
按验证集 detection fitness 选 best，test 不参与选模，完整测试集报告检测 AP。
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
| GET /api/health | Django 标识与远程 GPU 连接、实时设备 |
| GET /api/overview?dataset=<id> | 所选数据、质检、硬件和实验 |
| GET/POST /api/datasets | 列表 / multipart 导入 |
| GET /api/datasets/template | 下载合成样本 ZIP |
| GET /api/datasets/<id> | 导入状态与统计 |
| GET /api/datasets/<id>/previews | 原图与类别目标框预览 |
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
已通过真实 HTTP 提交 9 对合成样本到远程 GPU，完成 1 轮训练和独立测试，再使用新权重推理并回传可视化；
该记录仅验证功能链路，不代表业务精度。此前 CrackMap 50 轮实验见
[完整训练报告](../../docs/crackmap-training-20261026.md)。

实现参考 [Django 文件上传](https://docs.djangoproject.com/en/5.2/topics/http/file-uploads/)、
[Django CSRF](https://docs.djangoproject.com/en/5.2/howto/csrf/) 与
[Vue 无构建工具接入](https://vuejs.org/guide/quick-start.html#using-vue-from-cdn)。
