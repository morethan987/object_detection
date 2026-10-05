# 建筑裂缝目标检测：多架构对比训练与评测（作业二）

在统一的数据与软硬件基准下，训练并对比 6 个目标检测模型（YOLOv8s / YOLOv10s / YOLO11s / Faster R-CNN R50-FPN / RetinaNet R50-FPN / RT-DETR-L），覆盖单阶段 anchor-free、单阶段 anchor-based、两阶段、Transformer 端到端四种范式。

**作业要求与完整实施指南见 [`homework.md`](homework.md)**（§1–4 为作业原文，§5–17 为 handout：数据实况、预处理方案、训练基准、评测协议、预期结果、报告模板）。本 README 只讲工程：环境怎么搭、目录什么含义、流程怎么跑。

---

## 1. 环境搭建（uv）

本项目用 [uv](https://docs.astral.sh/uv/) 管理依赖：依赖只声明在 `pyproject.toml`，`uv sync` 按 `uv.lock` 精确复现环境。

```bash
# 1) 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 创建虚拟环境并安装全部依赖（.venv/ 就地生成）
uv sync

# 3) 之后一切命令通过 uv run 执行，自动使用项目环境
uv run python train/train_yolo.py --help
```

### 1.1 PyTorch 安装源（重要）

GPU 版 PyTorch 的 wheel 不发布在 PyPI，`pyproject.toml` 的 `[tool.uv.sources]` 已把 torch/torchvision 指向官方源。当前默认 **cu124**，适用于任何 CUDA 12.x 驱动（≥ 525）的 NVIDIA 显卡。

| 情形 | 操作 |
|---|---|
| 训练机有 NVIDIA 卡 | 先 `nvidia-smi` 确认驱动 ≥ 525，保持默认 cu124 即可 |
| 只想在无卡机器上跑通流程 | 把 `[tool.uv.sources]` 里 torch/torchvision 两行换成注释中的 `pytorch-cpu`，删除 `uv.lock` 后重新 `uv sync` |
| 需要更新版 CUDA | 把 index 名和 URL 中的 `cu124` 换成 `cu126` 等 |

> 注意：切换安装源后必须删除 `uv.lock` 再 sync，否则锁文件仍指向旧源。

### 1.2 预训练权重下载

ultralytics 首次训练会自动从 GitHub Releases 下载 `yolo11s.pt`、`rtdetr-l.pt` 等权重。国内网络不通时提前手动下载放入 `weights/`，或设置代理。COCO 预训练 → 微调的含义见 handout §7 与 README 第 4 节。

---

## 2. 目录结构

```
object_detection/
├── homework.md               # 作业要求（§1–4）+ 实施指南 handout（§5–17）
├── pyproject.toml            # uv 项目配置：依赖清单 + PyTorch 安装源
├── .python-version           # 固定解释器 3.11（uv 自动管理）
├── uv.lock                   # 精确锁文件，uv sync 后生成并提交
├── README.md                 # 本文件
│
├── configs/
│   └── models.yaml           # 六模型的统一训练基准（handout §10 的机器可读版，唯一超参来源）
│
├── tools/                    # 预处理与可视化脚本（按 handout §8 顺序执行）
│   ├── prepare_data.py       # ① 拉取原始数据集 → data/raw/，校验 3403 对文件
│   ├── mask2bbox.py          # ② 掩码→检测框（§8.1 核心：二值化>0 → 闭运算 → 连通域 → 过滤）
│   ├── split.py              # ③ 按“原图 ID”分组划分 7:1:2 → splits/（防泄漏，§8.3）
│   ├── export.py             # ④ 导出 YOLO txt + COCO json + crack.yaml（§8.4）
│   ├── analyze.py            # ⑤ 数据统计图：实例数/长宽比/目标尺寸分桶（§8.2）
│   └── render_compare.py     # ⑦ 硬样本并排渲染：GT + 6 模型网格（§13）
│
├── train/                    # 训练入口（两套框架，§10）
│   ├── train_yolo.py         # ultralytics：YOLOv8s / YOLOv10s / YOLO11s / RT-DETR-L
│   └── train_torchvision.py  # torchvision：Faster R-CNN / RetinaNet（手动换头 num_classes=2）
│
├── eval/                     # 统一评测（§11，公平性命门）
│   ├── eval_coco.py          # 全模型预测 → COCO json → 同一份 COCOeval（mAP50/50:95/APS-M-L + P/R）
│   ├── benchmark.py          # Params / GFLOPs / 时延（batch=1, 预热50+计时300, fp32&fp16）
│   ├── summarize.py          # 汇总 → results/summary.csv + speed-accuracy 散点图
│   └── hard_samples.txt      # 硬样本清单（人工挑选：小目标/重叠/反光低对比各≥4 张）
│
├── results/                  # 【进 git】最终交付素材：summary.csv、figures/（报告直接引用）
│
├── data/                     # 【gitignore】数据集
│   ├── raw/crackDataset/     #   prepare_data.py 拉取的原始仓库（jpg + mask png）
│   ├── interim/              #   mask2bbox 中间产物（逐图 bbox json + 目检图）
│   ├── yolo/{images,labels}/{train,val,test}/   # export.py 产出，ultralytics 用
│   ├── coco/annotations/     #   export.py 产出，torchvision 训练 + 统一评测用
│   └── crack.yaml            #   ultralytics 数据集配置（names: ['crack']）
│
├── splits/                   # 【gitignore】train/val/test.txt 清单（split.py 产出）
├── runs/                     # 【gitignore】训练输出：权重、args.yaml、results.csv、曲线
└── weights/                  # 【gitignore】手动放置的预训练权重
```

标注：**当前 `tools/ train/ eval/` 下的 `.py` 均为脚手架**——每个文件头的 docstring 写明了用法、输入/处理/输出契约与对应 handout 章节，按此实现即可；`【gitignore】` 目录由脚本在运行时生成，不入库。

---

## 3. 端到端工作流

```bash
# 拉数据
uv run python tools/prepare_data.py

# 掩码转框（先看 data/interim/spot_check/ 目检确认二值化方向没反）
uv run python tools/mask2bbox.py --src data/raw/crackDataset --out data/interim --min-area 64

# 分组划分（严禁按 tile 随机切，会泄漏）
uv run python tools/split.py --interim data/interim --out splits --seed 0

# 导出双格式
uv run python tools/export.py --interim data/interim --splits splits --out data

# 数据统计（报告第 2 章素材）
uv run python tools/analyze.py --coco data/coco/annotations/train.json --out results/figures

# 训练（先跑一个 yolo11s 打通全流程，再补其余 5 个）
uv run python train/train_yolo.py --config configs/models.yaml --models yolo11s
uv run python train/train_yolo.py --config configs/models.yaml --models yolov8s yolov10s rtdetr_l
uv run python train/train_torchvision.py --config configs/models.yaml --models fasterrcnn_r50_fpn retinanet_r50_fpn

# 统一评测 + 基准 + 汇总（对每个模型执行 eval_coco；随后汇总）
uv run python eval/eval_coco.py --gt data/coco/annotations/test.json --pred runs/yolo11s/predict_coco.json --out results/metrics/yolo11s.json
uv run python eval/benchmark.py --config configs/models.yaml --out results/benchmark.json
uv run python eval/summarize.py --metrics results/metrics --bench results/benchmark.json --out results/summary.csv

# 硬样本可视化（先人工填 eval/hard_samples.txt）
uv run python tools/render_compare.py --list eval/hard_samples.txt --runs runs --out results/figures/qualitative
```

训练产出的 `runs/<model>/`（args.yaml、results.csv、PR 曲线等）与 `results/` 汇总即为作业“模型日志 + 对比表格 + 可视化”的直接素材。

---

## 4. 设计说明

- **为什么两套框架并存**：ultralytics 一个包覆盖 YOLO 三代 + RT-DETR 且训练/日志/导出全自动化；Faster R-CNN 与 RetinaNet 无 ultralytics 实现，用 torchvision 官方实现。两框架训练机制差异如实写进报告（§10），但**指标一律转 COCO json 走同一份 COCOeval**（§11），公平性不受框架影响。
- **为什么所有超参收敛到 `configs/models.yaml`**：作业要求提交“完整的超参数配置文件”。单一来源既防命令行散落参数漏记，也直接作为提交物。
- **为什么 `data/ splits/ runs/ weights/` 不进 git**：数据集 ~2GB 级、训练产物可复现，入库只污染历史；`splits/*.txt` + `seed` 固定后完全可复现。`results/` 进 git——它是报告的最终素材。
- **下一步实现顺序**：按 handout §15 的 W1 计划——先 `prepare_data → mask2bbox → split → export` 四连，再实现 `train_yolo.py` 用 yolo11s 打通"训练→评测→汇总”最小闭环，之后其余模型只是批量重跑。
