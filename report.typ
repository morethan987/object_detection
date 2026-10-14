#import "@preview/ori:0.2.5": *

#let font = (
  main: "Source Han Serif",
  mono: "Maple Mono NF",
  cjk: "Noto Serif CJK SC",
  emph-cjk: "KaiTi",
  math: "New Computer Modern Math",
  math-cjk: "Noto Serif CJK SC",
)

#set heading(numbering: numbly("{1:一}、", default: "1.1  "), supplement: none)
#set math.equation(numbering: "(1)")

#show: ori.with(
  title: "建筑裂缝目标检测",
  author: "Morethan987",
  subject: "经典目标检测算法实践",
  semester: "2026 秋",
  date: datetime.today(),
  media: "print",
  font: font,
  theme: "light",
  maketitle: true,
  makeoutline: true,
)

= 引言

== 任务背景

建筑裂缝是结构安全巡检的核心观测对象。相比通用目标检测，裂缝检测有三个鲜明的领域特性：*单类别*（只有 crack 一类）、*极度细长*（长宽比中位数超过 3）、*低对比度*（灰度路面背景与裂缝纹理相近，且存在强反光干扰）。这些特性使得"通用检测器 + 直接微调"并不必然奏效，不同范式（两阶段、单阶段有锚框、单阶段无锚框、Transformer 端到端）在框回归精度、密集目标处理与推理速度上的差异会被放大。

本报告在*统一的数据集、统一的软硬件基准、统一的超参预算*下，完整训练并对比 6 个检测模型——YOLOv8s、YOLOv10s、YOLO11s、Faster R-CNN R50-FPN、RetinaNet R50-FPN、RT-DETR-L——覆盖当前目标检测的四种主流技术范式，并以同一份 COCO 评测协议给出精度、参数量、计算量与推理时延的全量对比。

== 四种检测范式原理

*两阶段 anchor-based（Faster R-CNN）*#cite(<ren2015>)：第一阶段 RPN 在特征图上以多尺度、多长宽比的 anchor 回归出候选区域（objectness + 偏移量），第二阶段对候选做 RoI Align 池化后由独立头部完成分类与框精修。两阶段"先粗筛、后精修"的结构换取高定位精度，代价是串行流程带来的推理开销。

*单阶段 anchor-based（RetinaNet）*#cite(<lin2017>)：在 FPN 金字塔的每个位置直接密集回归 anchor 的类别与偏移，一步完成检测。密集预测带来极端的前景/背景类别失衡，RetinaNet 以 Focal Loss 解决：

$ "FL"(p_t) = -alpha_t (1-p_t)^gamma log p_t $

其中 $p_t$ 为模型对真实类别的置信度，$alpha=0.25$、$gamma=2$。易分样本的损失被 $(1-p_t)^gamma$ 压低，训练聚焦难样本。

*单阶段 anchor-free（YOLOv8s / YOLO11s）*#cite(<jocher2024>)：摒弃 anchor，每个位置直接预测中心点与框宽高，分类与回归头解耦；标签分配采用 TAL（Task-Aligned Assigner），框回归采用 DFL 分布式回归。YOLO11 相对 v8 将骨干中的 C2 模块替换为 C3k1 等改进，在更少参数量下取得更高精度。*NMS-free 变体（YOLOv10s）*#cite(<wang2024>)引入一致双分配：训练时同时使用 one-to-one 与 one-to-many 两组匹配头，推理时仅保留 one-to-one 头输出，从而*消除 NMS 后处理*，降低延迟并避免 NMS 超参敏感。

*Transformer 端到端（RT-DETR-L）*#cite(<zhao2024>)：将检测视为集合预测。模型输出固定数量（300 个）object queries，训练时以匈牙利二分图匹配将预测与真值一一对应后计算集合预测损失（分类 + $L_1$ + GIoU），推理直接输出去重后的集合，*天然无 NMS*。RT-DETR 采用 HGNetv2 骨干 + 高效混合编码器 + IoU 感知查询选择，是首个实时端到端检测器系列。

四种范式最终都以同一套匹配指标评价。给定预测框 $b$ 与真值框 $g$：

$ "IoU"(b, g) = (b ∩ g) / (b ∪ g) $

预测按置信度降序贪心匹配真值，IoU 达到阈值记 TP，否则记 FP；未匹配真值记 FN。于是：

$ P = "TP"/("TP"+"FP"), quad R = "TP"/("TP"+"FN"), quad F_1 = (2 P R)/(P + R) $

AP 为 P-R 曲线下面积（按召回等距采样 101 点近似），mAP\@0.5:0.95 为 IoU 阈值从 0.50 到 0.95（步长 0.05）共 10 档下的平均，对定位精度要求更苛刻。

== 本文工作

- 预处理层面：将 3403 对"图片 + 像素掩码"转换为检测框标注，并*按原图 ID 分组划分* train/val/test，杜绝同源 tile 与增广副本跨集合造成的数据泄漏；
- 训练层面：ultralytics 与 torchvision 两套框架、六个模型，在 imgsz/batch/epochs/seed 完全对齐的预算下训练；
- 评测层面：全部预测转为 COCO json，同一份 COCOeval（maxDet=300）+ 工程 P/R（conf=0.25）双口径，附参数量 / GFLOPs / 三种数值精度的延迟基准；
- 分析层面：PR 曲线、FP 归因（背景误检 / 定位差 / 重复框）、漏检按目标尺寸分桶、12 张硬样本三场景（密集 / 小目标 / 低对比）的定量与可视化对比，以及 Speed-Accuracy 权衡讨论。

= 数据集与预处理

== 数据集实况

数据集来自 GitHub 仓库 guoguolord/crackDataset#cite(<crackdataset>)，由三个公开子集合并而成，共 *3403 对「图片 + 像素级裂缝掩码」*，见@tab-subsets。

#figure(
  three-line-table[
    | 子目录 | 数量 | 掩码位置 | 样本尺寸 | 文件名规律 |
    | --- | --- | --- | --- | --- |
    | crack500 train+val | 1517 + 379 | label/ + val_label/ | 640×360（边缘 tile 偏小） | 原图名 + tile 偏移坐标 |
    | AsphaltCrack300 train | 795 | label/ | 504×672 | 原图序号 + tile 序号 |
    | CFD train+val | 566 + 146 | label/ + val_label/ | 480×320 | 原图名 + 增广后缀 |
  ],
  caption: [三个子集的构成（Crack500 出自 Yang 等#cite(<yang2020>)，CFD 出自 Shi 等#cite(<shi2016>)，AsphaltCrack300 为仓库作者自采）],
) <tab-subsets>

实测勘察得到三个决定预处理方案的关键事实：

+ *标注是像素级掩码而非检测框*。掩码 PNG 混有 1-bit 二值（像素值 0/1）、8-bit 0/255 与调色板三种编码，必须自行转 bbox；
+ *仓库里的"一张图"≠一个独立样本*。3403 个 tile 还原自约 918 张原图（Crack500 500 张、AsphaltCrack300 300 张、CFD 118 张），CFD 还存在 add/sub/adjust/翻转等*同源增广副本*（@fig-data-enhance 展示了同一原图的增广痕迹）。若按 tile 随机划分，同一原图的不同 tile 将跨集合泄漏，指标虚高；
+ *三个子集分辨率与拍摄条件差异大*（640×360 / 504×672 / 480×320），训练统一 letterbox 到 640，评测输入尺度对所有模型强制一致。

#figure(
  image("results/data_enhance.png", width: 88%),
  caption: [CFD 子集的增广痕迹：同一原图衍生出 add/sub（亮度增减）等副本，是"按原图分组划分"的直接证据（`tools/prepare_data.py` 自动生成）],
) <fig-data-enhance>

== 掩码转检测框（mask → bbox）

转换规则（`tools/mask2bbox.py`）：

+ 二值化统一用 `arr > 0`，兼容 0/1、0/255 与调色板索引三种编码（若写 `== 255` 会漏掉前两类）；
+ 随机抽 10 张掩码叠加原图人工目检，确认二值化方向未反（调色板掩码可能把裂缝画成深色）；
+ 形态学*闭运算*（3×3 核，1–2 次迭代）桥接裂缝断裂处；
+ `connectedComponentsWithStats` 取每个连通域的外接框；
+ 过滤面积小于 64 px² 的噪点连通域。

输出同时落盘 COCO $[x, y, w, h]$（绝对像素）与 YOLO 归一化 $[x_c, y_c, w, h]$ 双格式。经 min-area=64 过滤后，训练集 2383 张图*全部保有至少一个标注*（无空背景图）。

== 防泄漏的按原图分组划分

划分单位是"原图 ID"而非 tile：crack500 以正则取原图名，AsphaltCrack300 取原图序号，CFD 取原图名（add/sub/adjust/翻转变体归入同一组）。在三个子集内部分层、按 7:1:2 抽样、固定 `seed=0`，清单落盘 `splits/{train,val,test}.txt` 保证完全可复现。@tab-split 汇总了最终规模。

#figure(
  three-line-table[
    | 划分 | 图片数 | 占比 | 实例数 | 占比 |
    | --- | --- | --- | --- | --- |
    | train | 2383 | 70.1% | 3945 | 70.3% |
    | val | 339 | 10.0% | 569 | 10.1% |
    | test | 681 | 20.0% | 1096 | 19.5% |
    | 合计 | 3403 | 100% | 5610 | 100% |
  ],
  caption: [按原图分组 7:1:2 划分后的数据规模],
) <tab-split>

== 训练集统计

@fig-inst 至 @fig-size 给出训练集的三项统计。

#figure(
  image("results/figures/data_instances_per_image.png", width: 62%),
  caption: [每图实例数分布：均值 1.66、中位数 1、最多 9，绝大多数图只有 1–2 条裂缝],
) <fig-inst>

#figure(
  image("results/figures/data_aspect_ratio.png", width: 62%),
  caption: [框长宽比分布：均值 3.84、中位数 3.08、P90 高达 6.99——裂缝极度细长],
) <fig-ar>

#figure(
  image("results/figures/data_size_distribution.png", width: 62%),
  caption: [COCO 口径目标尺寸分桶：小目标（面积 \<32²）12.2%、中目标（32²–96²）23.9%、大目标（>96²）63.9%],
) <fig-size>

三条统计直接预示了后文实验结果的形态：

- *细长*：框沿长边稍有偏移，IoU 即跌破阈值——mAP\@0.5 与 mAP\@0.5:0.95 之间必然出现大落差；
- *大目标为主*：整体 mAP 主要由 AP_L 贡献，但 12.2% 的小目标是难点；
- *每图实例少*：密集场景（网状裂缝）稀缺，恰好是 @sec-hard 节中最考验模型的一类。

= 实验设置

== 硬件与软件环境

训练与全部延迟测试在*同一台 NVIDIA RTX 4090（24 GB）*云实例（AutoDL）上完成，保证时延可比。软件栈由 uv 按 `uv.lock` 精确锁定复现：Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0、ultralytics 8.4.172、pycocotools 2.0.11（完整版本表见#link(<app-a>)[附录 A]）。

== 统一训练基准

六个模型共用同一预算（`configs/models.yaml` 为唯一超参来源，全文见#link(<app-c>)[附录 C]）：`imgsz=640`、`batch=16`、`epochs=150`、`patience=50`、`seed=0`，使各模型梯度步数与数据通过次数同时相等（train 2383 张 → 约 149 iter/epoch → 约 22.3k 步）。学习率不跨优化器强行统一数值，各模型用其「优化器 × 框架」的规范默认值并在日志留档。早停判据两套框架统一为 fitness：

$ "fitness" = 0.1 dot "mAP@0.5" + 0.9 dot "mAP@0.5:0.95" $

#figure(
  three-line-table[
    | 模型 | 框架 | 预训练 | 优化器 / lr | 关键设置 |
    | --- | --- | --- | --- | --- |
    | YOLOv8s / v10s / 11s | ultralytics | COCO | 框架 auto（SGD，lr0=0.01） | mosaic=1.0，close_mosaic=10 |
    | RT-DETR-L | ultralytics | COCO | AdamW，1e-4 | *amp=false*；deterministic=false |
    | Faster R-CNN R50-FPN | torchvision | COCO | SGD 0.005, momentum 0.9 | 预测头 num_classes=2；AMP；cosine 退火 + 500 iter 热身 |
    | RetinaNet R50-FPN | torchvision | COCO | 同上 | Focal Loss α=0.25, γ=2；AMP |
  ],
  caption: [统一训练基准（imgsz=640、batch=16、epochs=150、patience=50、seed=0 为全体共享）],
) <tab-bench>

*训练异常如实记录*：RT-DETR-L 开启混合精度（AMP）训练时损失出现 NaN，禁用 AMP 后从头重训成功（代价是单 epoch 约 64 s，为 YOLO 系的 6–8 倍，150 epoch 全程约 2.7 h）；其余模型无 OOM、无 batch 降级，全部按 batch=16 跑完。`deterministic=false` 仅用于 RT-DETR（deformable attention 无确定性反向，官方建议关闭以提速）。

== 统一评测协议

公平性是本对比的命门，协议如下：

+ *学术口径*：所有模型预测转 COCO json，在*同一份 test.json*（681 图 / 1096 实例）上跑同一份 COCOeval（COCO 评测口径#cite(<lin2014>)）：score ≥ 0.05 起评、maxDets=300，输出 mAP\@0.5 / mAP\@0.5:0.95 / mAP\@0.75 与按尺寸的 AP_S/AP_M/AP_L。RT-DETR 推理上限即 300 queries，天然对齐；
+ *工程口径*：conf=0.25、IoU=0.5 贪心匹配下的 Precision / Recall / F1，反映部署时固定阈值的实际表现；
+ *参数量 / GFLOPs*：统一 640×640 输入。ultralytics 侧用 `model.info()`，torchvision 侧用 `thop.profile`；YOLOv10s 为 NMS-free 融合后口径；
+ *时延*：GPU、batch=1、预热 50 次、正式计时 300 次取均值±标准差，CUDA event + `torch.cuda.synchronize()` 掐表。FP32 / FP16 / BF16 三档：ultralytics 侧为融合算子后的静态半精度权重，torchvision 侧为 autocast 混合精度。计时*包含*模型前向与各自内置后处理（YOLO 的融合 NMS 算子、Faster R-CNN 的 RPN + RoI-NMS、RetinaNet 的 NMS、RT-DETR 无 NMS），*不含*读图与 letterbox 预处理。FPS = 1000 / latency；
+ 推理截断 conf=0.001、imgsz=640 对全体一致，评测阈值由 eval 脚本统一施加。

= 定量结果

== 精度总表

@tab-metrics 至 @tab-speed 为六模型在同一 test 集上的全量结果。

#figure(
  three-line-table[
    | 模型 | 范式 | mAP\@0.5 | mAP\@0.5:0.95 | mAP\@0.75 | AP_S | AP_M | AP_L |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | YOLOv8s | 单阶段 anchor-free | 51.63 | 29.53 | 29.30 | 0.84 | 8.61 | 41.69 |
    | YOLOv10s | 单阶段 NMS-free | 52.10 | 32.04 | *33.94* | 0.26 | 7.88 | 46.17 |
    | YOLO11s | 单阶段 anchor-free | 53.61 | 32.88 | 33.59 | 1.13 | 8.79 | 46.46 |
    | RT-DETR-L | Transformer 端到端 | 50.73 | 30.48 | 29.89 | 0.19 | 6.49 | 44.40 |
    | Faster R-CNN R50 | 两阶段 | *62.15* | *35.93* | *37.18* | *1.40* | *15.10* | *49.22* |
    | RetinaNet R50 | 单阶段 anchor-based | 56.78 | 31.07 | 29.85 | 1.04 | 12.60 | 42.97 |
  ],
  caption: [学术口径精度（COCOeval，score ≥ 0.05，maxDets=300，单位 %；加粗为每列最优）],
) <tab-metrics>

#figure(
  three-line-table[
    | 模型 | Precision | Recall | F1 | TP | FP | FN |
    | --- | --- | --- | --- | --- | --- | --- |
    | YOLOv8s | 65.91 | 50.27 | 57.04 | 551 | 285 | 545 |
    | YOLOv10s | 63.75 | 51.19 | 56.78 | 561 | 319 | 535 |
    | YOLO11s | *66.63* | 54.65 | *60.05* | 599 | 300 | 497 |
    | RT-DETR-L | 37.64 | 59.58 | 46.13 | 653 | 1082 | 443 |
    | Faster R-CNN R50 | 30.45 | *70.62* | 42.55 | 774 | 1768 | 322 |
    | RetinaNet R50 | 49.27 | 64.96 | 56.04 | 712 | 733 | 384 |
  ],
  caption: [工程口径（conf=0.25，IoU=0.5，单位 %；TP/FP/FN 为框计数）],
) <tab-eng>

#figure(
  three-line-table[
    | 模型 | Params / M | GFLOPs | 延迟 FP32 / ms | 延迟 FP16 / ms | FPS（FP16） |
    | --- | --- | --- | --- | --- | --- |
    | YOLOv8s | 11.14 | 28.64 | *2.715* | 2.820 | *354.6* |
    | YOLOv10s | 8.07 | 24.89 | 3.430 | 3.674 | 272.2 |
    | YOLO11s | 9.43 | *21.67* | 3.921 | 3.973 | 251.7 |
    | RT-DETR-L | 32.81 | 109.89 | 25.495 | 26.292 | 38.0 |
    | Faster R-CNN R50 | 41.30 | 90.71 | 11.456 | 10.461 | 95.6 |
    | RetinaNet R50 | 32.17 | 81.15 | 10.740 | 10.507 | 95.2 |
  ],
  caption: [效率基准（640×640，batch=1，预热 50 + 计时 300；FP16 为静态半精度（ultralytics）/ autocast（torchvision））],
) <tab-speed>

四点核心观察：

+ *Faster R-CNN 双料第一*（mAP\@0.5 62.15、mAP\@0.5:0.95 35.93），且 mAP\@0.75 达 37.18、领先第二名 3.2 个百分点——两阶段 RoI 框精修在细长目标上的高 IoU 优势兑现，与引言的范式分析一致；
+ *YOLO 三代按 11s > v10s > v8s 排序*（mAP\@0.5 53.61 / 52.10 / 51.63），代际进步成立；值得注意 v10s 的 mAP\@0.75（33.94）反超 11s，NMS-free 双分配对框质量有正向作用；
+ *RT-DETR-L 的 mAP\@0.5（50.73）垫底*，但@fig-curve-rtdetr 显示其 val 指标在 150 epoch 预算结束时仍在上升（最优出现在最后一个 epoch），属*预算内未收敛*而非架构上限；
+ *AP_S 全体崩塌*（0.19–1.40%），AP_L（41.7–49.2%）贡献了几乎全部精度——小目标在 640 输入下信息量不足，是数据与分辨率共同决定的系统性短板（详见 @sec-miss 节）。

工程口径（@tab-eng）呈现*与 mAP 排序相反*的格局：固定 conf=0.25 下，Faster R-CNN 召回最高（70.62%）但精确率最低（30.45%，1768 个 FP）——两阶段输出大量低置信度框越过 0.25 阈值；YOLO 系置信度校准好（P 64–67%）但召回偏低。F1 上 YOLO11s（60.05）反而是最佳工程折中。这说明*单点阈值的 P/R 对比高度依赖阈值选择与各模型分数分布*，选型时应看 P-R 曲线全貌（@fig-pr）而非固定阈值的两个数。

== 训练过程

@tab-epochs 汇总各模型实际训练时长与最优 epoch（fitness 判据），@fig-curve-yolo11s 至 @fig-curve-tv 为训练曲线。

#figure(
  three-line-table[
    | 模型 | 实际 epoch | 最优 epoch | 最优点 val mAP\@0.5 / 0.5:0.95 | 备注 |
    | --- | --- | --- | --- | --- |
    | YOLOv8s | 112（早停） | 62 | 57.44 / 36.13 | |
    | YOLOv10s | 103（早停） | 64 | 60.56 / 36.67 | |
    | YOLO11s | 144（早停） | 94 | 58.76 / 36.17 | |
    | RT-DETR-L | 150（跑满） | 150 | 52.43 / 29.48 | 最优在最后一 epoch，仍上升 |
    | Faster R-CNN R50 | 56（早停） | 6 | 59.01 / 34.57 | 收敛极快，此后 val 停滞 |
    | RetinaNet R50 | 98（早停） | 48 | 51.88 / 31.20 | |
  ],
  caption: [训练日志摘要（早停 = patience 50 触发；val mAP 为最优点取值）],
) <tab-epochs>

#figure(
  image("runs/yolo11s/results.png", width: 95%),
  caption: [YOLO11s 训练曲线：损失、P/R、mAP 随 epoch 变化（ultralytics 自动产出）],
) <fig-curve-yolo11s>

#figure(
  image("runs/yolov10s/results.png", width: 95%),
  caption: [YOLOv10s 训练曲线],
) <fig-curve-yolov10s>

#figure(
  image("runs/yolov8s/results.png", width: 95%),
  caption: [YOLOv8s 训练曲线],
) <fig-curve-yolov8s>

#figure(
  image("runs/rtdetr_l/results.png", width: 95%),
  caption: [RT-DETR-L 训练曲线（禁用 AMP 重训，150 epoch 跑满）],
) <fig-curve-rtdetr>

#figure(
  image("results/figures/train_curves_torchvision.png", width: 95%),
  caption: [torchvision 两模型训练曲线（左：训练损失；右：val mAP，虚线为 mAP\@0.5:0.95，竖点线标出最优 epoch）],
) <fig-curve-tv>

训练动态本身即是一个结论：Faster R-CNN 借 COCO 预训练的 RPN 与 RoI 头，*第 6 个 epoch 即达 val 峰值*，之后训练损失仍在降但 val fitness 五十个 epoch 无提升触发早停——典型的快速迁移 + 过拟合形态；RetinaNet（Focal Loss 冷启动更慢）在第 48 epoch 达峰；YOLO 系在 62–94 epoch 达峰。RT-DETR 是唯一跑满预算且终点仍上升的模型，若延长训练或修复 AMP，其名次存在上升空间。

== PR 曲线

@fig-pr 在完全相同的 COCOeval 口径（IoU=0.5，maxDets=300）下重绘六模型 P-R 曲线，图例中的 AP50 与@tab-metrics 完全一致。

#figure(
  image("results/figures/pr_curves.png", width: 60%),
  caption: [六模型统一口径 P-R 曲线（test 集，IoU=0.5，maxDets=300）],
) <fig-pr>

Faster R-CNN 曲线整体位于右上；RetinaNet 在中低召回段与 YOLO 系互有交叉、高召回段占优；YOLO 系在低召回（高置信度）段保持高精确率、但在召回 0.6 附近断崖；RT-DETR 曲线拖尾最长（高召回倾向，对应其工程口径 R 高 P 低）但整体包络面积最小。

== 与预注册假设的对照

实验前预注册的预期区间为 mAP\@0.5 0.80–0.94，实测 0.51–0.62，*系统性低于预期*。偏差的主因可归纳为三点：

+ *细长目标的 IoU 天花板*：每连通域一框的转框策略下，细长框沿长边轻微偏移即令 IoU 大幅下降——mAP\@0.5 与 mAP\@0.75 之间 18–27 个百分点的落差、以及 FP 归因中"定位差"占六成（@sec-fp 节）都是同一现象的投影；
+ *严格口径*：score ≥ 0.05 起评、无任何后处理过滤，低分 FP 全部计入惩罚；
+ *预期参照系偏乐观*：预注册区间参照了文献中更宽松的评测协议与更大的训练预算。

相对排序上，两条假设兑现：Faster R-CNN 凭框精修在高 IoU 段（mAP\@0.75）反超全部单阶段模型；NMS-free 架构（RT-DETR、YOLOv10s）在重叠密集场景确有检出优势（@sec-hard 节）。一条证伪：RT-DETR 的全局注意力并未带来小目标收益（AP_S 0.19，六模型最低）。

= 定性对比与误差分析

== 硬样本三场景量化对比 <sec-hard>

从 test 集固定 12 张硬样本（`eval/hard_samples.txt`），按场景分三组各 4 张：*密集 / 重叠 / 网状*、*细长 / 小目标*、*低反差 / 弱光照*。全体模型以 conf=0.25 推理，@tab-hard 给出各场景的召回与 FP 计数，@fig-q-dense 至 @fig-q-lowc 为对应可视化（GT + 六模型网格并排渲染）。

#figure(
  three-line-table[
    | 场景（GT 数） | YOLOv8s | YOLOv10s | YOLO11s | RT-DETR-L | Faster R-CNN | RetinaNet |
    | --- | --- | --- | --- | --- | --- | --- |
    | 密集 / 重叠（23） | 4.3%（FP 6） | 13.0%（4） | 13.0%（6） | *39.1%*（16） | *39.1%*（40） | 30.4%（11） |
    | 小目标（5） | 20.0%（3） | 20.0%（1） | 20.0%（1） | *60.0%*（28） | 40.0%（11） | 40.0%（5） |
    | 低对比（8） | 12.5%（4） | 75.0%（4） | 62.5%（2） | 62.5%（4） | *100%*（12） | 62.5%（14） |
  ],
  caption: [硬样本三场景召回（conf=0.25，IoU=0.5；括号内为 FP 框数；小目标组 GT 仅 5 个，统计效力有限）],
) <tab-hard>

#figure(
  image("results/figures/qualitative/compare_crack500_20160328_153616_1921_1081.png", width: 100%),
  caption: [密集 / 网状裂缝硬样本：GT 与六模型预测并排（`crack500_20160328_153616_1921_1081`）],
) <fig-q-dense>

#figure(
  image("results/figures/qualitative/compare_crack500_20160318_181602_641_721.png", width: 100%),
  caption: [细长 / 小目标硬样本（`crack500_20160318_181602_641_721`）],
) <fig-q-small>

#figure(
  image("results/figures/qualitative/compare_CFD_080.png", width: 100%),
  caption: [低对比 / 弱光照硬样本（`CFD_080`）],
) <fig-q-lowc>

结合@tab-hard 与可视化：

+ *密集重叠场景全体崩塌*（最高召回仅 39.1%），是六模型共同的最难场景。其中 NMS-free 的 RT-DETR 与两阶段 Faster R-CNN 并列最优，YOLO 系仅 4.3–13.0%——网状裂缝的多条分支在 anchor-free 密集预测下相互竞争，NMS 又会抑制 IoU 较高的邻近框，两阶段 RoI 精修与端到端集合预测的逐一输出机制在此占优；
+ *小目标场景*（GT 仅 5 个，结论谨慎）：RT-DETR 召回 60% 最高但伴随 28 个 FP——"检得多也碎得多"，与 @sec-fp 节碎片框分析一致；
+ *低对比场景*分化最大：Faster R-CNN 8/8 全召回（代价 12 个 FP），YOLOv8s 仅 12.5%——CFD 子集的低对比特性对 v8s 的置信度校准冲击最大，而 v10s（75.0%）显著好于同代 v8s。

== FP 归因 <sec-fp>

将每模型在 conf=0.25 下的全部 FP 按与最近 GT 的 IoU 分为三类：*背景误检*（IoU < 0.1）、*定位差*（0.1 ≤ IoU < 0.5）、*重复框*（命中已被占用的 GT），见@fig-fp。

#figure(
  image("results/figures/fp_attribution.png", width: 62%),
  caption: [FP 三类归因（test 全集，conf=0.25，IoU=0.5）],
) <fig-fp>

+ *定位差在所有模型中都是第一大 FP 来源*（占比 53–63%）——细长裂缝框稍偏即跌破 0.5 IoU，再次印证 IoU 天花板是数据固有属性而非某个模型的缺陷；
+ *背景误检很少*（YOLO 系仅 33–55 个），裂缝与背景的语义区分对本组模型都不难；
+ *重复框占比分化明显*：YOLO 系 23–29% vs torchvision 6–8%。YOLO 倾向把同一条长裂缝输出成多个碎片框（碎片间互 IoU 低，NMS 抑制不住）；NMS-free 的 YOLOv10s 仍有 93 个重复框，说明其 one-to-one 头在细长目标上未能完全抑制碎片化输出；torchvision 系的 NMS 相对更彻底。Faster R-CNN 的 FP 总量（1768）远大于其他模型，其中定位差 1113 个——这正解释了@tab-eng 中它"高召回 + 低精确率"的形态：大量勉强过 0.25 阈值的低分框。

== 漏检的尺寸效应 <sec-miss>

@fig-miss 给出 conf=0.25 下按 GT 尺寸分桶的召回率（test 集 S/M/L 分别为 112/284/700 个实例）。

#figure(
  image("results/figures/miss_by_size.png", width: 66%),
  caption: [按 GT 尺寸分桶的召回率（conf=0.25，IoU=0.5）],
) <fig-miss>

+ *小目标（\<32²）召回 0–8%*，全体失效——细长裂缝的框面积天然偏小，640 输入下像素证据不足，与 AP_S 崩塌互为印证；
+ *中目标段差距最大*：Faster R-CNN 47.2%、RetinaNet 40.9%，显著高于 YOLO 系（13.7–20.8%）与 RT-DETR（34.2%）——多尺度 anchor 覆盖 + RoI 精修对中等尺寸细长目标收益最明显，这是两阶段总精度领先的主要来源；
+ *大目标段全体可用*（72.7–90.4%），Faster R-CNN 仍以 90.4% 居首。

= 速度-精度权衡的本质差异

== GFLOPs ≠ 延迟

以 FP32 延迟除以 GFLOPs 得到"每 GFLOP 耗时"（ms）：YOLOv8s 0.095 < Faster R-CNN 0.126 ≈ RetinaNet 0.132 < YOLOv10s 0.138 < YOLO11s 0.181 < RT-DETR-L 0.232。同样的算力量换来的时间相差 2.4 倍，原因在架构而非算力：

+ *RT-DETR-L 的 Transformer 编码器与逐 query 解码访存受限*，deformable attention 的稀疏采样与大量小矩阵运算对 GPU 并行不友好，算力利用率最低；
+ *Faster R-CNN 的 RoI 头是稀疏计算*（只对少量候选框做全连接），单位 FLOP 效率反而高于稠密单阶段——两阶段"慢"在串行结构与 NMS，而非算力本身；
+ *YOLO 系稠密单 pass + 融合算子*对现代 GPU 最友好。

一个反例直观说明问题：YOLOv10s 的 GFLOPs（24.89）低于 YOLOv8s（28.64），延迟（3.43 ms）反而更高——NMS-free 的双头结构虽省去 NMS，其 one-to-one 分支的额外计算与融合方式使实际耗时上升。*算力量不等于时间量，架构与访存模式才是延迟的第一决定因素。*

== 数值精度的影响

三档精度测试（#link(<app-a>)[附录 A] 完整数据）呈现一致规律：

+ *小模型半精度负增益*：YOLOv8s 静态 FP16（2.82 ms）慢于 FP32（2.715 ms）——毫秒级前向中类型转换与 kernel launch 开销占比超过 Tensor Core 收益；
+ *大模型有收益*：Faster R-CNN autocast FP16 提速 9.5%（11.456 → 10.461 ms）；
+ *BF16 全线持平或更慢*，RT-DETR-L 两种低精度均无收益（26.3 vs 25.5 ms）——其推理瓶颈不在矩阵乘精度档位。

== Pareto 前沿与选型建议

@fig-speed 给出速度-精度散点（横轴 FP16 延迟，纵轴 mAP\@0.5:0.95，红色虚线为 Pareto 前沿：YOLOv8s → YOLOv10s → YOLO11s → Faster R-CNN）。

#figure(
  image("results/figures/speed_accuracy.png", width: 70%),
  caption: [Speed-Accuracy 散点（FP16 延迟，batch=1；同一 test 集与同一硬件基准测得，按范式着色）],
) <fig-speed>

+ *精度极点*：Faster R-CNN（mAP\@0.5 62.15 / mAP\@0.5:0.95 35.93 @ 11.5 ms），但延迟是 YOLO 系的 3–4 倍、参数量 41.3 M 最大；
+ *速度极点*：YOLOv8s（2.715 ms，368 FPS），精度在六模型中垫底但仍达 51.63；
+ *综合效率最优*：YOLO11s（53.61 @ 3.9 ms）——精度第二、速度第二、GFLOPs 最低（21.67），且工程口径 F1 最高（60.05），是每毫秒延迟换得精度最多的模型；
+ *RT-DETR-L 落在前沿之外*：精度最低且延迟最长（25.5 ms）。这不是对范式的否定——其在密集场景的召回优势（@sec-hard 节）与"训练未收敛"的曲线形态都表明，在小规模单类数据 + 150 epoch 预算下，端到端集合预测的迁移效率不如卷积系；原版 DETR 在小数据上收敛更慢，本实验选 RT-DETR-L 已是对该流派最有利的配置。

*部署建议*：实时巡检（无人机 / 边缘设备）选 YOLO11s；离线精析、告警复核等对定位精度与召回敏感的环节选 Faster R-CNN，并按 P-R 曲线选取比 0.25 更高的工作点以压制其 FP。

= 结论与改进方向

== 结论

+ 在统一数据、统一硬件、统一预算下完成了四种范式六个模型的完整训练与同口径评测：*Faster R-CNN 以 62.15 / 35.93 的 mAP 双第一验证了两阶段框精修在细长目标上的定位优势*；YOLO 系以 3–4 ms 级延迟与最好的置信度校准成为工程首选；RT-DETR-L 在本预算下未收敛至其上限；
+ 裂缝数据的*IoU 天花板*（细长框偏移敏感性）是所有模型 mAP\@0.5:0.95 偏低、FP 以"定位差"为主、YOLO 碎片框重复的共同根因——这是数据表示（水平框 + 每连通域一框）决定的系统性约束；
+ *小目标（\<32²）是全体模型的盲区*（召回 0–8%），主要由 640 输入分辨率与细长目标的小面积特性造成；
+ 密集重叠场景（召回最高 39.1%）与小目标场景构成该任务真正的开放难题；
+ 工程口径（固定 conf 的 P/R）与学术口径（mAP）的排序*可以完全相反*，部署选型必须结合自身 P/R 偏好考察曲线全貌。

== 改进方向

+ *转框策略消融*：对比"每连通域一框"与"长条按网格切碎成多框"两种标注策略对 mAP\@0.5:0.95 的影响——碎片化标注可显著抬高高 IoU 段上限（@sec-fp 节的碎片框现象提示预测端已自发倾向多框覆盖）；
+ *旋转框（OBB）检测*：细长目标的朝向明确，OBB 直接消除水平框 IoU 天花板，是最对症的表示升级；
+ *高分辨率与切片推理（SAHI）*：针对 AP_S 崩塌，640 输入换 1280 或滑窗切片可成倍增加小目标像素证据；
+ *小目标数据增强*：copy-paste 裂缝粘贴、针对性过采样低对比 CFD 子集；
+ *修复 RT-DETR 训练稳定性并延长预算*：AMP NaN 问题可尝试 BF16 训练、梯度裁剪或降低学习率，并在 300+ epoch 预算下重估其上限。

#pagebreak()
#bibliography("refs.bib", title: [参考文献], style: "gb-7714-2015-numeric")

#pagebreak()
#heading(numbering: none)[附录 A 环境与完整延迟数据] <app-a>

#figure(
  three-line-table[
    | 组件 | 版本 / 型号 |
    | --- | --- |
    | GPU | NVIDIA RTX 4090（24 GB，AutoDL 云实例；训练与延迟测试同一实例） |
    | OS | Linux（云实例） |
    | Python | 3.12 |
    | PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
    | ultralytics | 8.4.172 |
    | pycocotools | 2.0.11 |
    | 依赖管理 | uv（uv.lock 锁定复现） |
  ],
  caption: [软件与硬件环境],
) <tab-env>

#figure(
  three-line-table[
    | 模型 | FP32 均值±std / ms | FP16 | BF16 | FPS（FP32 / FP16） |
    | --- | --- | --- | --- | --- |
    | YOLOv8s | 2.715±0.267 | 2.820±0.228 | 2.930±0.557 | 368.3 / 354.6 |
    | YOLOv10s | 3.430±0.313 | 3.674±0.560 | 3.579±0.421 | 291.6 / 272.2 |
    | YOLO11s | 3.921±0.764 | 3.973±0.589 | 3.907±0.335 | 255.0 / 251.7 |
    | RT-DETR-L | 25.495±1.116 | 26.292±0.588 | 26.203±0.899 | 39.2 / 38.0 |
    | Faster R-CNN R50 | 11.456±2.016 | 10.461±1.966 | 11.690±1.015 | 87.3 / 95.6 |
    | RetinaNet R50 | 10.740±2.950 | 10.507±1.981 | 12.541±2.335 | 93.1 / 95.2 |
  ],
  caption: [三档数值精度完整延迟（batch=1，预热 50 + 计时 300；ultralytics 侧 FP16/BF16 为静态半精度权重，torchvision 侧为 autocast）],
) <tab-latency>

#pagebreak()
#heading(numbering: none)[附录 B 复现命令] <app-b>

```bash
# 环境搭建（uv）
uv sync

# 数据预处理四连
uv run python tools/prepare_data.py
uv run python tools/mask2bbox.py --src data/raw/crackDataset --out data/interim --min-area 64
uv run python tools/split.py --interim data/interim --out splits --seed 0
uv run python tools/export.py --interim data/interim --splits splits --out data

# 数据统计
uv run python tools/analyze.py --coco data/coco/annotations/train.json --out results/figures

# 训练（两套框架）
uv run python train/train_yolo.py --config configs/models.yaml --models yolo11s
uv run python train/train_yolo.py --config configs/models.yaml --models yolov8s yolov10s rtdetr_l
uv run python train/train_torchvision.py --config configs/models.yaml --models fasterrcnn_r50_fpn retinanet_r50_fpn

# 测试集推理（全体 conf=0.001，imgsz=640）
for m in yolov8s yolov10s yolo11s rtdetr_l; do
  uv run python eval/infer_yolo.py --model-name $m \
    --weights runs/$m/weights/best.pt \
    --test-json data/coco/annotations/test.json \
    --img-dir data/yolo/images/test \
    --out runs/$m/predict_coco.json
done
uv run python eval/infer_torchvision.py --model-name fasterrcnn_r50_fpn retinanet_r50_fpn

# 统一评测 + 速度基准 + 汇总
uv run python eval/eval_coco.py --gt data/coco/annotations/test.json \
  --pred runs/yolo11s/predict_coco.json --out results/metrics/yolo11s.json   # 其余模型同构
uv run python eval/benchmark.py --config configs/models.yaml --out results/benchmark.json
uv run python eval/summarize.py --metrics results/metrics --bench results/benchmark.json \
  --out results/summary.csv --plot results/figures/speed_accuracy.png

# 硬样本可视化
uv run python tools/find_hard_samples.py
uv run python tools/render_compare.py --list eval/hard_samples.txt --runs runs --out results/figures/qualitative
```

#pagebreak()
#heading(numbering: none)[附录 C 超参数配置全文（configs/models.yaml）] <app-c>

```yaml
common:
  seed: 0
  imgsz: 640        # 统一输入边长（torchvision 侧语义为"短边 640"）
  batch: 16         # 全体一致 ⇒ 每 epoch 梯度步数一致
  epochs: 150
  patience: 50      # 连续 50 epoch 无提升即提前终止
  workers: 8
  project: runs

models:
  yolov8s:
    framework: ultralytics
    weights: yolov8s.pt        # COCO 预训练
    extra:
      close_mosaic: 10

  yolov10s:
    framework: ultralytics
    weights: yolov10s.pt
    extra:
      close_mosaic: 10

  yolo11s:
    framework: ultralytics
    weights: yolo11s.pt
    extra:
      close_mosaic: 10

  rtdetr_l:
    framework: ultralytics
    weights: rtdetr-l.pt
    extra:
      deterministic: false   # deformable attention 无确定性反向，官方建议关闭提速
      amp: false             # 禁用混合精度，容易 NaN
      optimizer: AdamW
      lr0: 0.0001

  fasterrcnn_r50_fpn:
    framework: torchvision
    weights: fasterrcnn_resnet50_fpn   # COCO 预训练，检测头换 num_classes=2
    lr: 0.005
    amp: true

  retinanet_r50_fpn:
    framework: torchvision
    weights: retinanet_resnet50_fpn
    lr: 0.005
    amp: true
```

#pagebreak()
#heading(numbering: none)[附录 D 其余硬样本可视化] <app-d>

以下为 @sec-hard 节三张主图之外的其余 9 张硬样本对比图（GT + 六模型网格，conf=0.25）。

#figure(
  image("results/figures/qualitative/compare_crack500_20160316_143433_1921_1081.png", width: 100%),
  caption: [密集 / 重叠：`crack500_20160316_143433_1921_1081`],
) <fig-q-d2>

#figure(
  image("results/figures/qualitative/compare_crack500_20160326_150104_1_361.png", width: 100%),
  caption: [密集 / 重叠：`crack500_20160326_150104_1_361`],
) <fig-q-d3>

#figure(
  image("results/figures/qualitative/compare_crack500_20160402_150317_1921_721.png", width: 100%),
  caption: [密集 / 重叠：`crack500_20160402_150317_1921_721`],
) <fig-q-d4>

#figure(
  image("results/figures/qualitative/compare_crack500_20160328_153536_1_1081.png", width: 100%),
  caption: [细长 / 小目标：`crack500_20160328_153536_1_1081`],
) <fig-q-s2>

#figure(
  image("results/figures/qualitative/compare_crack500_20160328_150821_1281_721.png", width: 100%),
  caption: [细长 / 小目标：`crack500_20160328_150821_1281_721`],
) <fig-q-s3>

#figure(
  image("results/figures/qualitative/compare_crack500_20160326_142327_641_721.png", width: 100%),
  caption: [细长 / 小目标：`crack500_20160326_142327_641_721`],
) <fig-q-s4>

#figure(
  image("results/figures/qualitative/compare_CFD_080add56.png", width: 100%),
  caption: [低对比：`CFD_080add56`],
) <fig-q-l2>

#figure(
  image("results/figures/qualitative/compare_CFD_080sub56.png", width: 100%),
  caption: [低对比：`CFD_080sub56`],
) <fig-q-l3>

#figure(
  image("results/figures/qualitative/compare_CFD_063__39.png", width: 100%),
  caption: [低对比：`CFD_063__39`],
) <fig-q-l4>
