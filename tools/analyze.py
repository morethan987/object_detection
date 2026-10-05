"""数据集统计与可视化（报告第 2 章素材）。

用法（实现后）:
    uv run python tools/analyze.py --coco data/coco/annotations/train.json --out results/figures

契约（对应 homework.md §8.2）:
    产出  每图实例数分布直方图
          box 长宽比分布直方图（裂缝极度细长——anchor 适用性与 mAP 上限的论据）
          COCO 口径尺寸分桶占比: 小 <32² / 中 32²–96² / 大 >96²（对应 AP_S/AP_M/AP_L）
    输出  results/figures/data_*.png + 一份统计摘要 json
"""
