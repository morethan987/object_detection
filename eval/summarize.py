"""汇总全部模型的指标 → results/summary.csv（作业要求的横向对比总表）。

用法（实现后）:
    uv run python eval/summarize.py --metrics results/metrics --bench results/benchmark.json \
        --out results/summary.csv

契约（对应 homework.md §11.4 / §12）:
    合并  eval_coco.py 产出（P/R/mAP@0.5/mAP@0.5:0.95/AP_S-M-L）
          + benchmark.py 产出（Params/GFLOPs/Latency fp32&fp16/FPS）
    输出  results/summary.csv + speed-accuracy 散点图（Pareto 前沿，报告第 6 章素材）
          与 homework.md §12 的“预注册预期区间”逐项对照
"""
