"""
硬样本并排渲染：GT + 6 个模型同一批图的结果网格

uv run python tools/render_compare.py --list eval/hard_samples.txt --runs runs --out results/figures/qualitative

输入  eval/hard_samples.txt（人工挑的 ≥12 张: 小目标/重叠/强反光·低对比）
      各模型预测结果（runs/ 下的预测 json 或直接加载权重推理）
输出  每张硬样本一张网格图（1 行 GT + 6 行模型），conf=0.25、统一框色
"""
