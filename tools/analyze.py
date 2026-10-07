"""
数据集统计与可视化

uv run python tools/analyze.py --coco data/coco/annotations/train.json --out results/figures

产出  每图实例数分布直方图
      box 长宽比分布直方图（裂缝极度细长——anchor 适用性与 mAP 上限的论据）
      COCO 口径尺寸分桶占比: 小 <32² / 中 32²–96² / 大 >96²（对应 AP_S/AP_M/AP_L）
输出  results/figures/data_*.png + 一份统计摘要 json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# 配置无 GUI 后端，防止服务器无显示器报错
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="COCO 格式数据集统计分析与可视化")
    parser.add_argument(
        "--coco",
        type=Path,
        default=Path("data/coco/annotations/train.json"),
        help="COCO 格式 JSON 标注文件路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures"),
        help="图表与统计摘要输出目录",
    )
    return parser.parse_args()


def plot_instances_per_image(counts: list[int], out_path: Path):
    """绘制每张图像的实例数分布直方图"""
    plt.figure(figsize=(7, 5))
    bins = range(max(counts) + 2)
    plt.hist(counts, bins=bins, align="left", rwidth=0.8, color="#3498db", edgecolor="black")
    plt.title("Distribution of Instances per Image", fontsize=14)
    plt.xlabel("Number of Instances", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_aspect_ratios(ratios: list[float], out_path: Path):
    """绘制目标框长短边比例分布直方图（max(w,h) / min(w,h)）"""
    plt.figure(figsize=(7, 5))
    # 截断极度细长（>15）的目标以便直方图主干清晰展示
    clipped_ratios = np.clip(ratios, 1, 15)
    plt.hist(clipped_ratios, bins=30, color="#e67e22", edgecolor="black")
    plt.title("BBox Aspect Ratio Distribution (Max/Min)", fontsize=14)
    plt.xlabel("Aspect Ratio (elongation)", fontsize=12)
    plt.ylabel("BBox Count", fontsize=12)
    plt.axvline(np.median(ratios), color="red", linestyle="--", label=f"Median: {np.median(ratios):.2f}")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_size_distribution(small_cnt: int, med_cnt: int, large_cnt: int, out_path: Path):
    """绘制 COCO 尺度分桶（小/中/大）占比柱状图"""
    plt.figure(figsize=(7, 5))
    categories = ["Small (<32²)", "Medium (32²–96²)", "Large (>96²)"]
    counts = [small_cnt, med_cnt, large_cnt]
    total = sum(counts)
    percentages = [c / total * 100 if total > 0 else 0 for c in counts]
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

    bars = plt.bar(categories, percentages, color=colors, edgecolor="black", width=0.55)
    plt.title("COCO Scale Distribution (AP_S / AP_M / AP_L)", fontsize=14)
    plt.xlabel("Scale Bucket", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.ylim(0, max(percentages + [10]) * 1.15)

    for bar, count, pct in zip(bars, counts, percentages):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.0,
            f"{pct:.1f}%\n({count})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 正在读取 COCO 标注文件: {args.coco}")
    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    total_images = len(images)
    total_annotations = len(annotations)

    if total_images == 0:
        print("[WARN] JSON 文件中没有任何图片数据。")
        return

    # 1. 统计每张图的实例数（包含 0 实例的背景负样本）
    ann_counts_map = Counter([ann["image_id"] for ann in annotations])
    instances_per_image = [ann_counts_map[img["id"]] for img in images]
    background_images = instances_per_image.count(0)

    # 2. 统计长宽比与细长程度（max(w, h) / min(w, h)）
    aspect_ratios = []
    for ann in annotations:
        _, _, w, h = ann["bbox"]
        if w > 0 and h > 0:
            elongation = max(w, h) / min(w, h)
            aspect_ratios.append(elongation)

    # 3. 统计 COCO 口径尺寸分桶 (< 32², 32²–96², > 96²)
    small_thresh = 32**2
    large_thresh = 96**2
    small_cnt = sum(1 for ann in annotations if ann.get("area", 0) < small_thresh)
    med_cnt = sum(1 for ann in annotations if small_thresh <= ann.get("area", 0) <= large_thresh)
    large_cnt = sum(1 for ann in annotations if ann.get("area", 0) > large_thresh)

    # 生成图表
    print("[INFO] 正在绘制统计图表...")
    plot_instances_per_image(instances_per_image, args.out / "data_instances_per_image.png")
    if aspect_ratios:
        plot_aspect_ratios(aspect_ratios, args.out / "data_aspect_ratio.png")
    plot_size_distribution(small_cnt, med_cnt, large_cnt, args.out / "data_size_distribution.png")

    # 构建并保存统计摘要 JSON
    summary = {
        "dataset_file": str(args.coco),
        "total_images": total_images,
        "background_images": background_images,
        "background_ratio_pct": round(background_images / total_images * 100, 2),
        "total_instances": total_annotations,
        "instances_per_image": {
            "mean": round(float(np.mean(instances_per_image)), 2),
            "median": round(float(np.median(instances_per_image)), 2),
            "max": int(np.max(instances_per_image)),
        },
        "aspect_ratio_elongation": {
            "mean": round(float(np.mean(aspect_ratios)), 2) if aspect_ratios else 0,
            "median": round(float(np.median(aspect_ratios)), 2) if aspect_ratios else 0,
            "p90": round(float(np.percentile(aspect_ratios, 90)), 2) if aspect_ratios else 0,
        },
        "coco_size_distribution": {
            "small_count": small_cnt,
            "small_pct": round(small_cnt / total_annotations * 100, 2) if total_annotations else 0,
            "medium_count": med_cnt,
            "medium_pct": round(med_cnt / total_annotations * 100, 2) if total_annotations else 0,
            "large_count": large_cnt,
            "large_pct": round(large_cnt / total_annotations * 100, 2) if total_annotations else 0,
        },
    }

    summary_path = args.out / "data_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印终端控制台摘要
    print("\n" + "=" * 60)
    print("           数据集探索性分析摘要 (EDA Summary)")
    print("=" * 60)
    print(f"总图片数:         {total_images}")
    print(f"纯背景图数:       {background_images} ({summary['background_ratio_pct']}%)")
    print(f"标注裂缝总数:     {total_annotations}")
    print(f"平均每图裂缝数:   {summary['instances_per_image']['mean']} (最大: {summary['instances_per_image']['max']})")
    if aspect_ratios:
        print(
            f"长短边比例中位数: {summary['aspect_ratio_elongation']['median']} (90%分位数: {summary['aspect_ratio_elongation']['p90']})"
        )
    print(f"小目标 (<32²):    {small_cnt} ({summary['coco_size_distribution']['small_pct']}%)")
    print(f"中目标 (32²–96²): {med_cnt} ({summary['coco_size_distribution']['medium_pct']}%)")
    print(f"大目标 (>96²):    {large_cnt} ({summary['coco_size_distribution']['large_pct']}%)")
    print("=" * 60)
    print(f"[SUCCESS] 图表与摘要已保存至: {args.out}/\n")


if __name__ == "__main__":
    main()
