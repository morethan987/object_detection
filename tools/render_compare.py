"""
硬样本并排渲染

uv run python tools/render_compare.py \
    --list eval/hard_samples.txt \
    --runs runs \
    --test-json data/coco/annotations/test.json \
    --img-dir data/yolo/images/test \
    --out results/figures/qualitative \
    --conf 0.25
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib import patches

# 6 个对比模型的标准名称
MODELS = [
    "yolov8s",
    "yolov10s",
    "yolo11s",
    "fasterrcnn_r50_fpn",
    "retinanet_r50_fpn",
    "rtdetr_l",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render Hard Sample Comparison Grid")
    parser.add_argument("--list", type=str, default="eval/hard_samples.txt", help="硬样本文件路径")
    parser.add_argument("--runs", type=str, default="runs", help="训练产出 runs 目录")
    parser.add_argument("--test-json", type=str, default="data/coco/annotations/test.json", help="COCO test.json")
    parser.add_argument("--img-dir", type=str, default="data/yolo/images/test", help="测试集图片目录")
    parser.add_argument("--out", type=str, default="results/figures/qualitative", help="结果保存目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    return parser.parse_args()


def load_hard_samples(list_path: Path) -> list[str]:
    """严格读取清单中的图片文件名（排除空行和注释）"""
    if not list_path.exists():
        sys.exit(f"[错误] 找不到硬样本清单文件: {list_path}")
    samples = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                samples.append(line)
    return samples


def load_coco_ground_truth(test_json_path: Path):
    """
    解析 test.json，返回:
    1. filename_to_id: {file_name: image_id}
    2. id_to_gt_boxes: {image_id: list of [x, y, w, h]}
    """
    if not test_json_path.exists():
        sys.exit(f"[错误] 找不到真值标注文件: {test_json_path}")

    with open(test_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    filename_to_id = {img["file_name"]: img["id"] for img in coco.get("images", [])}

    id_to_gt_boxes = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        id_to_gt_boxes.setdefault(img_id, []).append(ann["bbox"])

    return filename_to_id, id_to_gt_boxes


def load_all_predictions(runs_dir: Path, conf_thresh: float):
    """
    精确读取每个模型的 runs/<model>/predict_coco.json
    返回: {model_name: {image_id: [(bbox, score)]}}
    """
    preds = {}
    for m in MODELS:
        pred_json = runs_dir / m / "predict_coco.json"
        if not pred_json.exists():
            preds[m] = None
            continue

        with open(pred_json, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        model_dict = {}
        for item in raw_list:
            score = float(item.get("score", 0.0))
            if score >= conf_thresh:
                img_id = item["image_id"]
                bbox = item["bbox"]
                model_dict.setdefault(img_id, []).append((bbox, score))
        preds[m] = model_dict
    return preds


def draw_subplot(ax, img_rgb, boxes, title, is_gt=False, color="red"):
    """在子图上绘制图像与边界框"""
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    ax.axis("off")

    for item in boxes:
        if is_gt:
            x, y, w, h = item
            label_text = "GT"
        else:
            bbox, score = item
            x, y, w, h = bbox
            label_text = f"{score:.2f}"

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x,
            max(0, y - 4),
            label_text,
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
        )


def main():
    args = parse_args()
    list_path = Path(args.list)
    runs_dir = Path(args.runs)
    test_json_path = Path(args.test_json)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载文件列表与真值
    sample_files = load_hard_samples(list_path)
    filename_to_id, id_to_gt = load_coco_ground_truth(test_json_path)

    # 2. 精确加载预测
    predictions = load_all_predictions(runs_dir, args.conf)
    print(f"[*] 待渲染硬样本数量: {len(sample_files)}")
    for m in MODELS:
        status = "已找到 predict_coco.json" if predictions[m] is not None else "未找到（占位渲染）"
        print(f"    - {m:<20}: {status}")

    # 3. 渲染
    for idx, fname in enumerate(sample_files, 1):
        img_path = img_dir / fname
        if not img_path.exists():
            print(f"[!] 警告: 图片不存在，跳过 -> {img_path}")
            continue

        image_id = filename_to_id.get(fname)
        if image_id is None:
            print(f"[!] 警告: {fname} 在 test.json 中无对应 ID，跳过")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_boxes = id_to_gt.get(image_id, [])

        # 2x4 网格布局 (GT + 6 模型 + 1 信息面板)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        # Slot 0: GT (绿色)
        draw_subplot(
            axes[0],
            img_rgb,
            gt_boxes,
            title=f"Ground Truth ({len(gt_boxes)} boxes)",
            is_gt=True,
            color="#00FF00",
        )

        # Slot 1~6: 6 个模型预测 (统一红色)
        for i, m in enumerate(MODELS):
            ax = axes[i + 1]
            if predictions[m] is None:
                ax.imshow(img_rgb)
                ax.set_title(f"{m}\n[Not Found]", fontsize=10, color="gray")
                ax.axis("off")
                continue

            m_preds = predictions[m].get(image_id, [])
            draw_subplot(
                ax,
                img_rgb,
                m_preds,
                title=f"{m} ({len(m_preds)} boxes)",
                is_gt=False,
                color="#FF3838",
            )

        # Slot 7: 样本说明面板
        axes[7].axis("off")
        info = (
            f"Sample: {fname}\n"
            f"Size: {img_rgb.shape[1]}x{img_rgb.shape[0]}\n"
            f"GT Boxes: {len(gt_boxes)}\n"
            f"Conf Thresh: >= {args.conf}\n"
            f"-------------------\n"
            f"Green: Ground Truth\n"
            f"Red: Predictions"
        )
        axes[7].text(
            0.1,
            0.5,
            info,
            fontsize=11,
            fontfamily="monospace",
            va="center",
            bbox={"boxstyle": "round,pad=1", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
        )

        plt.suptitle(f"Hard Sample Qualitative Comparison: {fname}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        save_path = out_dir / f"compare_{Path(fname).stem}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[{idx}/{len(sample_files)}] 已保存: {save_path.name}")

    print(f"\n[+] 渲染全部完成，结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()
