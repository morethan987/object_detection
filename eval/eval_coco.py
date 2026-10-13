"""
统一指标评测：所有模型的预测转 COCO json，走同一份 COCOeval

uv run python eval/eval_coco.py \
    --gt data/coco/annotations/test.json \
    --pred runs/yolo11s/predict_coco.json \
    --out results/metrics/yolo11s.json

for model in fasterrcnn_r50_fpn retinanet_r50_fpn rtdetr_l yolo11s yolov10s yolov8s; do
    echo "========================================"
    echo "[*] Evaluating: ${model}..."
    echo "========================================"
    
    uv run python eval/eval_coco.py \
        --gt data/coco/annotations/test.json \
        --pred runs/${model}/predict_coco.json \
        --out results/metrics/${model}.json
done

口径说明:
    1. 学术标准口径: pycocotools COCOeval, max_det=300, score 阈值 0.05 起评
       产出: mAP@0.5 / mAP@0.5:0.95 / AP_S / AP_M / AP_L
    2. 工程应用口径: conf=0.25, IoU=0.50 下的精确率(Precision)、召回率(Recall)与 F1

约束:
    六个模型必须用同一份 test.json 标注评测，绝不依赖各训练框架自带的验证实现。
"""

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """
    计算两个 [x, y, w, h] 格式检测框的 IoU
    """
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, w1 * h1)
    area2 = max(0.0, w2 * h2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_engineering_metrics(
    coco_gt: COCO,
    predictions: list[dict[str, Any]],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> dict[str, Any]:
    """
    计算工程应用口径下的指标 (conf >= conf_thresh, IoU >= iou_thresh)
    采用标准贪心匹配计算 TP, FP, FN
    """
    # 筛选置信度
    filtered_preds = [p for p in predictions if p.get("score", 0.0) >= conf_thresh]

    # 按 image_id 归类
    preds_by_img: dict[int, list[dict[str, Any]]] = {}
    for p in filtered_preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        # 获取该图所有的 GT 目标框
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id, iscrowd=False)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        gt_boxes = [ann["bbox"] for ann in gt_anns]
        gt_matched = [False] * len(gt_boxes)

        # 获取该图预测框，并按置信度降序排序
        img_preds = preds_by_img.get(img_id, [])
        img_preds = sorted(img_preds, key=lambda x: x.get("score", 0.0), reverse=True)

        for pred in img_preds:
            pred_box = pred["bbox"]
            best_iou = 0.0
            best_gt_idx = -1

            # 寻找重合度最高且尚未匹配的 GT
            for idx, gt_box in enumerate(gt_boxes):
                if not gt_matched[idx]:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                total_tp += 1
                gt_matched[best_gt_idx] = True
            else:
                total_fp += 1

        # 未被检测到的 GT 算作漏检 (FN)
        total_fn += len(gt_boxes) - sum(gt_matched)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "conf_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def align_category_ids(coco_gt: COCO, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    检查并修复类别 ID 不一致问题:
    若检测集只有 1 个类别，而预测类别 ID 与 GT 类别 ID 不一致时（常见为 0 与 1 的偏差），自动对齐。
    """
    gt_cat_ids = coco_gt.getCatIds()
    if len(gt_cat_ids) == 1 and predictions:
        expected_cat_id = gt_cat_ids[0]
        sample_pred_cat = predictions[0].get("category_id")
        if sample_pred_cat != expected_cat_id:
            print(
                f"[*] 检测到预测类别 ID ({sample_pred_cat}) 与 GT 类别 ID ({expected_cat_id}) 不一致，"
                f"已自动对齐为 {expected_cat_id}。"
            )
            for p in predictions:
                p["category_id"] = expected_cat_id
    return predictions


def run_evaluation(
    gt_path: Path,
    pred_path: Path,
    out_path: Path | None,
    min_score_coco: float = 0.05,
    eng_conf: float = 0.25,
) -> dict[str, Any]:
    """核心评测流程"""
    print(f"[*] 正在加载真值标注 (GT): {gt_path}")
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))

    print(f"[*] 正在加载预测结果 (Pred): {pred_path}")
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        raise TypeError(f"预测文件格式错误，期望 JSON List，但得到了: {type(predictions)}")

    # 对齐类别 ID
    predictions = align_category_ids(coco_gt, predictions)

    # 1. 计算工程口径指标 (conf >= 0.25)
    print(f"[*] 正在计算工程应用指标 (conf >= {eng_conf}, IoU >= 0.50)...")
    eng_metrics = compute_engineering_metrics(coco_gt, predictions, conf_thresh=eng_conf, iou_thresh=0.50)

    # 2. 计算标准 COCOeval 指标 (score >= 0.05, maxDets=300)
    print(f"[*] 正在运行 COCOeval (score >= {min_score_coco}, max_det=300)...")
    coco_preds = [p for p in predictions if p.get("score", 0.0) >= min_score_coco]

    if len(coco_preds) == 0:
        print("[!] 警告: 筛选后没有有效的预测框（可能该模型训练坍塌），指标记为 0！")
        coco_metrics = {
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "mAP_75": 0.0,
            "AP_s": 0.0,
            "AP_m": 0.0,
            "AP_l": 0.0,
            "AR_max_300": 0.0,
        }
    else:
        with redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(coco_preds)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.maxDets = [1, 10, 100, 300]  # 严格对齐 handout: max_det=300
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats = coco_eval.stats
        # stats 索引对齐 pycocotools 规范
        coco_metrics = {
            "mAP_50_95": round(float(stats[0]), 4),
            "mAP_50": round(float(stats[1]), 4),
            "mAP_75": round(float(stats[2]), 4),
            "AP_s": round(float(stats[3]), 4) if stats[3] >= 0 else None,
            "AP_m": round(float(stats[4]), 4) if stats[4] >= 0 else None,
            "AP_l": round(float(stats[5]), 4) if stats[5] >= 0 else None,
            "AR_max_300": round(float(stats[8]), 4),
        }

    # 合并指标成果
    metrics_result = {
        "model_pred_file": str(pred_path),
        "gt_file": str(gt_path),
        "coco_eval_standard": coco_metrics,
        "engineering_metrics": eng_metrics,
    }

    # 打印排版整齐的终端报告
    print("\n" + "=" * 62)
    print(f" 评测结果汇总: {pred_path.parent.name} ".center(62, "="))
    print("=" * 62)
    print(" [1] 标准学术指标 (COCOeval 标准, max_det=300):")
    print(f"     - mAP@0.5      : {coco_metrics['mAP_50'] * 100:.2f}%")
    print(f"     - mAP@0.5:0.95 : {coco_metrics['mAP_50_95'] * 100:.2f}%")
    print(f"     - mAP@0.75     : {coco_metrics['mAP_75'] * 100:.2f}%")
    ap_s_str = f"{coco_metrics['AP_s'] * 100:.2f}%" if coco_metrics["AP_s"] is not None else "N/A"
    ap_m_str = f"{coco_metrics['AP_m'] * 100:.2f}%" if coco_metrics["AP_m"] is not None else "N/A"
    ap_l_str = f"{coco_metrics['AP_l'] * 100:.2f}%" if coco_metrics["AP_l"] is not None else "N/A"
    print(f"     - AP_s (小目标): {ap_s_str}")
    print(f"     - AP_m (中目标): {ap_m_str}")
    print(f"     - AP_l (大目标): {ap_l_str}")
    print("-" * 62)
    print(" [2] 工程实用指标 (conf >= 0.25, IoU >= 0.50):")
    print(f"     - Precision (查准率): {eng_metrics['precision'] * 100:.2f}%")
    print(f"     - Recall    (查全率): {eng_metrics['recall'] * 100:.2f}%")
    print(f"     - F1-score  (平衡值): {eng_metrics['f1'] * 100:.2f}%")
    print(f"     - TP / FP / FN      : {eng_metrics['tp']} / {eng_metrics['fp']} / {eng_metrics['fn']}")
    print("=" * 62 + "\n")

    # 落盘保存 JSON
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics_result, f, indent=2, ensure_ascii=False)
        print(f"[+] 评测结果已持久化保存至: {out_path}")

    return metrics_result


def parse_args():
    parser = argparse.ArgumentParser(description="跨框架统一 COCOeval 评测脚本")
    parser.add_argument(
        "--gt",
        type=str,
        default="data/coco/annotations/test.json",
        help="Ground Truth COCO 标注文件路径",
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="待评测模型预测结果的 COCO JSON 文件路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="指标结果保存路径 (例如 results/metrics/yolo11s.json)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="COCOeval 纳入评测的最小置信度（默认: 0.05）",
    )
    parser.add_argument(
        "--eng-conf",
        type=float,
        default=0.25,
        help="工程应用口径下的固定置信度阈值（默认: 0.25）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gt_file = Path(args.gt).resolve()
    pred_file = Path(args.pred).resolve()
    out_file = Path(args.out).resolve() if args.out else None

    if not gt_file.exists():
        sys.exit(f"[错误] 真值文件不存在: {gt_file}")
    if not pred_file.exists():
        sys.exit(f"[错误] 预测文件不存在: {pred_file}")

    run_evaluation(
        gt_path=gt_file,
        pred_path=pred_file,
        out_path=out_file,
        min_score_coco=args.min_score,
        eng_conf=args.eng_conf,
    )


if __name__ == "__main__":
    main()
