"""
Ultralytics 模型测试集批量推理脚本：
将 YOLOv8s / YOLOv10s / YOLO11s / RT-DETR-L 的预测结果转为标准 COCO 预测 JSON

# Single run
uv run python eval/infer_yolo.py \
    --model-name yolo11s \
    --weights runs/yolo11s/weights/best.pt \
    --test-json data/coco/annotations/test.json \
    --img-dir data/yolo/images/test \
    --out runs/yolo11s/predict_coco.json

# Batch run
for model in yolov8s yolov10s yolo11s rtdetr_l; do
    echo "[*] Infering: ${model}..."
    uv run python eval/infer_yolo.py \
        --model-name ${model} \
        --weights runs/${model}/weights/best.pt \
        --test-json data/coco/annotations/test.json \
        --img-dir data/yolo/images/test \
        --out runs/${model}/predict_coco.json
done
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ultralytics import RTDETR, YOLO


def get_project_root() -> Path:
    """获取项目根目录绝对路径（本脚本位于 eval/ 目录下）"""
    return Path(__file__).resolve().parent.parent


def load_coco_image_mapping(test_json_path: Path) -> dict[str, int]:
    """
    读取 test.json，建立 {图片文件名: image_id} 字典映射，
    保证预测结果能精确挂载到 COCO 真值体系上。
    """
    if not test_json_path.exists():
        raise FileNotFoundError(f"找不到测试集真值标注文件: {test_json_path}")

    with open(test_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    if not images:
        raise ValueError(f"{test_json_path} 中没有找到任何 images 字段记录！")

    mapping = {img["file_name"]: img["id"] for img in images}
    print(f"[*] 成功从 {test_json_path.name} 加载 {len(mapping)} 张测试图片的 ID 映射关系。")
    return mapping


def get_category_id(test_json_path: Path) -> int:
    """自动获取裂缝类别的 category_id（通常为 1 或 0）"""
    with open(test_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    categories = coco_data.get("categories", [])
    if categories:
        return categories[0]["id"]
    return 0  # 默认兜底为 0


def instantiate_model(weights_path: str, model_name: str):
    """根据模型名称或权重路径选择正确的 Ultralytics 加载器"""
    lower_str = (model_name + weights_path).lower()
    if "rtdetr" in lower_str:
        return RTDETR(weights_path)
    return YOLO(weights_path)


def run_inference(
    model_name: str,
    weights_path: Path,
    test_json_path: Path,
    img_dir: Path,
    out_path: Path,
    imgsz: int = 640,
    conf_thresh: float = 0.001,
    device: str = "",
    batch: int = 16,
) -> None:
    """执行批量推理并导出标准 JSON"""
    if not weights_path.exists():
        sys.exit(f"[错误] 权重文件不存在: {weights_path}")
    if not img_dir.exists():
        sys.exit(f"[错误] 测试集图片目录不存在: {img_dir}")

    # 1. 准备映射字典与类别 ID
    name_to_id = load_coco_image_mapping(test_json_path)
    target_cat_id = get_category_id(test_json_path)

    # 2. 实例化模型
    print(f"[*] 正在加载模型权重: {weights_path}")
    model = instantiate_model(str(weights_path), model_name)

    # 3. 收集测试图片路径
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in supported_exts]
    if not img_paths:
        sys.exit(f"[错误] 在目录 {img_dir} 中未找到任何支持的图片文件！")
    print(f"[*] 准备对 {len(img_paths)} 张测试图片进行批量推理...")

    # 4. 执行推理
    # conf 设置为 0.001 是标准学术评测操作，保证保留足够的候选框以供精准绘制 PR 曲线
    predict_args: dict[str, Any] = {
        "source": img_paths,
        "imgsz": imgsz,
        "conf": conf_thresh,
        "batch": batch,
        "verbose": False,
        "save": False,
    }
    if device:
        predict_args["device"] = device

    results = model.predict(**predict_args)

    # 5. 格式转换：从 Ultralytics Results 转为 COCO 评测格式列表
    coco_predictions: list[dict[str, Any]] = []
    skipped_count = 0

    for r in results:
        img_file_name = Path(r.path).name
        if img_file_name not in name_to_id:
            skipped_count += 1
            continue

        image_id = name_to_id[img_file_name]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            continue

        # 提取 xyxy 坐标 (Tensor/Numpy -> List)
        xyxy_boxes = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()

        for box, score in zip(xyxy_boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": target_cat_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(float(score), 4),
                }
            )

    if skipped_count > 0:
        print(f"[!] 警告: 有 {skipped_count} 张图片在 test.json 中未找到匹配的 image_id。")

    print(f"[*] 推理完成，共计产出 {len(coco_predictions)} 个检测预测框。")

    # 6. 保存落盘
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_predictions, f, indent=2)

    print(f"[+] 预测结果已成功导出至: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ultralytics 模型测试集批量推理转 COCO JSON")
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolo11s",
        help="模型名称（例如 yolo11s, yolov8s, yolov10s, rtdetr_l）",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="权重文件路径（默认: runs/<model-name>/weights/best.pt）",
    )
    parser.add_argument(
        "--test-json",
        type=str,
        default="data/coco/annotations/test.json",
        help="COCO 格式的测试集标注文件路径",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="data/yolo/images/test",
        help="测试集图片目录",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出的预测结果 json 路径（默认: runs/<model-name>/predict_coco.json）",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="推理置信度截断 (默认: 0.001)")
    parser.add_argument("--batch", type=int, default=16, help="推理 Batch 大小 (默认: 16)")
    parser.add_argument("--device", type=str, default="", help="计算设备（如 0 或 cpu，默认自动）")
    return parser.parse_args()


def main():
    args = parse_args()
    root = get_project_root()

    weights = Path(args.weights).resolve() if args.weights else root / f"runs/{args.model_name}/weights/best.pt"
    test_json = Path(args.test_json).resolve()
    img_dir = Path(args.img_dir).resolve()
    out = Path(args.out).resolve() if args.out else root / f"runs/{args.model_name}/predict_coco.json"

    run_inference(
        model_name=args.model_name,
        weights_path=weights,
        test_json_path=test_json,
        img_dir=img_dir,
        out_path=out,
        imgsz=args.imgsz,
        conf_thresh=args.conf,
        device=args.device,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
