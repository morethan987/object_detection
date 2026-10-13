"""
Torchvision 模型测试集批量推理脚本：
将 Faster R-CNN / RetinaNet 的预测结果导出为标准 COCO 预测 JSON

uv run python eval/infer_torchvision.py --model-name fasterrcnn_r50_fpn
uv run python eval/infer_torchvision.py --model-name retinanet_r50_fpn
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class TestDataset(Dataset[tuple[torch.Tensor, int]]):
    """测试集数据集读取：根据 test.json 图片列表与文件目录加载图像"""

    def __init__(self, test_json_path: Path, img_dir: Path) -> None:
        with open(test_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = data.get("images", [])
        self.img_dir = img_dir

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_info = self.images[index]
        img_path = self.img_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        # 归一化为 [0, 1] 的 (C, H, W) Tensor
        img_np = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        return img_tensor, img_info["id"]


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(model_name: str, imgsz: int, conf_thresh: float = 0.001) -> torch.nn.Module:
    """构建与训练阶段一致的单类别检测模型架构（背景=0, crack=1）"""
    if model_name == "fasterrcnn_r50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            min_size=imgsz,
            max_size=imgsz,
            box_score_thresh=conf_thresh,
            box_detections_per_img=300,  # 允许单图保留更多检测框供评估
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    elif model_name == "retinanet_r50_fpn":
        model = retinanet_resnet50_fpn(
            weights=None,
            min_size=imgsz,
            max_size=imgsz,
            score_thresh=conf_thresh,
            topk_candidates=1000,
            detections_per_img=300,
        )
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=2,
        )
    else:
        raise ValueError(f"未知的 torchvision 模型: {model_name}")

    return model


def get_category_id(test_json_path: Path) -> int:
    """获取类别 ID（默认为 0）"""
    with open(test_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    categories = coco_data.get("categories", [])
    return categories[0]["id"] if categories else 0


@torch.no_grad()
def run_inference(
    model_name: str,
    weights_path: Path,
    test_json_path: Path,
    img_dir: Path,
    out_path: Path,
    imgsz: int = 640,
    conf_thresh: float = 0.001,
    batch_size: int = 16,
    device_str: str = "",
) -> None:
    if not weights_path.exists():
        sys.exit(f"[错误] 权重文件不存在: {weights_path}")
    if not test_json_path.exists():
        sys.exit(f"[错误] 标注文件不存在: {test_json_path}")
    if not img_dir.exists():
        sys.exit(f"[错误] 图片目录不存在: {img_dir}")

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] 使用设备: {device}")

    # 1. 初始化模型并加载权重
    print(f"[*] 正在加载模型架构与权重: {weights_path}")
    model = build_model(model_name, imgsz=imgsz, conf_thresh=conf_thresh).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 准备数据集与 DataLoader
    dataset = TestDataset(test_json_path, img_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    target_cat_id = get_category_id(test_json_path)
    print(f"[*] 准备对 {len(dataset)} 张测试图片进行批量推理...")

    # 3. 批量推理
    coco_predictions: list[dict[str, Any]] = []

    for images, image_ids in loader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        for img_id, pred in zip(image_ids, predictions):
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                # 过滤背景 (label 0) 及低于阈值的预测
                if label != 1 or score < conf_thresh:
                    continue

                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                coco_predictions.append(
                    {
                        "image_id": img_id,
                        "category_id": target_cat_id,
                        "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                        "score": round(float(score), 4),
                    }
                )

    print(f"[*] 推理完成，共计产出 {len(coco_predictions)} 个检测预测框。")

    # 4. 保存结果
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_predictions, f, indent=2)

    print(f"[+] 预测结果已成功导出至: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Torchvision 模型测试集批量推理转 COCO JSON")
    parser.add_argument(
        "--model-name",
        type=str,
        default="fasterrcnn_r50_fpn",
        choices=["fasterrcnn_r50_fpn", "retinanet_r50_fpn"],
        help="模型名称",
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="权重文件路径（默认: runs/<model-name>/weights/best.pt）"
    )
    parser.add_argument(
        "--test-json", type=str, default="data/coco/annotations/test.json", help="COCO 格式的测试集标注文件路径"
    )
    parser.add_argument("--img-dir", type=str, default="data/yolo/images/test", help="测试集图片目录")
    parser.add_argument(
        "--out", type=str, default=None, help="输出的预测结果 json 路径（默认: runs/<model-name>/predict_coco.json）"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="推理置信度截断 (默认: 0.001)")
    parser.add_argument("--batch", type=int, default=16, help="推理 Batch 大小 (默认: 16)")
    parser.add_argument("--device", type=str, default="", help="计算设备（如 cuda:0 或 cpu）")
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
        batch_size=args.batch,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
