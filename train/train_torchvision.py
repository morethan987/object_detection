"""
torchvision 侧训练入口：Faster R-CNN R50-FPN / RetinaNet R50-FPN

uv run python train/train_torchvision.py --config configs/models.yaml --models fasterrcnn_r50_fpn retinanet_r50_fpn

依赖路径约定 (由 tools/export.py 生成):
    标注: data/coco/annotations/{split}.json
    图像: data/yolo/images/{split}/{file_name}
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CocoCrackDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, ann_file: Path, img_dir: Path, is_train: bool = True) -> None:
        self.coco = COCO(str(ann_file))
        self.img_dir = img_dir
        self.ids = sorted(self.coco.imgs.keys())
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_id = self.ids[index]
        ann_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / ann_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([float(x), float(y), float(x + w), float(y + h)])
                # torchvision 将 0 视为背景，前景 crack 统一映射为 1
                labels.append(1)

        # 训练集 PIL 原生镜像翻转（规避 torchvision.transforms.functional 类型错误）
        if self.is_train and torch.rand(1).item() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            w, _ = image.size
            if boxes:
                boxes = [[float(w) - x2, y1, float(w) - x1, y2] for x1, y1, x2, y2 in boxes]

        # 转为 (C, H, W) 归一化 Tensor
        img_np = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
        }
        return img_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


def build_model(model_name: str, imgsz: int, project_root: Path) -> torch.nn.Module:
    weights_dir = project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "fasterrcnn_r50_fpn":
        weights_enum = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        # 1. 先构建无权重网络结构
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            min_size=imgsz,
            max_size=imgsz,
        )
        # 2. 权重下载/加载直接指定落盘在 weights/ 目录下
        state_dict = torch.hub.load_state_dict_from_url(
            weights_enum.url,
            model_dir=str(weights_dir),
            progress=True,
        )
        model.load_state_dict(state_dict)

        # 3. 替换分类头
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    elif model_name == "retinanet_r50_fpn":
        weights_enum = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        model = retinanet_resnet50_fpn(
            weights=None,
            min_size=imgsz,
            max_size=imgsz,
        )
        state_dict = torch.hub.load_state_dict_from_url(
            weights_enum.url,
            model_dir=str(weights_dir),
            progress=True,
        )
        model.load_state_dict(state_dict)

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


@torch.no_grad()
def evaluate_coco(
    model: torch.nn.Module,
    val_loader: DataLoader,
    coco_gt: COCO,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    results = []

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        for target, pred in zip(targets, predictions):
            img_id = int(target["image_id"].item())
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                # 仅预测为 crack (label==1) 计入评估
                if label != 1:
                    continue
                x1, y1, x2, y2 = box
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": 0,  # 映射回 export.py 导出的 crack 类别 id (0)
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )

    if not results:
        return 0.0, 0.0, 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.catIds = [0]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_all = float(coco_eval.stats[0])
    map50 = float(coco_eval.stats[1])
    fitness = 0.1 * map50 + 0.9 * map_all
    return map50, map_all, fitness


def train_single_model(
    model_name: str,
    common_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    project_root: Path,
) -> None:
    if model_cfg.get("framework") != "torchvision":
        raise ValueError(f"[{model_name}] 不是 torchvision 模型，跳过。")

    print("\n" + "=" * 70)
    print(f" 开始训练 Torchvision 模型: {model_name} ".center(70, "="))
    print("=" * 70)

    # 1. 基础超参
    seed = common_cfg.get("seed", 0)
    imgsz = common_cfg.get("imgsz", 640)
    batch_size = common_cfg.get("batch", 16)
    epochs = common_cfg.get("epochs", 150)
    patience = common_cfg.get("patience", 50)
    workers = common_cfg.get("workers", 8)
    lr = model_cfg.get("lr", 0.005)
    use_amp = model_cfg.get("amp", True)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_amp = bool(use_amp and device.type == "cuda")

    # 2. 数据路径
    train_ann = project_root / "data" / "coco" / "annotations" / "train.json"
    val_ann = project_root / "data" / "coco" / "annotations" / "val.json"
    train_img_dir = project_root / "data" / "yolo" / "images" / "train"
    val_img_dir = project_root / "data" / "yolo" / "images" / "val"

    if not train_ann.exists() or not val_ann.exists():
        sys.exit("[错误] 标注文件未找到，请先运行: uv run python tools/export.py")

    train_dataset = CocoCrackDataset(train_ann, train_img_dir, is_train=True)
    val_dataset = CocoCrackDataset(val_ann, val_img_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 3. 构建模型、优化器与调度器
    model = build_model(model_name, imgsz=imgsz, project_root=project_root).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    warmup_iters = 500
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler(device=device.type, enabled=enable_amp)

    # 4. 输出目录与日志
    save_dir = project_root / common_cfg.get("project", "runs") / model_name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"common": common_cfg, "model": model_cfg}, f, indent=2, ensure_ascii=False)

    log_csv = save_dir / "log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_map50", "val_map", "fitness", "lr"])

    # 5. 训练循环
    best_fitness = 0.0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:
            global_step += 1
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if global_step <= warmup_iters:
                warmup_factor = float(global_step) / float(max(1, warmup_iters))
                for g in optimizer.param_groups:
                    g["lr"] = lr * warmup_factor

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=enable_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not math.isfinite(losses.item()):
                continue

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()

        if global_step > warmup_iters:
            lr_scheduler.step()

        avg_loss = total_loss / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]

        map50, map_all, fitness = evaluate_coco(
            model=model,
            val_loader=val_loader,
            coco_gt=val_dataset.coco,
            device=device,
        )

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"Loss: {avg_loss:.4f} | "
            f"mAP@0.5: {map50:.4f} | "
            f"mAP@0.5:0.95: {map_all:.4f} | "
            f"Fitness: {fitness:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{avg_loss:.5f}",
                    f"{map50:.5f}",
                    f"{map_all:.5f}",
                    f"{fitness:.5f}",
                    f"{current_lr:.7f}",
                ]
            )

        if fitness > best_fitness:
            best_fitness = fitness
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "fitness": best_fitness,
                    "mAP50": map50,
                    "mAP": map_all,
                },
                weights_dir / "best.pt",
            )
            print(f"[*] 权重已更新 (Fitness: {best_fitness:.4f}) -> {weights_dir / 'best.pt'}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] 连续 {patience} 个 Epoch 无提升，触发早停！")
                break

    print(f"\n[+] 模型 [{model_name}] 训练完成！最佳 Fitness: {best_fitness:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Torchvision 训练入口脚本")
    parser.add_argument("--config", type=str, default="configs/models.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fasterrcnn_r50_fpn"],
        choices=["fasterrcnn_r50_fpn", "retinanet_r50_fpn"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = get_project_root()
    full_config = load_yaml((project_root / args.config).resolve())

    common_cfg = full_config.get("common", {})
    models_pool = full_config.get("models", {})

    for model_name in args.models:
        if model_name not in models_pool:
            print(f"[警告] 模型 '{model_name}' 未在配置中定义，跳过！", file=sys.stderr)
            continue
        train_single_model(
            model_name=model_name,
            common_cfg=common_cfg,
            model_cfg=models_pool[model_name],
            project_root=project_root,
        )


if __name__ == "__main__":
    main()
