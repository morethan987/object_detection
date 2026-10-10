"""
导出双格式数据集：YOLO 目录结构 + COCO json + crack.yaml。

uv run python tools/export.py --interim data/interim --splits splits --out data

输入  data/interim（mask2bbox 产物）+ splits/*.txt
输出  data/yolo/{images,labels}/{train,val,test}/   （YOLO txt: xc,yc,w,h 归一化）
      data/coco/annotations/{train,val,test}.json   （COCO: [x,y,w,h] 绝对像素）
      data/crack.yaml                              （ultralytics 数据集配置, names: ['crack']）
细节  转框后为空的 tile 保留为背景图（YOLO 侧写空 label 文件）
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm
from utils import DATA_DIR, DATASET_PAIRS, IMG_EXTENSIONS, SUBSETS


def parse_args():
    parser = argparse.ArgumentParser(description="导出 YOLO/COCO 数据集格式及 Ultralytics 配置文件")
    parser.add_argument(
        "--interim",
        type=Path,
        default=Path("data/interim"),
        help="中间 JSON 标注目录 (默认: data/interim)",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("splits"),
        help="划分文件目录 (默认: splits)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="最终数据导出根目录 (默认: data)",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DATA_DIR,
        help="原始图像所在目录 (默认: data/raw/crackDataset)",
    )
    return parser.parse_args()


def build_image_index(raw_dir: Path) -> dict[tuple[str, str], Path]:
    """建立 (subset, stem) -> image_path 的快速查找字典，避免反复扫盘"""
    index = {}
    for subset in SUBSETS:
        subset_dir = raw_dir / subset
        for img_subname, _ in DATASET_PAIRS.get(subset, []):
            img_dir = subset_dir / img_subname
            if not img_dir.exists():
                continue
            for p in img_dir.iterdir():
                if p.suffix.lower() in IMG_EXTENSIONS:
                    index[(subset, p.stem)] = p
    return index


def export_split(
    split_name: str,
    split_file: Path,
    interim_dir: Path,
    image_index: dict[tuple[str, str], Path],
    yolo_dir: Path,
    coco_ann_dir: Path,
):
    """处理单个数据集划分 (train/val/test) 的转换与导出"""
    if not split_file.exists():
        print(f"[WARN] 划分文件不存在: {split_file}，跳过该 split")
        return

    # 准备输出目录
    yolo_img_dir = yolo_dir / "images" / split_name
    yolo_lbl_dir = yolo_dir / "labels" / split_name
    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(split_file, "r", encoding="utf-8") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    # COCO 格式数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "crack"}],
    }

    coco_img_id = 1
    coco_ann_id = 1

    pbar = tqdm(rel_paths, desc=f"Exporting {split_name}", unit="file")
    for rel_path in pbar:
        json_path = interim_dir / rel_path
        if not json_path.exists():
            continue

        # 解析子集名称和样本文件名 (例: crack500 / sample_01)
        subset = json_path.parent.name
        stem = json_path.stem

        # 查找原始图像文件
        raw_img_path = image_index.get((subset, stem))
        if not raw_img_path or not raw_img_path.exists():
            continue

        # 获取图像宽高并拷贝原图
        img = cv2.imread(str(raw_img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 目标图像与标签文件名：添加 subset 前缀避免跨子集同名覆盖
        target_name = f"{subset}_{stem}"
        target_img_name = f"{target_name}{raw_img_path.suffix}"
        target_img_path = yolo_img_dir / target_img_name
        target_lbl_path = yolo_lbl_dir / f"{target_name}.txt"

        # 拷贝图片到 YOLO 图像目录
        shutil.copy2(raw_img_path, target_img_path)

        # 读取检测框列表 [[x1, y1, x2, y2], ...]
        with open(json_path, "r", encoding="utf-8") as f:
            boxes = json.load(f).get("boxes", [])

        # 1. 写入 COCO images 节点
        coco_data["images"].append(
            {
                "id": coco_img_id,
                "file_name": target_img_name,
                "width": w,
                "height": h,
            }
        )

        yolo_lines = []
        for x1, y1, x2, y2 in boxes:
            # 边界保护与有效性校验
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue

            # (1) YOLO 归一化中心点及宽高 [0, 1]
            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            norm_w = box_w / w
            norm_h = box_h / h
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            # (2) COCO 绝对像素框 [x, y, w, h]
            coco_data["annotations"].append(
                {
                    "id": coco_ann_id,
                    "image_id": coco_img_id,
                    "category_id": 0,
                    "bbox": [x1, y1, box_w, box_h],
                    "area": float(box_w * box_h),
                    "iscrowd": 0,
                }
            )
            coco_ann_id += 1

        # 写入 YOLO 标签：无框时写空文件，作为负样本背景图
        with open(target_lbl_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

        coco_img_id += 1

    # 导出 COCO JSON
    coco_json_path = coco_ann_dir / f"{split_name}.json"
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)


def main():
    args = parse_args()
    interim_dir = args.interim
    splits_dir = args.splits
    out_dir = args.out
    raw_dir = args.raw

    yolo_dir = out_dir / "yolo"
    coco_ann_dir = out_dir / "coco" / "annotations"
    coco_ann_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 正在建立原图索引...")
    image_index = build_image_index(raw_dir)
    print(f"[INFO] 成功索引 {len(image_index)} 张原始图像。")

    # 分别导出 train, val, test
    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}.txt"
        export_split(split, split_file, interim_dir, image_index, yolo_dir, coco_ann_dir)

    # 生成 Ultralytics crack.yaml 配置文件
    crack_yaml = out_dir / "crack.yaml"
    yolo_abs_path = (out_dir / "yolo").resolve()
    yaml_text = (
        f"path: {yolo_abs_path}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: crack\n"
    )
    with open(crack_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("\n[SUCCESS] 数据导出完成！")
    print(f" - YOLO 数据集: {yolo_dir}")
    print(f" - COCO 标注:   {coco_ann_dir}")
    print(f" - YOLO 配置文件: {crack_yaml}")


if __name__ == "__main__":
    main()
