"""
mask → bbox：把像素级裂缝掩码转成检测框

uv run python tools/mask2bbox.py --src data/raw/crackDataset --out data/interim --min-area 64

输入  三个子集的 {train,val}/*.jpg 与 {label,val_label}/*.png
处理  二值化(arr > 0，兼容 1-bit 的 0/1、0/255、调色板索引；禁用 ==255)
      → 闭运算(3×3 核, 1–2 次迭代) → cv2.connectedComponentsWithStats
      → 过滤 area < min-area 的噪点连通域
输出  data/interim/<dataset>/<name>.json: {"boxes": [[x1,y1,x2,y2], ...]}
自检  随机 10 张掩码叠加原图存 data/interim/spot_check/，人工目检二值化方向
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from utils import DATA_DIR, DATASET_PAIRS, IMG_EXTENSIONS, SUBSETS


def parse_args():
    parser = argparse.ArgumentParser(description="Convert crack masks to bounding boxes (JSON).")
    parser.add_argument(
        "--src",
        type=Path,
        default=DATA_DIR,
        help="Path to raw dataset directory (default: data/raw/crackDataset)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim"),
        help="Path to output interim directory (default: data/interim)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=64,
        help="Minimum connected component area to keep (default: 64)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Morphological closing iterations (default: 1)",
    )
    parser.add_argument(
        "--spot-check-num",
        type=int,
        default=10,
        help="Number of spot check overlay images to generate (default: 10)",
    )
    return parser.parse_args()


def find_matching_image(mask_stem: str, img_dir: Path) -> Path | None:
    """在原图目录寻找匹配的图像文件"""
    for ext in IMG_EXTENSIONS:
        candidate = img_dir / f"{mask_stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def mask_to_boxes(mask_path: Path, min_area: int, iterations: int):
    """读取掩码 -> 二值化(arr > 0) -> 闭运算 -> 提取连通域外接矩形"""
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw_mask is None:
        return [], None

    # 二值化契约：arr > 0，禁用 == 255
    if raw_mask.ndim == 3:
        binary = (np.any(raw_mask[..., :3] > 0, axis=-1)).astype(np.uint8) * 255
    else:
        binary = (raw_mask > 0).astype(np.uint8) * 255

    # 闭运算 (3×3 矩形核，缝合微小裂隙)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # 连通域统计 (8邻域)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    boxes = []
    # 索引 0 为背景，从 1 开始遍历每个连通域
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, x + w, y + h])

    return boxes, binary


def main():
    args = parse_args()
    src_dir = args.src
    out_dir = args.out
    min_area = args.min_area
    spot_check_dir = out_dir / "spot_check"
    spot_check_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 正在扫描数据集: {src_dir}")
    print(f"[INFO] 过滤阈值 min-area: {min_area}, 闭运算迭代次数: {args.iterations}")

    # 预先扫描所有待处理任务列表，以便构建准确的总进度条
    tasks = []
    for subset in SUBSETS:
        subset_src = src_dir / subset
        if not subset_src.exists():
            continue

        subset_out = out_dir / subset
        subset_out.mkdir(parents=True, exist_ok=True)

        pairs = DATASET_PAIRS.get(subset, [])
        for img_subname, mask_subname in pairs:
            img_dir = subset_src / img_subname
            mask_dir = subset_src / mask_subname

            if not mask_dir.exists():
                continue

            mask_files = [p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
            for mask_path in mask_files:
                tasks.append((subset, subset_out, img_dir, mask_path))

    if not tasks:
        print("[WARN] 未找到任何待处理掩码文件，请检查数据集路径。")
        return

    processed_samples = []  # 用于后续随机抽检 (subset, sample_name, img_path, binary, boxes)
    total_images_processed = 0

    # 主进度条
    pbar = tqdm(tasks, desc="Mask → BBox", unit="img")
    for subset, subset_out, img_dir, mask_path in pbar:
        pbar.set_postfix(dataset=subset, file=mask_path.name[:15])

        boxes, binary = mask_to_boxes(mask_path, min_area, args.iterations)
        if binary is None:
            continue

        img_path = find_matching_image(mask_path.stem, img_dir)
        sample_name = img_path.stem if img_path else mask_path.stem

        # 保存 JSON 标注
        json_path = subset_out / f"{sample_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"boxes": boxes}, f, indent=2)

        total_images_processed += 1

        if img_path and img_path.exists():
            processed_samples.append((subset, sample_name, img_path, binary, boxes))

    print(f"\n[INFO] 掩码转换完成，共处理 {total_images_processed} 张图片。")

    # 执行自检抽样可视化
    num_spot_check = min(args.spot_check_num, len(processed_samples))
    if num_spot_check > 0:
        print(f"[INFO] 正在生成 {num_spot_check} 张自检叠加图到: {spot_check_dir}")
        selected_samples = random.sample(processed_samples, num_spot_check)

        for subset, name, img_path, binary, boxes in tqdm(selected_samples, desc="Spot checking", unit="img"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if (binary.shape[0], binary.shape[1]) != (img.shape[0], img.shape[1]):
                binary = cv2.resize(binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 半透明掩码（红色）
            overlay = img.copy()
            overlay[binary > 0] = [0, 0, 255]
            blended = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

            # 矩形检测框（绿色）
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)

            save_path = spot_check_dir / f"{subset}_{name}_check.jpg"
            cv2.imwrite(str(save_path), blended)

        print(f"[INFO] 自检图生成完毕，请打开 {spot_check_dir} 人工核验裂缝和框是否对齐。")


if __name__ == "__main__":
    main()
