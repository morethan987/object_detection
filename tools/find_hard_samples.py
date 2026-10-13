import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_TEST_JSON = PROJECT_ROOT / "data" / "coco" / "annotations" / "test.json"
YOLO_TEST_IMG_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "test"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "hard_samples.txt"


def get_image_group_id(filename: str) -> str:
    """提取母图ID用于多样性去重（如 CFD_080、crack500_20160328_153616）"""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "crack500":
        # crack500_YYYYMMDD_HHMMSS 为母图
        return "_".join(parts[:3])
    elif len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def get_contrast(img_path: Path) -> float:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 999.0
    return float(np.std(img))


def main():
    if not COCO_TEST_JSON.exists():
        print(f"[错误] 未找到 COCO 标注文件: {COCO_TEST_JSON}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(COCO_TEST_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_id_to_anns = {}
    for ann in coco.get("annotations", []):
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    samples_meta = []
    for img_info in coco.get("images", []):
        file_name = img_info["file_name"]
        img_path = YOLO_TEST_IMG_DIR / file_name
        if not img_path.exists():
            # 兼容路径带子目录或平铺的情况
            img_path = YOLO_TEST_IMG_DIR / Path(file_name).name

        anns = img_id_to_anns.get(img_info["id"], [])
        if len(anns) == 0:
            continue

        areas = [ann.get("area", ann["bbox"][2] * ann["bbox"][3]) for ann in anns]
        avg_area = sum(areas) / len(areas)
        contrast = get_contrast(img_path) if img_path.exists() else 50.0

        samples_meta.append(
            {
                "name": img_path.name if img_path.exists() else Path(file_name).name,
                "group": get_image_group_id(file_name),
                "num_boxes": len(anns),
                "avg_area": avg_area,
                "contrast": contrast,
            }
        )

    print(f"成功解析 {len(samples_meta)} 个测试集样本，正在按多样性筛选...")

    selected_names = set()
    selected_groups = set()

    def pick_samples(sorted_list, k=4):
        picked = []
        for s in sorted_list:
            if s["name"] in selected_names or s["group"] in selected_groups:
                continue
            picked.append(s)
            selected_names.add(s["name"])
            selected_groups.add(s["group"])
            if len(picked) >= k:
                break
        return picked

    # 1. 密集裂缝
    dense_samples = pick_samples(sorted(samples_meta, key=lambda x: x["num_boxes"], reverse=True), k=4)
    # 2. 小目标裂缝
    small_samples = pick_samples(sorted(samples_meta, key=lambda x: x["avg_area"]), k=4)
    # 3. 低对比度弱光
    contrast_samples = pick_samples(sorted(samples_meta, key=lambda x: x["contrast"]), k=4)

    lines = [
        "# ==================================================",
        "# 硬样本清单 (来自测试集，涵盖小目标/密集/低对比各 4 张)",
        "# ==================================================",
        "",
        "# 1. 密集 / 重叠 / 复杂网状裂缝 (Dense & Overlapped)",
    ]
    for s in dense_samples:
        lines.append(s["name"])

    lines.extend(["", "# 2. 细长 / 小目标裂缝 (Small & Tiny Cracks)"])
    for s in small_samples:
        lines.append(s["name"])

    lines.extend(["", "# 3. 低对比度 / 弱光照 / 噪声干扰 (Low Contrast & Noise)"])
    for s in contrast_samples:
        lines.append(s["name"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[完成] 挑选结果已更新至: {OUTPUT_FILE}")
    print("--------------------------------------------------")
    print("1. 密集重叠样本:", [f"{s['name']} (boxes={s['num_boxes']})" for s in dense_samples])
    print("2. 小目标裂缝样本:", [f"{s['name']} (avg_area={int(s['avg_area'])})" for s in small_samples])
    print("3. 低对比度样本:", [f"{s['name']} (std={s['contrast']:.2f})" for s in contrast_samples])
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
