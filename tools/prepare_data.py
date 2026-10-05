"""
拉取并统计原始数据集 -> data/raw/crackDataset/

uv run tools/prepare_data.py

动作  git clone https://github.com/guoguolord/crackDataset（已存在则跳过）
统计  文件对数，样本尺寸，数据增强痕迹可视化
输出  data/raw/crackDataset/{crack500,AsphaltCrack300,CFD}/... and data/raw/crackDataset/data_enhance.png
"""

import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# 配置无 GUI 环境下的 Matplotlib 后端
matplotlib.use("Agg")

REPO_URL = "https://github.com/guoguolord/crackDataset"
DATA_DIR = Path("data/raw/crackDataset")
SUBSETS = ["crack500", "AsphaltCrack300", "CFD"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 数据集子文件夹配对结构
DATASET_PAIRS = {
    "crack500": [("train", "label"), ("val", "val_label")],
    "AsphaltCrack300": [("train", "label")],
    "CFD": [("train", "label"), ("val", "val_label")],
}


@dataclass
class SubsetStats:
    """子集统计结果的数据结构"""

    matched_count: int = 0
    missing_masks: int = 0
    missing_images: int = 0
    sizes: Counter[tuple[int, int]] = field(default_factory=Counter)
    pairs: list[tuple[Path, Path]] = field(default_factory=list)


def run_git_clone() -> None:
    """若数据集目录不存在或未包含目标子集，则执行浅克隆"""
    if DATA_DIR.exists() and any((DATA_DIR / s).exists() for s in SUBSETS):
        print(f"[✓] 目标数据集目录已存在: {DATA_DIR}，跳过克隆。")
        return

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[+] 正在克隆远程仓库 {REPO_URL} 到 {DATA_DIR} ...")
    cmd = ["git", "clone", "--depth", "1", REPO_URL, str(DATA_DIR)]
    try:
        subprocess.run(cmd, check=True)
        print("[✓] 克隆完成。")
    except subprocess.CalledProcessError as e:
        print(f"[!] 克隆失败，请检查网络或 Git 配置: {e}")
        sys.exit(1)


def get_file_dict(folder: Path) -> dict[str, Path]:
    """获取文件夹内所有图片，以 stem（不带后缀的文件名）为 key"""
    if not folder.exists():
        return {}
    return {f.stem: f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS}


def audit_and_collect_stats(data_root: Path) -> dict[str, dict[str, SubsetStats]]:
    """检查图像与掩膜配对情况，并统计图像分辨率分布"""
    print(f"\n{'=' * 20} 数据集统计报告 {'=' * 20}")
    all_stats: dict[str, dict[str, SubsetStats]] = {}

    for dataset_name, pairs in DATASET_PAIRS.items():
        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            print(f"[警告] 数据集目录不存在: {dataset_path}")
            continue

        all_stats[dataset_name] = {}
        print(f"\n📂 数据集: {dataset_name}")

        for img_sub, mask_sub in pairs:
            img_dir = dataset_path / img_sub
            mask_dir = dataset_path / mask_sub

            img_files = get_file_dict(img_dir)
            mask_files = get_file_dict(mask_dir)

            img_stems = set(img_files.keys())
            mask_stems = set(mask_files.keys())

            matched_stems = img_stems & mask_stems
            only_in_img = img_stems - mask_stems
            only_in_mask = mask_stems - img_stems

            sub_key = f"{img_sub} <-> {mask_sub}"
            stats = SubsetStats(
                matched_count=len(matched_stems),
                missing_masks=len(only_in_img),
                missing_images=len(only_in_mask),
            )

            # 统计配对图片的分辨率
            for stem in matched_stems:
                img_path = img_files[stem]
                mask_path = mask_files[stem]
                stats.pairs.append((img_path, mask_path))
                try:
                    with Image.open(img_path) as im:
                        stats.sizes[im.size] += 1
                except (OSError, UnidentifiedImageError):
                    continue

            all_stats[dataset_name][sub_key] = stats

            # 控制台输出报告
            status = "✅" if not only_in_img and not only_in_mask else "⚠️"
            print(f"  {status} [{sub_key}] 配对成功: {stats.matched_count} 对")
            if only_in_img:
                print(f"     - 图像缺失对应掩膜: {len(only_in_img)} 个")
            if only_in_mask:
                print(f"     - 掩膜缺失对应原图: {len(only_in_mask)} 个")

            print("     - 图像尺寸分布 (宽 x 高):")
            for size, count in stats.sizes.most_common(3):
                print(f"       * {size[0]} x {size[1]}: {count} 张")
            if len(stats.sizes) > 3:
                print(f"       * ... 共 {len(stats.sizes)} 种不同尺寸")

    print(f"\n{'=' * 56}")
    return all_stats


def extract_base_name(filename: str, dataset_name: str) -> str:
    """提取各数据集同一原图/衍生样本的通用 Base 前缀"""
    stem = Path(filename).stem
    if dataset_name == "CFD":
        # CFD: 025.jpg, 025_1.jpg, 025add1.jpg -> 匹配开头的数字 "025"
        m = re.match(r"^(\d+)", stem)
        return m.group(1) if m else stem
    if dataset_name == "crack500":
        # crack500: 20160222_081011_1281_721 -> 匹配拍摄时间戳 "20160222_081011"
        m = re.match(r"^(\d{8}_\d{6})", stem)
        return m.group(1) if m else stem.rsplit("_", 2)[0]
    if dataset_name == "AsphaltCrack300":
        # AsphaltCrack300: 001_04.jpg, 001_05.jpg -> 匹配 "001"
        return stem.split("_")[0]
    return stem


def pick_augmented_samples(train_dir: Path, dataset_name: str, target_prefix: str | None = None) -> list[Path]:
    """在一组样本中找出 1 个原始/基准样本及其 3 个衍生切片或增强样本，共 4 张"""
    if not train_dir.exists():
        return []

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(train_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            base = extract_base_name(p.name, dataset_name)
            groups[base].append(p)

    # 优先使用典型前缀展示，若没有则选取首个具有 4 张以上衍生图的组
    selected_group: list[Path] = []
    if target_prefix and target_prefix in groups and len(groups[target_prefix]) >= 4:
        selected_group = groups[target_prefix]
    else:
        for group in groups.values():
            if len(group) >= 4:
                selected_group = group
                break

    # CFD 排序优化：若存在原图（如 025.jpg），置于第一张显示
    if dataset_name == "CFD" and selected_group:
        base_id = extract_base_name(selected_group[0].name, dataset_name)
        selected_group.sort(key=lambda x: (x.stem != base_id, len(x.stem), x.stem))

    return selected_group[:4]


def visualize_augmentation(data_root: Path, out_path: Path) -> None:
    """对比并可视化 3 个数据集的数据增强/衍生切片痕迹"""
    dataset_targets = {
        "crack500": "20160222_081011",
        "AsphaltCrack300": "001",
        "CFD": "025",
    }

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    fig.suptitle("Crack Dataset - Data Augmentation & Crop Trace", fontsize=16, y=0.99)

    for row_idx, dataset_name in enumerate(SUBSETS):
        train_dir = data_root / dataset_name / "train"
        target_prefix = dataset_targets.get(dataset_name)
        samples = pick_augmented_samples(train_dir, dataset_name, target_prefix)

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                img_path = samples[col_idx]
                try:
                    with Image.open(img_path) as img:
                        ax.imshow(img)
                        w, h = img.size
                        title = f"{dataset_name}\n{img_path.name}\n({w}x{h})"
                        ax.set_title(title, fontsize=9)
                except (OSError, UnidentifiedImageError) as e:
                    ax.set_title(f"Error loading:\n{e}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No sample", ha="center", va="center")
            ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[✓] 数据增强/衍生痕迹对比图已保存至: {out_path}")


def main() -> None:
    # 1. 克隆代码仓库
    run_git_clone()

    # 2. 配对与图像尺寸统计
    audit_and_collect_stats(DATA_DIR)

    # 3. 数据增强与裁剪痕迹可视化对比
    out_img = DATA_DIR / "data_enhance.png"
    visualize_augmentation(DATA_DIR, out_img)


if __name__ == "__main__":
    main()
