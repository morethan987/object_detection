# tools/split.py
r"""
按“原图 ID”分组划分 train/val/test = 7:1:2，防 tile/增广泄漏

uv run python tools/split.py --interim data/interim --out splits --seed 0

分组键  crack500: ^(.*?)_\d+_\d+$ 取原图名
        AsphaltCrack300: ^(\d+)_ 取原图号
        CFD: ^(\d+) 取原图号 (add/sub/adjust/翻转变体归入同一组)
划分    组级 7:1:2，按三个子集分层抽样，seed 固定
输出    splits/{train,val,test}.txt (每行一个 tile 的相对路径)
        划分摘要打印各 split 的 tile 数与原图组数，供写进报告
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from utils import SUBSETS

# 正则表达式提取"原图分组键"
GROUP_PATTERNS = {
    "crack500": re.compile(r"^(.*?)_\d+_\d+$"),
    "AsphaltCrack300": re.compile(r"^(\d+)_"),
    "CFD": re.compile(r"^(\d+)"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="按原图 ID 分组划分训练/验证/测试集，防止数据泄漏。")
    parser.add_argument(
        "--interim",
        type=Path,
        default=Path("data/interim"),
        help="中间数据目录（含各子集的 JSON 标签或图像）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("splits"),
        help="划分输出目录（存放 train.txt, val.txt, test.txt）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机数种子，保证划分可复现",
    )
    return parser.parse_args()


def extract_group_id(subset_name: str, stem: str) -> str:
    """根据子集规则提取原图组 ID；若不匹配则降级为 stem 本身独立成组"""
    pattern = GROUP_PATTERNS.get(subset_name)
    if pattern:
        match = pattern.match(stem)
        if match:
            return match.group(1)
    return stem


def split_groups(groups: list[str], seed: int, ratio=(0.7, 0.1, 0.2)):
    """对组列表进行 7:1:2 划分"""
    rng = random.Random(seed)
    shuffled = sorted(groups)  # 先排序再打乱，确保完全确定性
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * ratio[0])
    n_val = round(n * ratio[1])

    # 对极端小样本做边界保底
    if n >= 10:
        n_val = max(1, n_val)
    _ = n - n_train - n_val

    train_g = shuffled[:n_train]
    val_g = shuffled[n_train : n_train + n_val]
    test_g = shuffled[n_train + n_val :]

    return train_g, val_g, test_g


def main():
    args = parse_args()
    interim_dir = args.interim
    out_dir = args.out
    seed = args.seed

    out_dir.mkdir(parents=True, exist_ok=True)

    split_results = {"train": [], "val": [], "test": []}
    # 统计数据结构：summary[subset] = {split: {"groups": X, "tiles": Y}}
    summary = defaultdict(lambda: {s: {"groups": 0, "tiles": 0} for s in ["train", "val", "test"]})

    for subset in SUBSETS:
        subset_dir = interim_dir / subset
        if not subset_dir.exists():
            print(f"[WARN] 子集目录不存在: {subset_dir}，跳过")
            continue

        # 扫描该子集下的所有样本标注文件 (.json)
        sample_files = sorted(subset_dir.glob("*.json"))
        if not sample_files:
            # 兼容如果后续存的是图片
            sample_files = sorted([p for p in subset_dir.iterdir() if p.is_file()])

        # 按原图 ID 进行分组收集
        group_to_tiles = defaultdict(list)
        for file_path in sample_files:
            group_id = extract_group_id(subset, file_path.stem)
            rel_path = file_path.relative_to(interim_dir).as_posix()
            group_to_tiles[group_id].append(rel_path)

        all_groups = list(group_to_tiles.keys())
        train_g, val_g, test_g = split_groups(all_groups, seed=seed)

        for split_name, g_list in [("train", train_g), ("val", val_g), ("test", test_g)]:
            summary[subset][split_name]["groups"] = len(g_list)
            for gid in g_list:
                tiles = group_to_tiles[gid]
                split_results[split_name].extend(tiles)
                summary[subset][split_name]["tiles"] += len(tiles)

    # 写入输出文本文件
    for split_name, paths in split_results.items():
        out_txt = out_dir / f"{split_name}.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.writelines(f"{p}\n" for p in paths)

    # 打印排版整齐的划分摘要
    print("\n" + "=" * 90)
    print(f"数据划分摘要报告 (Seed: {seed}, 目标比例 Train:Val:Test = 7:1:2)")
    print("=" * 90)
    print(
        f"{'子集名称 (Subset)':<20} "
        f"{'Train (组/切片)':<20} "
        f"{'Val (组/切片)':<20} "
        f"{'Test (组/切片)':<20} "
        f"{'总计 (组/切片)':<15}"
    )
    print("-" * 90)

    total_groups = {"train": 0, "val": 0, "test": 0, "all": 0}
    total_tiles = {"train": 0, "val": 0, "test": 0, "all": 0}

    for subset in SUBSETS:
        if subset not in summary:
            continue
        st = summary[subset]
        sub_all_g = sum(st[s]["groups"] for s in ["train", "val", "test"])
        sub_all_t = sum(st[s]["tiles"] for s in ["train", "val", "test"])

        for s in ["train", "val", "test"]:
            total_groups[s] += st[s]["groups"]
            total_tiles[s] += st[s]["tiles"]
        total_groups["all"] += sub_all_g
        total_tiles["all"] += sub_all_t

        train_str = f"{st['train']['groups']} / {st['train']['tiles']}"
        val_str = f"{st['val']['groups']} / {st['val']['tiles']}"
        test_str = f"{st['test']['groups']} / {st['test']['tiles']}"
        all_str = f"{sub_all_g} / {sub_all_t}"

        print(f"{subset:<20} {train_str:<20} {val_str:<20} {test_str:<20} {all_str:<15}")

    print("-" * 90)
    tot_train_str = f"{total_groups['train']} / {total_tiles['train']}"
    tot_val_str = f"{total_groups['val']} / {total_tiles['val']}"
    tot_test_str = f"{total_groups['test']} / {total_tiles['test']}"
    tot_all_str = f"{total_groups['all']} / {total_tiles['all']}"

    print(f"{'总计 (Total)':<20} {tot_train_str:<20} {tot_val_str:<20} {tot_test_str:<20} {tot_all_str:<15}")
    print("=" * 90)
    print(f"[INFO] 划分列表已生成至: {out_dir}/{{train,val,test}}.txt\n")


if __name__ == "__main__":
    main()
