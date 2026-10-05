"""按“原图 ID”分组划分 train/val/test = 7:1:2（防 tile/增广泄漏的关键步骤）。

用法（实现后）:
    uv run python tools/split.py --interim data/interim --out splits --seed 0

契约（对应 homework.md §8.3）:
    分组键  crack500: ^(.*?)_\d+_\d+$ 取原图名
            AsphaltCrack300: ^(\d+)_ 取原图号
            CFD: ^(\d+) 取原图号（add/sub/adjust/翻转变体归入同一组）
    划分    组级 7:1:2，按三个子集分层抽样，seed 固定
    输出    splits/{train,val,test}.txt（每行一个 tile 的相对路径）
            划分摘要打印各 split 的 tile 数与原图组数，供写进报告
"""
