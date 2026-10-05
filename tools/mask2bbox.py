"""mask → bbox：把像素级裂缝掩码转成检测框（本作业预处理的核心）。

用法（实现后）:
    uv run python tools/mask2bbox.py --src data/raw/crackDataset --out data/interim --min-area 64

契约（对应 homework.md §8.1，方案 A：一个连通域一个框）:
    输入  三个子集的 {train,val}/*.jpg 与 {label,val_label}/*.png
    处理  二值化(arr > 0，兼容 1-bit 的 0/1、0/255、调色板索引；禁用 ==255)
          → 闭运算(3×3 核, 1–2 次迭代) → cv2.connectedComponentsWithStats
          → 过滤 area < min-area 的噪点连通域
    输出  data/interim/<dataset>/<name>.json: {"boxes": [[x1,y1,x2,y2], ...]}（绝对像素坐标）
    自检  随机 10 张掩码叠加原图存 data/interim/spot_check/，人工目检二值化方向
"""
