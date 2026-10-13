"""
汇总全部模型的指标 → results/summary.csv（横向对比总表与 Pareto 效率分析）

uv run python eval/summarize.py \
    --metrics results/metrics \
    --bench results/benchmark.json \
    --out results/summary.csv \
    --plot results/figures/speed_accuracy.png
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# 模型到范式的映射关系（对齐 homework.md 与 models.yaml）
MODEL_METADATA: dict[str, dict[str, str]] = {
    "yolov8s": {
        "display_name": "YOLOv8s",
        "paradigm": "One-stage (Anchor-free)",
    },
    "yolov10s": {
        "display_name": "YOLOv10s",
        "paradigm": "One-stage (Anchor-free, NMS-free)",
    },
    "yolo11s": {
        "display_name": "YOLO11s",
        "paradigm": "One-stage (Anchor-free)",
    },
    "fasterrcnn_r50_fpn": {
        "display_name": "Faster R-CNN R50-FPN",
        "paradigm": "Two-stage (Anchor-based)",
    },
    "retinanet_r50_fpn": {
        "display_name": "RetinaNet R50-FPN",
        "paradigm": "One-stage (Anchor-based)",
    },
    "rtdetr_l": {
        "display_name": "RT-DETR-L",
        "paradigm": "Transformer (End-to-end)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="聚合评测结果与基准测试数据，生成横向对比总表与 Pareto 图")
    parser.add_argument(
        "--metrics",
        type=str,
        default="results/metrics",
        help="eval_coco.py 输出的 JSON 指标目录",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default="results/benchmark.json",
        help="benchmark.py 输出的性能参数 JSON 文件路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/summary.csv",
        help="汇总 CSV 文件的导出路径",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="results/figures/speed_accuracy.png",
        help="Speed-Accuracy 帕累托散点图保存路径 (设为 none 可跳过绘图)",
    )
    return parser.parse_args()


def load_benchmark_data(bench_path: Path) -> dict[str, dict[str, Any]]:
    """加载并解析 benchmark.json"""
    if not bench_path.exists():
        print(f"[!] 警告: 未找到 benchmark 文件: {bench_path}，硬件耗时与算力指标将留空。")
        return {}

    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bench_dict = {}
    if isinstance(data, dict):
        bench_dict = data
    elif isinstance(data, list):
        for item in data:
            name = item.get("model") or item.get("name")
            if name:
                bench_dict[name] = item

    return bench_dict


def collect_model_records(
    metrics_dir: Path,
    bench_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """扫描 metrics 目录中的 JSON 并与 benchmark 整合"""
    json_files = list(metrics_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录 {metrics_dir} 下未找到任何评测指标 JSON 文件！")

    records = []
    for metric_file in sorted(json_files):
        model_key = metric_file.stem
        with open(metric_file, "r", encoding="utf-8") as f:
            metric_data = json.load(f)

        coco = metric_data.get("coco_eval_standard", {})
        eng = metric_data.get("engineering_metrics", {})
        b_data = bench_data.get(model_key, {})

        meta = MODEL_METADATA.get(
            model_key,
            {
                "display_name": model_key,
                "paradigm": "Unknown",
            },
        )

        def to_pct(val):
            return round(val * 100, 2) if val is not None else None

        # 从 benchmark.json 的结构中直接提取
        fp32_info = b_data.get("fp32", {})
        fp16_info = b_data.get("fp16", {})

        record = {
            "model_key": model_key,
            "model_name": meta["display_name"],
            "paradigm": meta["paradigm"],
            # 学术指标 (转换为百分比 0~100)
            "mAP_50": to_pct(coco.get("mAP_50")),
            "mAP_50_95": to_pct(coco.get("mAP_50_95")),
            "mAP_75": to_pct(coco.get("mAP_75")),
            "AP_s": to_pct(coco.get("AP_s")),
            "AP_m": to_pct(coco.get("AP_m")),
            "AP_l": to_pct(coco.get("AP_l")),
            # 工程指标 (转换为百分比 0~100)
            "precision": to_pct(eng.get("precision")),
            "recall": to_pct(eng.get("recall")),
            "f1": to_pct(eng.get("f1")),
            "tp": eng.get("tp"),
            "fp": eng.get("fp"),
            "fn": eng.get("fn"),
            # 性能基准指标
            "params_m": b_data.get("params_m"),
            "gflops": b_data.get("gflops"),
            "latency_fp32_ms": fp32_info.get("mean_ms"),
            "latency_fp16_ms": fp16_info.get("mean_ms"),
            "fps_fp16": fp16_info.get("fps"),
        }
        records.append(record)

    return records


def print_terminal_table(records: list[dict[str, Any]]):
    """在终端格式化输出对齐的对比表格"""
    headers = [
        "Model",
        "Paradigm",
        "mAP@50",
        "mAP@50:95",
        "P (%)",
        "R (%)",
        "F1 (%)",
        "Params(M)",
        "GFLOPs",
        "Lat-fp16(ms)",
        "FPS(fp16)",
    ]

    rows = []
    for r in records:
        rows.append(
            [
                r["model_name"],
                r["paradigm"].split()[0],  # 截取主要范式类型保持紧凑
                f"{r['mAP_50']:.1f}%" if r["mAP_50"] is not None else "-",
                f"{r['mAP_50_95']:.1f}%" if r["mAP_50_95"] is not None else "-",
                f"{r['precision']:.1f}%" if r["precision"] is not None else "-",
                f"{r['recall']:.1f}%" if r["recall"] is not None else "-",
                f"{r['f1']:.1f}%" if r["f1"] is not None else "-",
                f"{r['params_m']:.2f}" if isinstance(r["params_m"], (int, float)) else "-",
                f"{r['gflops']:.1f}" if isinstance(r["gflops"], (int, float)) else "-",
                f"{r['latency_fp16_ms']:.2f}" if isinstance(r["latency_fp16_ms"], (int, float)) else "-",
                f"{r['fps_fp16']:.1f}" if isinstance(r["fps_fp16"], (int, float)) else "-",
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    sep_line = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_line = "| " + " | ".join([f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)]) + " |"

    print("\n" + "=" * len(sep_line))
    print(" 各架构模型统一评测与效率对比汇总 ".center(len(sep_line) - 2))
    print(sep_line)
    print(header_line)
    print(sep_line)
    for row in rows:
        line = "| " + " | ".join([f"{v!s:<{col_widths[i]}}" for i, v in enumerate(row)]) + " |"
        print(line)
    print(sep_line + "\n")


def write_summary_csv(records: list[dict[str, Any]], out_path: Path):
    """将汇总记录写入 CSV 文件"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_key",
        "model_name",
        "paradigm",
        "mAP_50",
        "mAP_50_95",
        "mAP_75",
        "AP_s",
        "AP_m",
        "AP_l",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "params_m",
        "gflops",
        "latency_fp32_ms",
        "latency_fp16_ms",
        "fps_fp16",
    ]

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"[+] 汇总对比表已成功导出至: {out_path}")


def plot_pareto_curve(records: list[dict[str, Any]], plot_path: Path):
    """
    绘制 Speed (Latency fp16) vs Accuracy (mAP@0.5:0.95) 散点图并标记 Pareto 前沿
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] 未安装 matplotlib，跳过绘制 Speed-Accuracy 散点图。")
        return

    # 提取有效数据点 (以 latency_fp16_ms 或 latency_fp32_ms 为横轴，mAP_50_95 为纵轴)
    valid_points = []
    for r in records:
        lat = r.get("latency_fp16_ms") or r.get("latency_fp32_ms")
        acc = r.get("mAP_50_95")
        if lat is not None and acc is not None:
            valid_points.append(
                {
                    "name": r["model_name"],
                    "paradigm": r["paradigm"],
                    "latency": float(lat),
                    "map": float(acc),
                }
            )

    if len(valid_points) < 2:
        print("[!] 具备时延与精度数据的模型不足两个，跳过 Pareto 曲线绘制。")
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # 范式颜色编码
    color_map = {
        "One-stage (Anchor-free)": "#1f77b4",
        "One-stage (Anchor-free, NMS-free)": "#2ca02c",
        "Two-stage (Anchor-based)": "#ff7f0e",
        "One-stage (Anchor-based)": "#d62728",
        "Transformer (End-to-end)": "#9467bd",
    }

    plt.figure(figsize=(10, 6), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.5)

    # 绘制各模型散点
    for pt in valid_points:
        color = color_map.get(pt["paradigm"], "#7f7f7f")
        plt.scatter(pt["latency"], pt["map"], color=color, s=110, edgecolors="black", zorder=4)
        plt.annotate(
            str(pt["name"]),
            (pt["latency"], pt["map"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            fontweight="bold",
        )

    # 计算 Pareto 前沿点 (要求：时延更低且精度更高)
    sorted_pts = sorted(valid_points, key=lambda x: x["latency"])
    pareto_front = []
    max_acc = -1.0
    for p in sorted_pts:
        if p["map"] > max_acc:
            pareto_front.append(p)
            max_acc = p["map"]

    if len(pareto_front) >= 2:
        fx = [p["latency"] for p in pareto_front]
        fy = [p["map"] for p in pareto_front]
        plt.plot(fx, fy, linestyle="--", color="crimson", linewidth=1.8, label="Pareto Frontier", zorder=3)

    plt.xlabel("Inference Latency @ FP16 (ms / batch=1)", fontsize=11)
    plt.ylabel("Accuracy: mAP@0.5:0.95 (%)", fontsize=11)
    plt.title("Speed vs Accuracy Comparison on Crack Test Set", fontsize=13, pad=12)

    # 添加范式图例
    handles = []
    labels = []
    for paradigm, c in color_map.items():
        if any(p["paradigm"] == paradigm for p in valid_points):
            handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8))
            labels.append(paradigm)

    if len(pareto_front) >= 2:
        handles.append(plt.Line2D([0], [0], linestyle="--", color="crimson", linewidth=2))
        labels.append("Pareto Frontier")

    plt.legend(handles, labels, loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Speed-Accuracy 散点图已保存至: {plot_path}")


def main():
    args = parse_args()

    metrics_dir = Path(args.metrics).resolve()
    bench_file = Path(args.bench).resolve()
    out_csv = Path(args.out).resolve()

    if not metrics_dir.exists():
        raise FileNotFoundError(f"指标目录不存在: {metrics_dir}")

    # 1. 加载并整合数据
    bench_data = load_benchmark_data(bench_file)
    records = collect_model_records(metrics_dir, bench_data)

    # 2. 导出 CSV 汇总
    write_summary_csv(records, out_csv)

    # 3. 打印终端对齐表格
    print_terminal_table(records)

    # 4. 绘制 Speed-Accuracy / Pareto 曲线
    if args.plot and args.plot.lower() != "none":
        plot_path = Path(args.plot).resolve()
        plot_pareto_curve(records, plot_path)


if __name__ == "__main__":
    main()
