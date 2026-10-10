"""
参数量 / GFLOPs / 推理时延基准测试脚本

用法示例:
    uv run python eval/benchmark.py --config configs/models.yaml --out results/benchmark.json

口径说明:
    1. Params / GFLOPs: 统一 640x640 输入尺寸
       - Ultralytics 侧: 模型原生 info(imgsz=640)，YOLOv10 采用融合后单预测头口径
       - Torchvision 侧: thop.profile 统计
    2. Latency / FPS:
       - 硬件: 单 GPU, Batch=1, 预热 50 次 + 计时 300 次计算 mean ± std
       - 精度: FP32 与 FP16 (autocast) 分开测定
       - FPS: 1000.0 / latency_ms
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import RTDETR, YOLO


# ==============================================================================
# 1. 核心计时工具（纯粹、直接）
# ==============================================================================
def measure_latency(forward_fn, use_fp16: bool, warmup: int = 50, runs: int = 300) -> dict[str, float]:
    """使用 CUDA Event 掐表测定纯前向耗时 (ms) 与 FPS"""
    # 预热
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            forward_fn()
    torch.cuda.synchronize()

    # 正式计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []

    for _ in range(runs):
        start.record()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            forward_fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    return {
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


# ==============================================================================
# 2. Ultralytics 侧测定 (YOLOv8s / YOLOv10s / YOLO11s / RT-DETR-L)
# ==============================================================================
def benchmark_ultralytics(name: str, root: Path, device: torch.device, imgsz: int = 640):
    best_weight = root / f"runs/{name}/weights/best.pt"
    raw_weight = root / f"weights/{name}.pt"
    weight_path = str(best_weight if best_weight.exists() else raw_weight)

    # 1. 加载模型并读取 Params 与 GFLOPs
    model = RTDETR(weight_path) if "rtdetr" in name else YOLO(weight_path)
    _, params, _, gflops = model.info(detailed=False, imgsz=imgsz)
    model.fuse()

    # 2. 获取底层真实 PyTorch 网络（一行断言满足 LSP 类型安全）
    assert isinstance(model.model, torch.nn.Module), f"{name} 底层网络初始化失败"
    net = model.model.to(device).eval()

    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    def forward_fn():
        with torch.inference_mode():
            return net(dummy)

    return {
        "params_m": round(params / 1e6, 2),
        "gflops": round(float(gflops), 2),
        "fp32": measure_latency(forward_fn, use_fp16=False),
        "fp16": measure_latency(forward_fn, use_fp16=True),
        "notes": "原生 forward; 融合算子; 无外部 CPU-NMS",
    }


# ==============================================================================
# 3. Torchvision 侧测定 (Faster R-CNN / RetinaNet)
# ==============================================================================
def benchmark_torchvision(name: str, root: Path, device: torch.device, imgsz: int = 640):
    if "fasterrcnn" in name:
        net = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_feat = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
    else:
        net = detection.retinanet_resnet50_fpn(weights="DEFAULT", num_classes=2)

    # 若有微调好的权重则加载
    ckpt = root / f"runs/{name}/best.pth"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(state.get("model", state))

    net = net.to(device).eval()

    # 统计参数量与 GFLOPs
    params_m = round(sum(p.numel() for p in net.parameters()) / 1e6, 2)
    dummy_input = [torch.randn(3, imgsz, imgsz, device=device)]

    gflops = None
    try:
        import thop

        flops, _ = thop.profile(net, inputs=(dummy_input,), verbose=False)
        gflops = round(flops / 1e9, 2)
    except (ImportError, RuntimeError):
        pass

    def forward_fn():
        with torch.inference_mode():
            return net(dummy_input)

    return {
        "params_m": params_m,
        "gflops": gflops,
        "fp32": measure_latency(forward_fn, use_fp16=False),
        "fp16": measure_latency(forward_fn, use_fp16=True),
        "notes": "torchvision 架构; forward 内含 RPN 与 RoI-NMS",
    }


# ==============================================================================
# 4. 主流程与表格展示
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="多模型基准性能测试")
    parser.add_argument("--config", default="configs/models.yaml", help="配置文件路径")
    parser.add_argument("--out", default="results/benchmark.json", help="输出路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_file = (root / args.config).resolve()
    if not cfg_file.exists():
        sys.exit(f"找不到配置文件: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        models = yaml.safe_load(f).get("models", {})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] 测试设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    results = {}
    for name, cfg in models.items():
        print(f"[*] 正在评测: {name} ...")
        if cfg.get("framework") == "ultralytics":
            results[name] = benchmark_ultralytics(name, root, device, args.imgsz)
        elif cfg.get("framework") == "torchvision":
            results[name] = benchmark_torchvision(name, root, device, args.imgsz)

    # 打印排版表格
    print("\n" + "=" * 92)
    header = f"| {'Model':<20} | {'Params (M)':<10} | {'GFLOPs':<8} | {'FP32 (ms)':<10} | {'FP32 FPS':<9} | {'FP16 (ms)':<10} | {'FP16 FPS':<9} |"
    print(header)
    print(
        "|"
        + "-" * 22
        + "|"
        + "-" * 12
        + "|"
        + "-" * 10
        + "|"
        + "-" * 12
        + "|"
        + "-" * 11
        + "|"
        + "-" * 12
        + "|"
        + "-" * 11
        + "|"
    )

    for name, r in results.items():
        gflops_str = f"{r['gflops']:.2f}" if r["gflops"] else "N/A"
        fp32_str = f"{r['fp32']['mean_ms']:.2f}±{r['fp32']['std_ms']:.2f}"
        fp16_str = f"{r['fp16']['mean_ms']:.2f}±{r['fp16']['std_ms']:.2f}"
        row = f"| {name:<20} | {r['params_m']:<10.2f} | {gflops_str:<8} | {fp32_str:<10} | {r['fp32']['fps']:<9.1f} | {fp16_str:<10} | {r['fp16']['fps']:<9.1f} |"
        print(row)
    print("=" * 92 + "\n")

    # 写入 JSON
    out_file = (root / args.out).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] 结果已保存至: {out_file}")


if __name__ == "__main__":
    main()
