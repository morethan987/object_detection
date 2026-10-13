"""
参数量 / GFLOPs / 推理时延基准测试脚本 (支持 FP32 / FP16 原生 / BF16 原生)

uv run python eval/benchmark.py --config configs/models.yaml --out results/benchmark.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from ultralytics import RTDETR, YOLO


# ==============================================================================
# 1. 核心纯净计时工具
# ==============================================================================
def measure_latency(forward_fn, warmup: int = 50, runs: int = 300) -> dict[str, float]:
    """使用 CUDA Event 掐表测定纯前向耗时 (ms) 与 FPS，杜绝任何外部上下文干扰"""
    # 预热
    for _ in range(warmup):
        forward_fn()
    torch.cuda.synchronize()

    # 正式计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []

    for _ in range(runs):
        start.record()
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

    assert isinstance(model.model, torch.nn.Module), f"{name} 底层网络初始化失败"
    net_fp32 = model.model.to(device).eval()

    # 2. 原生 FP32 测定
    dummy_fp32 = torch.rand(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)

    def forward_fp32():
        with torch.inference_mode():
            return net_fp32(dummy_fp32)

    # 3. 原生 FP16 (直接半精度化，激活 Tensor Core 极限)
    net_fp16 = copy.deepcopy(net_fp32).half()
    dummy_fp16 = dummy_fp32.half()

    def forward_fp16():
        with torch.inference_mode():
            return net_fp16(dummy_fp16)

    # 4. 原生 BF16 (对比测试)
    net_bf16 = copy.deepcopy(net_fp32).bfloat16()
    dummy_bf16 = dummy_fp32.bfloat16()

    def forward_bf16():
        with torch.inference_mode():
            return net_bf16(dummy_bf16)

    return {
        "params_m": round(params / 1e6, 2),
        "gflops": round(float(gflops), 2),
        "fp32": measure_latency(forward_fp32),
        "fp16": measure_latency(forward_fp16),
        "bf16": measure_latency(forward_bf16),
        "notes": "原生前向; 融合算子; 纯静态显式转换",
    }


# ==============================================================================
# 3. Torchvision 侧测定 (Faster R-CNN / RetinaNet)
# ==============================================================================
def benchmark_torchvision(name: str, root: Path, device: torch.device, imgsz: int = 640):
    if "fasterrcnn" in name:
        net = detection.fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=imgsz,
            max_size=imgsz,
        )
        in_feat = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
    elif "retinanet" in name:
        net = detection.retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=imgsz,
            max_size=imgsz,
        )
        num_anchors = net.head.classification_head.num_anchors
        in_channels = net.backbone.out_channels
        net.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=2,
        )
    else:
        raise ValueError(f"未知的 torchvision 模型: {name}")

    weight_candidates = [
        root / f"runs/{name}/weights/best.pt",
        root / f"runs/{name}/weights/best.pth",
        root / f"weights/{name}.pt",
        root / f"weights/{name}.pth",
    ]
    ckpt_path = next((p for p in weight_candidates if p.exists()), None)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        net.load_state_dict(state_dict)
        print(f"[*] 成功加载本地权重: {ckpt_path.relative_to(root)}")

    net_fp32 = net.to(device).eval()

    # 统计参数与 GFLOPs
    params_m = round(sum(p.numel() for p in net_fp32.parameters()) / 1e6, 2)
    dummy_fp32 = [torch.rand(3, imgsz, imgsz, device=device, dtype=torch.float32)]

    gflops = None
    try:
        import thop

        flops, _ = thop.profile(net_fp32, inputs=(dummy_fp32,), verbose=False)
        gflops = round(flops / 1e9, 2)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[!] 提示: {name} GFLOPs 统计跳过 ({exc})")

    # 1. FP32 测定
    def forward_fp32():
        with torch.inference_mode():
            return net_fp32(dummy_fp32)

    # 2. FP16 测定 (支持静态与 autocast 稳健回退)
    try:
        net_fp16 = copy.deepcopy(net_fp32).half()
        dummy_fp16 = [img.half() for img in dummy_fp32]
        with torch.inference_mode():
            _ = net_fp16(dummy_fp16)

        def forward_fp16():
            with torch.inference_mode():
                return net_fp16(dummy_fp16)

    except Exception:
        # 若 torchvision 某些特异算子不支持纯 half，则在外层开启 autocast
        def forward_fp16():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return net_fp32(dummy_fp32)

    # 3. BF16 测定
    try:
        net_bf16 = copy.deepcopy(net_fp32).bfloat16()
        dummy_bf16 = [img.bfloat16() for img in dummy_fp32]
        with torch.inference_mode():
            _ = net_bf16(dummy_bf16)

        def forward_bf16():
            with torch.inference_mode():
                return net_bf16(dummy_bf16)

    except Exception:

        def forward_bf16():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                return net_fp32(dummy_fp32)

    return {
        "params_m": params_m,
        "gflops": gflops,
        "fp32": measure_latency(forward_fp32),
        "fp16": measure_latency(forward_fp16),
        "bf16": measure_latency(forward_bf16),
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
    print("\n" + "=" * 115)
    header = (
        f"| {'Model':<20} | {'Params(M)':<9} | {'GFLOPs':<7} | "
        f"{'FP32(ms)':<9} | {'FP16(ms)':<9} | {'FP16 FPS':<9} | "
        f"{'BF16(ms)':<9} | {'BF16 FPS':<9} |"
    )
    print(header)
    print(
        "|"
        + "-" * 22
        + "|"
        + "-" * 11
        + "|"
        + "-" * 9
        + "|"
        + "-" * 11
        + "|"
        + "-" * 11
        + "|"
        + "-" * 11
        + "|"
        + "-" * 11
        + "|"
        + "-" * 11
        + "|"
    )

    for name, r in results.items():
        gflops_str = f"{r['gflops']:.2f}" if r["gflops"] else "N/A"
        fp32_str = f"{r['fp32']['mean_ms']:.2f}"
        fp16_str = f"{r['fp16']['mean_ms']:.2f}"
        bf16_str = f"{r['bf16']['mean_ms']:.2f}"
        row = (
            f"| {name:<20} | {r['params_m']:<9.2f} | {gflops_str:<7} | "
            f"{fp32_str:<9} | {fp16_str:<9} | {r['fp16']['fps']:<9.1f} | "
            f"{bf16_str:<9} | {r['bf16']['fps']:<9.1f} |"
        )
        print(row)
    print("=" * 115 + "\n")

    # 写入 JSON
    out_file = (root / args.out).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] 结果已保存至: {out_file}")


if __name__ == "__main__":
    main()
