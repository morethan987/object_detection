"""
ultralytics 侧训练入口：YOLOv8s / YOLOv10s / YOLO11s / RT-DETR-L

uv run python train/train_yolo.py --config configs/models.yaml --models rtdetr_l

读取  configs/models.yaml（唯一超参来源，勿在命令行散落覆盖）
数据  data/crack.yaml（由 tools/export.py 生成）
行为  COCO 预训练权重加载 -> 自动替换检测头为 1 类 -> 训练
      每模型输出 runs/<model>/（args.yaml + weights + results.csv + 曲线图）
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from ultralytics import RTDETR, YOLO


def get_project_root() -> Path:
    """获取项目根目录绝对路径（本脚本位于 train/ 目录下）"""
    return Path(__file__).resolve().parent.parent


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    """安全读取 YAML 配置文件"""
    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_weights_path(project_root: Path, weight_name: str) -> str:
    """
    自动确保预训练权重落盘在 weights/ 目录下:
    1. 确保 weights/ 目录存在
    2. 若 weights/<weight_name> 存在，直接加载
    3. 若不存在，将 weights/<weight_name> 的完整路径传给 Ultralytics，
       框架会自动从官方 Release 下载并直接保存到该目标路径。
    """
    weights_dir = project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    target_path = weights_dir / weight_name
    if target_path.exists():
        print(f"[*] 发现本地预训练权重: {target_path}")
    else:
        print(f"[*] 未检测到权重，Ultralytics 将自动下载并保存至: {target_path}")

    # 直接返回 weights/yolo11s.pt 的绝对路径或相对路径
    return str(target_path)


def instantiate_model(weights: str, model_key: str):
    """根据模型类型实例化 YOLO 或 RTDETR"""
    if "rtdetr" in model_key.lower() or "rtdetr" in weights.lower():
        return RTDETR(weights)
    return YOLO(weights)


def train_single_model(
    model_name: str,
    common_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    data_yaml_path: Path,
    project_root: Path,
) -> None:
    """执行单个模型的训练全流程"""
    # 框架合法性校验
    framework = model_cfg.get("framework")
    if framework != "ultralytics":
        raise ValueError(
            f"模型 [{model_name}] 的 framework 为 '{framework}'，"
            f"不是 ultralytics。请使用 train_torchvision.py 训练此模型。"
        )

    print("\n" + "=" * 70)
    print(f" 开始训练 Ultralytics 模型: {model_name} ".center(70, "="))
    print("=" * 70)

    # 1. 权重准备与模型构建
    raw_weight_name = model_cfg.get("weights")
    if not raw_weight_name:
        raise KeyError(f"模型 [{model_name}] 未在配置中指定 'weights'")

    weights_path = resolve_weights_path(project_root, raw_weight_name)
    model = instantiate_model(weights_path, model_name)

    # 2. 组装 Ultralytics 训练参数
    # 严格对齐 handout §10 与 configs/models.yaml
    train_args: dict[str, Any] = {
        "data": str(data_yaml_path),
        "imgsz": common_cfg.get("imgsz", 640),
        "batch": common_cfg.get("batch", 16),
        "epochs": common_cfg.get("epochs", 150),
        "patience": common_cfg.get("patience", 50),
        "workers": common_cfg.get("workers", 8),
        "seed": common_cfg.get("seed", 0),
        "project": str(project_root / common_cfg.get("project", "runs")),
        "name": model_name,
        "exist_ok": True,  # 保持 runs/<model>/ 目录纯净，覆盖同名重跑目录
        "verbose": True,
        "plots": True,  # 自动导出 PR 曲线、混淆矩阵、F1 曲线等图表
        "save": True,
    }

    # 3. 合并模型私有 extra 训练参数（例如 close_mosaic, deterministic 等）
    extra_args = model_cfg.get("extra", {})
    if isinstance(extra_args, dict):
        train_args.update(extra_args)

    print("\n[训练配置参数汇总]:")
    for k, v in train_args.items():
        print(f"  - {k}: {v}")
    print("-" * 70)

    # 4. 触发训练（Ultralytics 会根据 crack.yaml 的 nc=1 自动重构检测头）
    results = model.train(**train_args)

    print(f"\n[+] 模型 [{model_name}] 训练完成！")
    save_dir = Path(results.save_dir)  # 从 results 中直接获取真实的输出目录
    print(f"[+] 成果保存路径: {save_dir}")
    print(f"    - 最佳权重: {save_dir / 'weights' / 'best.pt'}")
    print(f"    - 训练日志: {save_dir / 'results.csv'}")

    if hasattr(results, "box") and results.box is not None:
        print("[+] 最终验证指标 (Validation):")
        print(f"    - mAP@0.5      : {results.box.map50:.4f}")
        print(f"    - mAP@0.5:0.95 : {results.box.map:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ultralytics 训练入口脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models.yaml",
        help="统一模型配置文件路径（默认: configs/models.yaml）",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11s"],
        choices=["yolo11s", "yolov8s", "yolov10s", "rtdetr_l"],
        help="待训练模型列表，空格分隔（例如: --models yolo11s 或 --models yolo11s rtdetr_l）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = get_project_root()

    # 校验数据集描述文件
    data_yaml_path = project_root / "data" / "crack.yaml"
    if not data_yaml_path.exists():
        sys.exit(
            f"[错误] 数据集配置文件未生成: {data_yaml_path}\n请先运行工具脚本导出数据集: uv run python tools/export.py"
        )

    # 读取超参数配置文件
    config_file_path = (project_root / args.config).resolve()
    full_config = load_yaml(config_file_path)

    common_cfg = full_config.get("common", {})
    models_pool = full_config.get("models", {})

    # 执行所选模型的训练
    for model_name in args.models:
        if model_name not in models_pool:
            print(
                f"[警告] 模型 '{model_name}' 未在 {args.config} 中定义，已跳过！",
                file=sys.stderr,
            )
            continue

        model_cfg = models_pool[model_name]
        train_single_model(
            model_name=model_name,
            common_cfg=common_cfg,
            model_cfg=model_cfg,
            data_yaml_path=data_yaml_path,
            project_root=project_root,
        )


if __name__ == "__main__":
    main()
