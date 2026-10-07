"""
ultralytics 侧训练入口：YOLOv8s / YOLOv10s / YOLO11s / RT-DETR-L

uv run python train/train_yolo.py --config configs/models.yaml --models yolo11s rtdetr_l

读取  configs/models.yaml（唯一超参来源，勿在命令行散落覆盖）
数据  data/crack.yaml（由 tools/export.py 生成）
行为  COCO 预训练权重加载 -> 自动替换检测头为 1 类 -> 训练
      每模型输出 runs/<model>/（args.yaml + weights + results.csv + 曲线图）
"""
