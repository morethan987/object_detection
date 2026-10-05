"""参数量 / GFLOPs / 推理时延基准。

用法（实现后）:
    uv run python eval/benchmark.py --config configs/models.yaml --out results/benchmark.json

契约（对应 homework.md §11.2–11.3）:
    Params/GFLOPs  统一 640×640 输入
                   ultralytics: model.info() / torchvision: thop.profile
                   （YOLOv10 需注明“融合后”口径）
    Latency        GPU, batch=1, 预热 50 次 + 计时 300 次取均值±标准差,
                   torch.cuda.synchronize() 掐表, FP32 与 FP16(autocast) 分开报,
                   是否含预处理/NMS 后处理需在输出中注明; FPS = 1000 / latency
    约束           所有模型必须在同一台机器同一状态下测完
"""
