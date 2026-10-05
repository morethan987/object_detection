"""torchvision 侧训练入口：Faster R-CNN R50-FPN / RetinaNet R50-FPN。

用法（实现后）:
    uv run python train/train_torchvision.py --config configs/models.yaml --models fasterrcnn_r50_fpn

契约（对应 homework.md §10）:
    读取  configs/models.yaml + data/coco/annotations/*.json
    行为  加载 COCO 预训练权重 → 手动替换检测头为 num_classes=2（背景+crack）:
            Faster R-CNN: roi_heads.box_predictor 的 cls_score / bbox_pred
            RetinaNet:    classification_head / regression_head
          SGD lr=0.005 momentum=0.9, cosine 退火 + 500 iter 热身, AMP 混合精度
          每 epoch 在 val 上算 COCO 指标
    输出  runs/<model>/（config.json + best.pt + log.csv: 各 loss 分量 + val mAP50/mAP50:95）
"""
