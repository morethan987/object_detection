"""统一指标评测：所有模型的预测转 COCO json，走同一份 COCOeval。

用法（实现后）:
    uv run python eval/eval_coco.py --gt data/coco/annotations/test.json \
        --pred runs/<model>/predict_coco.json --out results/metrics/<model>.json

契约（对应 homework.md §11.1，公平性命门）:
    口径  pycocotools COCOeval, max_det=300, score 阈值 0.05 起评（COCO 标准）
    产出  mAP@0.5 / mAP@0.5:0.95 / AP_S / AP_M / AP_L
          另报 conf=0.25 应用阈值下的 Precision / Recall（工程口径）
    约束  六个模型必须用同一份 test 标注、同一组参数跑——不使用各框架自带口径
"""
