"""导出双格式数据集：YOLO 目录结构 + COCO json + crack.yaml。

用法（实现后）:
    uv run python tools/export.py --interim data/interim --splits splits --out data

契约（对应 homework.md §8.4）:
    输入  data/interim（mask2bbox 产物）+ splits/*.txt
    输出  data/yolo/{images,labels}/{train,val,test}/   （YOLO txt: xc,yc,w,h 归一化）
          data/coco/annotations/{train,val,test}.json   （COCO: [x,y,w,h] 绝对像素）
          data/crack.yaml                              （ultralytics 数据集配置, names: ['crack']）
    细节  转框后为空的 tile 保留为背景图（YOLO 侧写空 label 文件）
"""
