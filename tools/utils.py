from pathlib import Path

DATA_DIR = Path("data/raw/crackDataset")
SUBSETS = ["crack500", "AsphaltCrack300", "CFD"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 数据集子文件夹配对结构
DATASET_PAIRS = {
    "crack500": [("train", "label"), ("val", "val_label")],
    "AsphaltCrack300": [("train", "label")],
    "CFD": [("train", "label"), ("val", "val_label")],
}
