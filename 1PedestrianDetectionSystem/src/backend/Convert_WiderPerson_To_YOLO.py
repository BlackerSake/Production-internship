import logging
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "assets" / "WiderPerson" / "Images"
ANNOTATION_DIR = PROJECT_ROOT / "assets" / "WiderPerson" / "Annotations"
LABEL_DIR = PROJECT_ROOT / "outputs" / "data" / "labels"
CONVERT_LIMIT = None

# WiderPerson 中所有有效的人物类别统一作为 YOLO 的 person 类别 0。
CLASS_MAPPING = {1: 0, 2: 0, 3: 0, 5: 0}


def image_size(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图片：{image_path}")
    height, width = image.shape[:2]
    return width, height


def convert_annotation(annotation_path, image_path, label_path):
    image_width, image_height = image_size(image_path)
    lines = annotation_path.read_text(encoding="utf-8").splitlines()
    labels = []

    for line in lines[1:]:
        values = line.split()
        if len(values) != 5:
            continue

        original_class, x1, y1, x2, y2 = map(int, values)
        if original_class not in CLASS_MAPPING:
            continue

        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        if x2 <= x1 or y2 <= y1:
            continue

        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height
        labels.append(
            f"{CLASS_MAPPING[original_class]} {center_x:.6f} "
            f"{center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return len(labels)


def convert_dataset():
    annotation_paths = sorted(ANNOTATION_DIR.glob("*.jpg.txt"))
    if CONVERT_LIMIT is not None:
        annotation_paths = annotation_paths[:CONVERT_LIMIT]
    if not annotation_paths:
        raise FileNotFoundError(f"未找到标注文件：{ANNOTATION_DIR}")

    converted = 0
    boxes = 0
    for annotation_path in annotation_paths:
        image_path = IMAGE_DIR / annotation_path.name.removesuffix(".txt")
        if not image_path.exists():
            logger.warning(f"跳过缺失图片：{image_path.name}")
            continue

        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        boxes += convert_annotation(annotation_path, image_path, label_path)
        converted += 1

    return converted, boxes


if __name__ == "__main__":
    converted_count, box_count = convert_dataset()
    logger.info(f"转换完成：{converted_count} 张图片，{box_count} 个标注框")
    logger.info(f"YOLO 标注目录：{LABEL_DIR}")
