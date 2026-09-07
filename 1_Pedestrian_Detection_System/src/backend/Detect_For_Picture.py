
from __future__ import print_function
import numpy as np
import cv2
import os
from PedestrianDetectorSystem import PedestrianDetectorSystem

# 日志配置
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

# 超参
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR = PROJECT_ROOT / "assets" / "WiderPerson" / "Images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detections"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"

INPUT_SIZE = 640 # 输入图像大小
CONFIDENCE_THRESHOLD = 0.3 # 置信度阈值
NMS_THRESHOLD = 0.4 # 非极大值抑制阈值
PROCESS_IMAGE_COUNT = 10 # 按文件名排序后处理前多少张图片

# 加载单张图片
def load_image(image_path):
    if not os.path.exists(image_path):
        logger.error(f"图像不存在:{image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"无法读取图像:{image_path}")
    return image

# 检测行人


def validate_image(image: np.ndarray, name: str = "图像") -> np.ndarray:
    """
    验证图像是否有效
    """
    if image is None:
        raise RuntimeError(f"{name}为 None，无法处理")
    
    if not isinstance(image, np.ndarray):
        raise RuntimeError(f"{name}类型错误: 期望 np.ndarray，实际 {type(image)}")
    
    if image.size == 0:
        raise RuntimeError(f"{name}为空数组，无法处理")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise RuntimeError(f"{name}通道数错误: 期望 3 通道，实际 {image.shape}")
    
    return image

def main():
    if PROCESS_IMAGE_COUNT <= 0:
        raise ValueError("检测图像数量 须大于 0")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型不存在，请将 YOLOv8 ONNX 模型放入：{MODEL_PATH}"
        )

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))[:PROCESS_IMAGE_COUNT]
    if not image_paths:
        raise FileNotFoundError(f"未在目录中找到 JPG 图片：{IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = PedestrianDetectorSystem(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE,
        conf_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD
    )

    for image_path in image_paths:
        image = load_image(str(image_path))
        if image is None:
            continue

        detections = detector.detect(image)
        result = detector.draw_detections(image, detections)
        output_path = OUTPUT_DIR / f"{image_path.stem}_result.jpg"

        if not cv2.imwrite(str(output_path), result):
            raise RuntimeError(f"检测结果保存失败：{output_path}")

        logger.info(f"{image_path.name}: 检测到 {len(detections)} 人，结果：{output_path}")


if __name__ == "__main__":
    main()




