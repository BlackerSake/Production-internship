from __future__ import print_function
import logging
import cv2
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from src.backend.PedestrianDetectorSystem import PedestrianDetectorSystem

# 超参


VEDIO_PATH = PROJECT_ROOT / "assets" / "lane_detection.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vedio"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"

def process_vedio(vedio_path=VEDIO_PATH):
    vedio_path = Path(vedio_path)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")

    detector = PedestrianDetectorSystem(MODEL_PATH)

    # 打开视频文件
    cap = cv2.VideoCapture(str(vedio_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{vedio_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 获取视频的帧率和尺寸
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 使用 mp4v 编码
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = OUTPUT_DIR / f"output_{vedio_path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc, 
        fps, 
        (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建结果视频：{output_path}")

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = detector.detect(frame)
        result_frame = detector.draw_detections(frame, detections)

        writer.write(result_frame)
        frame_count += 1

    cap.release()
    writer.release()
    logging.info("处理完成，输出视频保存为: %s", output_path)
    return output_path, frame_count

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型不存在，请将 YOLOv8 ONNX 模型放入：{MODEL_PATH}"
        )

    process_vedio()

if __name__ == "__main__":
    main()
