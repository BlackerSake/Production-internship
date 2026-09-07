from __future__ import print_function
import logging
import cv2
from PedestrianDetectorSystem import PedestrianDetectorSystem

# 超参
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VEDIO_PATH = PROJECT_ROOT / "assets" / "lane_detection.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vedio"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"

def process_vedio(vedio_path, 
                  model_path, 
                  output_dir):
    detector = PedestrianDetectorSystem(model_path)

    # 打开视频文件
    cap = cv2.VideoCapture(vedio_path)
    if not cap.isOpened():
        logging.info(f"无法打开视频文件: {vedio_path}")
        return

    writer = None
    # 获取视频的帧率和尺寸
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 使用 mp4v 编码
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_dir / f"output_{vedio_path.stem}.mp4"), 
        fourcc, 
        fps, 
        (width, height)
    )

    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = detector.detect(frame)
        result_frame = detector.draw_detections(frame, detections)

        if writer is not None:
            writer.write(result_frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logging.info(f"处理完成，输出视频保存为: {output_dir / f"output_{vedio_path.stem}.mp4"}")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型不存在，请将 YOLOv8 ONNX 模型放入：{MODEL_PATH}"
        )

    process_vedio(VEDIO_PATH, MODEL_PATH, OUTPUT_DIR)

if __name__ == "__main__":
    main()