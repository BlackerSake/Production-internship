from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.backend.PedestrianDetectorSystem import PedestrianDetectorSystem


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detections"


@lru_cache(maxsize=1)
def get_detector():
    return PedestrianDetectorSystem(MODEL_PATH)


def process_image(image: np.ndarray):
    if image is None:
        raise gr.Error("请先上传图片")

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detector = get_detector()
    detections = detector.detect(image_bgr)
    result_bgr = detector.draw_detections(image_bgr, detections)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "gradio_result.jpg"
    cv2.imwrite(str(output_path), result_bgr)

    return result_rgb, f"检测完成：{len(detections)} 人，结果已保存到 `{output_path}`"


def build_UI():
    gr.Markdown("## 行人识别系统")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="输入图片",
                type="numpy",
            )
            process_button = gr.Button("开始检测", variant="primary")
        
        with gr.Column():
            result_image = gr.Image(
                label="检测结果",
                type="numpy",
            )
            status_text = gr.Markdown("状态：等待检测")

    process_button.click( # type: ignore
        fn=process_image,
        inputs=input_image,
        outputs=[result_image, status_text],
    )
