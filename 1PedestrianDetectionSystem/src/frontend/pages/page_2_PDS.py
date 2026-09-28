from pathlib import Path
import sys

import cv2
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.backend.Detect_For_Picture import process_images
from src.backend.Detect_For_Vedio import process_vedio


IMAGE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detections"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vedio"
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "assets" / "lane_detection.mp4"
PROCESS_IMAGE_COUNT = 10


def _scan_image_results():
    return sorted(IMAGE_OUTPUT_DIR.glob("*_result.jpg"))


def _scan_video_results():
    return sorted(VIDEO_OUTPUT_DIR.glob("*.mp4"))


def _read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"无法读取结果图片：{path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _image_view(paths, index):
    if not paths:
        return None, 0, "暂无图片结果"
    index = max(0, min(int(index), len(paths) - 1))
    return _read_image(paths[index]), index, f"当前结果：{index + 1}/{len(paths)}"


def run_image_detection(image_count):
    paths = process_images(int(image_count))
    if not paths:
        raise gr.Error("没有成功生成图片结果")
    result, index, _ = _image_view(paths, 0)
    return result, [str(path) for path in paths], index, f"图片处理完成：生成 {len(paths)} 张结果"


def load_image_results():
    paths = _scan_image_results()
    result, index, status = _image_view(paths, 0)
    return result, [str(path) for path in paths], index, status


def change_image(paths, index, step):
    return _image_view(paths, int(index) + step)


def run_video_detection(video_path):
    if not video_path:
        raise gr.Error("请先选择视频")
    output_path, frame_count = process_vedio(video_path)
    return str(output_path), f"视频处理完成：共处理 {frame_count} 帧"


def load_video_results():
    paths = _scan_video_results()
    if not paths:
        return None, "暂无视频结果"
    return str(paths[0]), f"已加载已有结果：共 {len(paths)} 个视频"


def build_1_UI():
    gr.Markdown("# 行人识别系统")

    with gr.Tab("图片处理"):
        _build_1_images_ui()

    with gr.Tab("视频处理"):
        _build_1_video_ui()


def _build_1_images_ui():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 本地数据集处理")
            image_count = gr.Slider(
                minimum=1,
                maximum=100,
                value=PROCESS_IMAGE_COUNT,
                step=1,
                label="处理前 N 张图片",
            )
            process_btn = gr.Button("开始处理", variant="primary")
            load_btn = gr.Button("加载已有结果", variant="secondary")
            status_text = gr.Markdown("状态：等待处理")

        with gr.Column(scale=2):
            gr.Markdown("### 结果浏览")
            output_image = gr.Image(label="处理结果", type="numpy")
            with gr.Row():
                prev_btn = gr.Button("← 上一张", variant="secondary")
                next_btn = gr.Button("下一张 →", variant="primary")

    result_paths = gr.State([])
    result_index = gr.State(0)

    process_btn.click(
        run_image_detection,
        inputs=image_count,
        outputs=[output_image, result_paths, result_index, status_text],
    )
    load_btn.click(
        load_image_results,
        outputs=[output_image, result_paths, result_index, status_text],
    )
    prev_btn.click(
        lambda paths, index: change_image(paths, index, -1),
        inputs=[result_paths, result_index],
        outputs=[output_image, result_index, status_text],
    )
    next_btn.click(
        lambda paths, index: change_image(paths, index, 1),
        inputs=[result_paths, result_index],
        outputs=[output_image, result_index, status_text],
    )


def _build_1_video_ui():
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="输入视频",
                sources=["upload"],
                value=str(DEFAULT_VIDEO_PATH) if DEFAULT_VIDEO_PATH.exists() else None,
            )
            process_btn = gr.Button("开始视频检测", variant="primary")
            load_btn = gr.Button("加载已有结果", variant="secondary")
            status_text = gr.Markdown("状态：等待处理")
        with gr.Column():
            video_output = gr.Video(label="检测结果视频")

    process_btn.click(
        run_video_detection,
        inputs=video_input,
        outputs=[video_output, status_text],
    )
    load_btn.click(
        load_video_results,
        outputs=[video_output, status_text],
    )
