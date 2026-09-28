from pathlib import Path
import sys

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)
from Image_Style_Transfer_System.src.backend.config import GatysConfig
from Image_Style_Transfer_System.src.backend.main import run_style_transfer


CONTENT_DIR = PROJECT_ROOT / "assets" / "val2014" / "val2014"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transferred"
DEFAULT_CONTENT_PATH = CONTENT_DIR / "COCO_val2014_000000157581.jpg"
DEFAULT_STYLE_PATH = CONTENT_DIR / "COCO_val2014_000000181249.jpg"


def _default_image(path):
    return str(path) if path.exists() else None


def _run_style_transfer(
    content_path,
    style_path,
    image_size,
    steps,
    learning_rate,
    content_weight,
    style_weight,
    progress=gr.Progress(),
):
    if not content_path:
        raise gr.Error("请选择内容图片")
    if not style_path:
        raise gr.Error("请选择风格图片")

    def update_progress(step, total, losses):
        progress(
            step / total,
            desc=f"第 {step}/{total} 步，总损失：{losses['total']:.4f}",
        )

    try:
        result = run_style_transfer(
            content_path=content_path,
            style_path=style_path,
            image_size=image_size,
            steps=steps,
            learning_rate=learning_rate,
            content_weight=content_weight,
            style_weight=style_weight,
            progress_callback=update_progress,
        )
    except Exception as exc:
        raise gr.Error(f"风格迁移失败：{exc}") from exc

    status = (
        f"处理完成：耗时 {result['elapsed_seconds']:.2f} 秒\n\n"
        f"统计文件：`{result['statistics']['json']}`\n\n"
        f"结果图片：`{result['result']}`"
    )
    return str(result["result"]), str(result["statistics"]["chart"]), status


def _load_latest_result():
    results = sorted(OUTPUT_DIR.glob("*.jpg"), key=lambda path: path.stat().st_mtime)
    if not results:
        raise gr.Error("暂无已有迁移结果")

    result_path = results[-1]
    chart_path = result_path.with_name(f"{result_path.stem}_losses.png")
    return (
        str(result_path),
        str(chart_path) if chart_path.exists() else None,
        f"已加载：`{result_path}`",
    )


def build_2_UI():
    gr.Markdown("# Gatys 图像风格迁移")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 输入图片")
            content_image = gr.Image(
                label="内容图片",
                type="filepath",
                sources=["upload"],
                value=_default_image(DEFAULT_CONTENT_PATH),
            )
            style_image = gr.Image(
                label="风格图片",
                type="filepath",
                sources=["upload"],
                value=_default_image(DEFAULT_STYLE_PATH),
            )

            gr.Markdown("### 迁移参数")
            image_size = gr.Slider(
                minimum=128,
                maximum=1024,
                value=GatysConfig.image_size,
                step=32,
                label="图片尺寸",
            )
            steps = gr.Slider(
                minimum=1,
                maximum=1500,
                value=GatysConfig.steps,
                step=1,
                label="优化步数",
            )
            learning_rate = gr.Number(
                value=GatysConfig.learning_rate,
                minimum=0.0001,
                label="学习率",
            )
            content_weight = gr.Number(
                value=GatysConfig.content_weight,
                minimum=0,
                label="内容损失权重",
            )
            style_weight = gr.Number(
                value=GatysConfig.style_weight,
                minimum=0,
                label="风格损失权重",
            )

            with gr.Row():
                process_button = gr.Button("开始风格迁移", variant="primary")
                load_button = gr.Button("加载已有结果", variant="secondary")
            status_text = gr.Markdown("状态：等待处理")

        with gr.Column(scale=2):
            result_image = gr.Image(label="迁移结果", type="filepath")
            loss_chart = gr.Image(label="损失变化曲线", type="filepath")

    process_button.click(
        _run_style_transfer,
        inputs=[
            content_image,
            style_image,
            image_size,
            steps,
            learning_rate,
            content_weight,
            style_weight,
        ],
        outputs=[result_image, loss_chart, status_text],
    )
    load_button.click(
        _load_latest_result,
        outputs=[result_image, loss_chart, status_text],
    )
