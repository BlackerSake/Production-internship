from pathlib import Path
import logging
import sys

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)
from Image_Style_Transfer_System.src.backend.config import GatysConfig
from Image_Style_Transfer_System.src.backend.main import (
    OUTPUT_DIR,
    STATISTICS_DIR,
    run_style_transfer
)


logger = logging.getLogger(__name__)


ASSETS_DIR = PROJECT_ROOT / "Image_Style_Transfer_System" / "assets"
CONTENT_DIR = ASSETS_DIR / "test"
DEFAULT_STYLE_PATH = ASSETS_DIR / "sunout.jpg"


def _load_latest_result():
    results = sorted(OUTPUT_DIR.glob("*.jpg"), key=lambda path: path.stat().st_mtime)
    if not results:
        raise gr.Error("暂无已有迁移结果")

    result_path = results[-1]
    chart_path = STATISTICS_DIR / f"{result_path.stem}_losses.png"
    return (
        str(result_path),
        str(chart_path) if chart_path.exists() else None,
        f"已加载：`{result_path}`",
    )


def _run_batch_style_transfer(
    count,
    image_size,
    steps,
    learning_rate,
    content_weight,
    style_weight,
    progress=gr.Progress(),
):
    images = sorted(CONTENT_DIR.glob("*.jpg"))[: int(count)]
    if not images:
        raise gr.Error(f"没有找到内容图片：{CONTENT_DIR}")
    if not DEFAULT_STYLE_PATH.is_file():
        raise gr.Error(f"风格图片不存在：{DEFAULT_STYLE_PATH}")

    results = []
    for index, content_path in enumerate(images):
        logger.info("开始批量处理第 %d/%d 张：%s", index + 1, len(images), content_path)
        def update_progress(step, total, losses):
            progress(
                (index + step / total) / len(images),
                desc=f"第 {index + 1}/{len(images)} 张，第 {step}/{total} 步",
            )

        try:
            result = run_style_transfer(
                content_path=content_path,
                style_path=DEFAULT_STYLE_PATH,
                image_size=image_size,
                steps=steps,
                learning_rate=learning_rate,
                content_weight=content_weight,
                style_weight=style_weight,
                progress_callback=update_progress,
            )
        except Exception as exc:
            raise gr.Error(f"处理 {content_path.name} 失败：{exc}") from exc
        results.append(str(result["result"]))
        logger.info("批量处理完成第 %d/%d 张：%s", index + 1, len(images), result["result"])

    logger.info("批量风格迁移完成：共处理 %d 张，风格图=%s", len(results), DEFAULT_STYLE_PATH)
    return results, f"已完成 {len(results)} 张，风格图：`{DEFAULT_STYLE_PATH}`"


_run_batch_style_transfer_impl = _run_batch_style_transfer
_load_latest_result_impl = _load_latest_result


def _scan_results():
    return sorted(OUTPUT_DIR.glob("*.jpg"), key=lambda path: path.stat().st_mtime)


def _result_view(paths, index):
    if not paths:
        return None, 0, "暂无迁移结果"

    index = max(0, min(int(index), len(paths) - 1))
    return paths[index], index, f"当前结果：{index + 1}/{len(paths)}"


def _change_result(paths, index, step):
    return _result_view(paths, int(index) + step)


def _run_batch_style_transfer(
    count,
    image_size,
    steps,
    learning_rate,
    content_weight,
    style_weight,
    progress=gr.Progress(),
):
    results, status = _run_batch_style_transfer_impl(
        count,
        image_size,
        steps,
        learning_rate,
        content_weight,
        style_weight,
        progress=progress,
    )
    result_image, result_index, _ = _result_view(results, 0)
    return results, result_image, results, result_index, status


def _load_latest_result():
    result_image, loss_chart, status = _load_latest_result_impl()
    results = [str(path) for path in _scan_results()]
    return result_image, loss_chart, results, len(results) - 1, status


def build_2_UI():
    gr.Markdown("# Gatys 图像风格迁移")
    image_count = len(list(CONTENT_DIR.glob("*.jpg")))

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 数据来源")
            gr.Markdown(
                f"内容目录：`{CONTENT_DIR}`\n\n"
                f"风格图片：`{DEFAULT_STYLE_PATH}`"
            )

            batch_count = gr.Number(
                value=max(1, image_count),
                minimum=1,
                maximum=max(1, image_count),
                precision=0,
                label="处理前 N 张（默认全部）",
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
            with gr.Row():
                prev_button = gr.Button("Previous", variant="secondary")
                next_button = gr.Button("Next", variant="primary")
            batch_results = gr.Gallery(label="批量迁移结果", columns=3)
            result_image = gr.Image(label="最新迁移结果", type="filepath")
            loss_chart = gr.Image(label="损失变化曲线", type="filepath")

    result_paths = gr.State([])
    result_index = gr.State(0)

    process_button.click(
        _run_batch_style_transfer,
        inputs=[
            batch_count,
            image_size,
            steps,
            learning_rate,
            content_weight,
            style_weight,
        ],
        outputs=[batch_results, result_image, result_paths, result_index, status_text],
    )
    load_button.click(
        _load_latest_result,
        outputs=[result_image, loss_chart, result_paths, result_index, status_text],
    )
    prev_button.click(
        lambda paths, index: _change_result(paths, index, -1),
        inputs=[result_paths, result_index],
        outputs=[result_image, result_index, status_text],
    )
    next_button.click(
        lambda paths, index: _change_result(paths, index, 1),
        inputs=[result_paths, result_index],
        outputs=[result_image, result_index, status_text],
    )
