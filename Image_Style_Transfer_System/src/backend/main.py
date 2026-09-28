"""Gatys 图像风格迁移后端入口（库函数，供前端/Gradio 调用）。

只暴露 pick_device() 与 transfer_style()，本模块不提供命令行入口。
"""

from pathlib import Path

import torch

try:
    from .config import GatysConfig
    from .dataset import load_image
    from .model import VGGFeatureExtractor
    from .train import optimize_image
    from .utils import save_result, save_statistics
except ImportError:
    from config import GatysConfig
    from dataset import load_image
    from model import VGGFeatureExtractor
    from train import optimize_image
    from utils import save_result, save_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transferred"
STATISTICS_DIR = PROJECT_ROOT / "outputs" / "statistics"


def pick_device():
    """选择一个可用设备：有 CUDA 用 GPU，否则退回 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def transfer_style(content_path, style_path, config=None, output_path=None, progress_callback=None):
    """把 style_path 的风格迁移到 content_path，保存结果图片和统计数据。

    参数:
        content_path / style_path: 本地图片路径
        config: GatysConfig；为 None 时使用默认参数
        output_path: 结果保存路径，默认 outputs/transferred/<内容名>_<风格名>.jpg
        progress_callback: 可选，接收 (step, total, losses)，供前端进度条使用
    返回:
        dict: result(结果图), statistics(统计文件), history(损失记录), elapsed_seconds(耗时秒)
    """
    config = config or GatysConfig()
    config.validate()
    device = pick_device()

    content_path = Path(content_path)
    style_path = Path(style_path)
    result_name = f"{content_path.stem}_{style_path.stem}"

    content = load_image(content_path, config.image_size, device)
    style = load_image(style_path, config.image_size, device, content.shape[-2:])
    extractor = VGGFeatureExtractor(device)
    generated, history, elapsed_seconds = optimize_image(
        content, style, extractor, config, progress_callback
    )

    result_path = save_result(
        generated,
        Path(output_path) if output_path else OUTPUT_DIR / f"{result_name}.jpg",
    )
    statistics = save_statistics(
        history,
        STATISTICS_DIR,
        result_name,
        {
            "content_image": str(content_path),
            "style_image": str(style_path),
            "result_image": str(result_path),
            "device": str(device),
            "image_size": list(content.shape[-2:]),
            "steps": config.steps,
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
    )
    return {
        "result": result_path,
        "statistics": statistics,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
    }


def run_style_transfer(
    content_path,
    style_path,
    image_size=GatysConfig.image_size,
    steps=GatysConfig.steps,
    learning_rate=GatysConfig.learning_rate,
    content_weight=GatysConfig.content_weight,
    style_weight=GatysConfig.style_weight,
    progress_callback=None,
):
    """接收前端参数并执行一次风格迁移。"""
    config = GatysConfig(
        image_size=int(image_size),
        steps=int(steps),
        learning_rate=float(learning_rate),
        content_weight=float(content_weight),
        style_weight=float(style_weight),
        log_interval=max(1, int(steps) // 20),
    )
    return transfer_style(
        content_path,
        style_path,
        config=config,
        progress_callback=progress_callback,
    )

