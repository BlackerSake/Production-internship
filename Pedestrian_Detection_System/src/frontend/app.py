"""
Gradio 统一前端入口
启动方式: 
    gradio Pedestrian_Detection_System/src/frontend/app.py
    浏览器访问: 
"""

import sys
import gradio as gr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Pedestrian_Detection_System.src.frontend.pages.page_2_PDS import build_1_UI
from Image_Style_Transfer_System.src.frontend.pages.page_Gatys import build_2_UI
def build_app() -> gr.Blocks:
    with gr.Blocks(title="生产实习") as app:
        gr.Markdown("# 2026年秋生产实习演示平台")

        with gr.Tab("3.1 行人检测系统"):
            build_1_UI()

        with gr.Tab('3.2 图像风格迁移'):
            build_2_UI()

    return app


if __name__ == "__main__":
    build_app().launch()
