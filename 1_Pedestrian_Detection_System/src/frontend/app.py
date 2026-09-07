"""
Gradio 统一前端入口
启动方式: 
    gradio 1_Pedestrian_Detection_System/src/frontend/app.py
    浏览器访问: 
"""
from pathlib import Path

import gradio as gr

from src.frontend.pages.page_2_PDS import build_UI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def build_app() -> gr.Blocks:
    with gr.Blocks(title="生产实习") as app:
        gr.Markdown("# 2026年秋生产实习演示平台")

        with gr.Tab("3.1 行人检测系统"):
            build_UI()

    return app


if __name__ == "__main__":
    build_app().launch()
