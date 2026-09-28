"""结果图片与统计数据的保存工具。"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.utils import save_image

from .dataset import to_display_tensor


def save_result(image, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_display_tensor(image), path)
    return path


def save_statistics(history, statistics_dir, result_name, metadata):
    directory = Path(statistics_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{result_name}_losses.csv"
    json_path = directory / f"{result_name}_summary.json"
    chart_path = directory / f"{result_name}_losses.png"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("step", "total", "content", "style"))
        writer.writeheader()
        writer.writerows(history)

    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    steps = [row["step"] for row in history]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, [row["content"] for row in history], label="content loss")
    plt.plot(steps, [row["style"] for row in history], label="style loss")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return {"csv": csv_path, "json": json_path, "chart": chart_path}
