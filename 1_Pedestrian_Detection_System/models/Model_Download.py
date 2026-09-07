from ultralytics import YOLO

# 加载预训练模型（会自动下载 .pt 文件）
model = YOLO("yolov8n.pt")

# 导出为 ONNX
model.export(format="onnx", imgsz=640)