

from __future__ import print_function
import numpy as np
import cv2
import time

# 日志配置
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# 超参

INPUT_SIZE = 640 # 输入图像大小
CONFIDENCE_THRESHOLD = 0.3 # 置信度阈值
NMS_THRESHOLD = 0.4 # 非极大值抑制阈值

# 检测行人
class PedestrianDetectorSystem:
    def __init__(self, 
                 model_path,
                 input_size=INPUT_SIZE,
                 conf_threshold=CONFIDENCE_THRESHOLD,
                 nms_threshold=NMS_THRESHOLD):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 加载model
        self.network = cv2.dnn.readNetFromONNX(str(model_path))
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def letterbox(self, image):
        """图像预处理: 缩放 + 填充
        Args:
            image: 原始图像
        Returns:
            canvas: 预处理后的图像
            scale: 缩放比例
            padding_x: x方向的填充
            padding_y: y方向的填充
        """
        # 缩放
        height, width = image.shape[:2]
        scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
        new_height, new_width = round(height * scale), round(width * scale)
        new_image = cv2.resize(image, (new_width, new_height))

        # 填充
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padding_x, padding_y = (INPUT_SIZE - new_width) // 2, (INPUT_SIZE - new_height) // 2

        canvas[padding_y:padding_y + new_height, padding_x:padding_x + new_width] = new_image

        return canvas, scale, padding_x, padding_y


    def detect(self, image):
        """对单张图片进行检测, 返回检测框列表"""

        # 1. 图像预处理
        input_image, scale, padding_x, padding_y = self.letterbox(image)

        # 2. 将图像转换为blob(4D张量)
        blob = cv2.dnn.blobFromImage(
            image=input_image, 
            scalefactor=1/255.0, 
            size=(INPUT_SIZE, INPUT_SIZE), 
            mean=(0, 0, 0), 
            swapRB=True, 
            crop=False
        )

        # 3. 将blob输入网络进行前向传播 推理
        self.network.setInput(blob)
        start_time = time.time()
        preds = self.network.forward()
        logger.info(f"OpenCV DNN向前传播完成, 花费时间:{(time.time() - start_time)*1000:.2f} ms")

        # 4. 后处理, 解析输出结果
        preds = np.squeeze(preds) # 去掉多余的 batch 维度, 变为二维数组
        if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
            preds = preds.T # 转置, 变为 (num_detections, 84)

        boxes, confs, classes = [], [], [] # 分别为 检测框列表, 置信度列表, 类别列表
        for p in preds:

            conf, cls = p[4], int(p[5]) # 置信度和类别
            if conf < self.conf_threshold:
                # 如果 conf 低于阈值, 则跳过该检测框
                continue

            cx, cy, w, h = p[:4] # 分别为中心点坐标(center_x,center_y)和宽高(w,h)
            # 还原到原图坐标系(还要考虑缩放和填充)
            left = (cx - w/2 - padding_x) / scale
            top = (cy - h/2 - padding_y) / scale
            right = (cx + w/2 - padding_x) / scale
            bottom = (cy + h/2 - padding_y) / scale
            
            # 边界裁剪
            img_h, img_w = image.shape[:2]
            left = max(0, min(int(left), img_w - 1))
            top = max(0, min(int(top), img_h - 1))
            right = max(0, min(int(right), img_w - 1))
            bottom = max(0, min(int(bottom), img_h - 1))
            
            if right <= left or bottom <= top:
                continue
            
            boxes.append([left, top, right - left, bottom - top])
            confs.append(float(conf))
            classes.append(cls)

        # NMS,即非极大值抑制, 用于去除重叠的检测框
        idx = cv2.dnn.NMSBoxes(boxes, confs, 
                                self.conf_threshold, 
                                self.nms_threshold) 
        detecs = []
        if len(idx) > 0:
            idx = np.asarray(idx).flatten() # 展平,将idx转换为一维数组
            for i in idx:
                x, y, w, h = boxes[i]
                detecs.append([x, y, w, h, confs[i]])
        return detecs
        
    def draw_detections(self, image, detections):
        """绘制所有行人检测框"""
        result_image = image.copy()

        for x, y, w, h, conf in detections:
            cv2.rectangle(result_image,
                          (int(x), int(y)), 
                          (int(x + w), int(y + h)), 
                          (0, 255, 0), 2)
            label = f"Person: {conf:.2f}"

            cv2.putText(result_image,
                        label,
                        (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)
        return result_image

