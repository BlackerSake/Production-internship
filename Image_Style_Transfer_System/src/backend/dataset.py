from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def load_image(image_path, image_size, device, shape=None):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")

    with Image.open(path) as source:
        image = source.convert("RGB")

    if shape is None:
        scale = min(1.0, image_size / max(image.size))
        shape = (round(image.height * scale), round(image.width * scale))

    tensor = TF.to_tensor(TF.resize(image, shape, antialias=True)).unsqueeze(0)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return ((tensor.to(device) - mean) / std).contiguous()


def to_display_tensor(image):
    mean = IMAGENET_MEAN.to(image.device)
    std = IMAGENET_STD.to(image.device)
    return (image.detach() * std + mean).squeeze(0).clamp(0, 1).cpu()
