import logging
from pathlib import Path
import sys
from PIL import Image
from torchvision import transforms
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))



def load_image(image_path, max_size, shape=None):
    """加载图片并进行标准化"""
    img = Image.open(image_path).convert('RGB')

    if max(img.size) > max_size:
        size = max_size
    else:
        size = max(img.size)
    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[],
            std=[]
        )
    ])