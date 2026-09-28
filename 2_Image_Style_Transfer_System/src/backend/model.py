
import torch
from torchvision import models

class VGGFeatureExtractor:
    """VGG19特征提取器,冻结参数,负责向前传播"""

    def __init__(self, device):
        self.device = device
        self.model = self._load_and_freeze()
        self.layer_name_mapping = {
            '0' : 'conv1_1', 
            '5' : 'conv2_1', 
            '10': 'conv3_1', 
            '19': 'conv4_1', 
            '28': 'conv5_1'
        }

    def _load_and_freeze(self):
        vgg = models.vgg19(pretrained=True).features.to(self.device).eval()

        for p in vgg.parameters():
            p.requires_grad = False
        return vgg
    
    