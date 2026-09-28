from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class VGGFeatureExtractor(nn.Module):
    LAYERS = {
        "0": "conv1_1",
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "21": "conv4_2",
        "28": "conv5_1",
    }

    def __init__(self, device):
        super().__init__()
        self.features = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        self.features.requires_grad_(False)

    def forward(self, image):
        outputs = {}
        for index, layer in self.features._modules.items():
            image = layer(image)  # type: ignore
            if index in self.LAYERS:
                outputs[self.LAYERS[index]] = image
            if index == "28":
                break
        return outputs
