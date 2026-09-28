import torch.nn.functional as F


STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")
CONTENT_LAYER = "conv4_2"


def gram_matrix(feature):
    batch, channels, height, width = feature.shape
    values = feature.reshape(batch, channels, height * width)
    return values @ values.transpose(1, 2) / (channels * height * width)


def calculate_losses(generated, content_target, style_targets):
    content_loss = F.mse_loss(generated[CONTENT_LAYER], content_target)
    style_loss = sum(
        F.mse_loss(gram_matrix(generated[layer]), style_targets[layer])
        for layer in STYLE_LAYERS
    )
    return content_loss, style_loss
