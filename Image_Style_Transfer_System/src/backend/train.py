"""Gatys 风格迁移的迭代优化过程。"""

import time

import torch

try:
    from .loss import CONTENT_LAYER, STYLE_LAYERS, calculate_losses, gram_matrix
except ImportError:
    from loss import CONTENT_LAYER, STYLE_LAYERS, calculate_losses, gram_matrix


def optimize_image(content, style, extractor, config, progress_callback=None):
    with torch.no_grad():
        content_features = extractor(content)
        style_features = extractor(style)
        content_target = content_features[CONTENT_LAYER]
        style_targets = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

    generated = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=config.learning_rate)
    history = []
    started_at = time.perf_counter()

    for step in range(1, config.steps + 1):
        content_loss, style_loss = calculate_losses(
            extractor(generated), content_target, style_targets
        )
        total_loss = (
            config.content_weight * content_loss
            + config.style_weight * style_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            row = {
                "step": step,
                "total": total_loss.item(),
                "content": content_loss.item(),
                "style": style_loss.item(),
            }
            history.append(row)
            if progress_callback:
                progress_callback(step, config.steps, row)

    return generated, history, time.perf_counter() - started_at
