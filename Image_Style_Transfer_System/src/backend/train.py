"""Gatys 风格迁移的迭代优化过程。"""

import logging
import time

import torch

try:
    from .loss import CONTENT_LAYER, STYLE_LAYERS, calculate_losses, gram_matrix
except ImportError:
    from loss import CONTENT_LAYER, STYLE_LAYERS, calculate_losses, gram_matrix


logger = logging.getLogger(__name__)

def optimize_image(content, style, extractor, config, progress_callback=None):
    with torch.no_grad():
        content_features = extractor(content)
        style_features = extractor(style)
        content_target = content_features[CONTENT_LAYER]
        style_targets = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

    generated = content.clone().requires_grad_(True)

    # LBFGS 关键参数：lr=1.0，max_iter=20
    optimizer = torch.optim.LBFGS([generated], lr=config.learning_rate, max_iter=20)

    history = []
    started_at = time.perf_counter()
    latest = {"total": 0.0, "content": 0.0, "style": 0.0}

    for step in range(1, config.steps + 1):
        # closure 会被 LBFGS 内部反复调用，所有 loss 计算都放进去
        def closure():
            optimizer.zero_grad()
            content_loss, style_loss = calculate_losses(
                extractor(generated), content_target, style_targets
            )
            total_loss = (
                config.content_weight * content_loss
                + config.style_weight * style_loss
            )
            total_loss.backward()
            # 记录最后一次的 loss，供日志使用
            latest["total"] = total_loss.item()
            latest["content"] = content_loss.item()
            latest["style"] = style_loss.item()
            return total_loss

        optimizer.step(closure)   # 注意：step 需要传 closure

        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            row = {
                "step": step,
                "total": latest["total"],
                "content": latest["content"],
                "style": latest["style"],
            }
            history.append(row)
            logger.info(
                "风格迁移迭代 %d/%d：总损失=%.4f，内容损失=%.4f，风格损失=%.4f",
                step, config.steps,
                row["total"], row["content"], row["style"],
            )
            if progress_callback:
                progress_callback(step, config.steps, row)

    return generated, history, time.perf_counter() - started_at