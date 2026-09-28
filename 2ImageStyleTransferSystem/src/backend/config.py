from dataclasses import dataclass


@dataclass(frozen=True)
class GatysConfig:
    image_size: int = 512
    steps: int = 300
    learning_rate: float = 0.02
    content_weight: float = 1.0
    style_weight: float = 1_000_000.0
    log_interval: int = 25

    def validate(self) -> None:
        if self.image_size <= 0 or self.steps <= 0 or self.log_interval <= 0:
            raise ValueError("image_size、steps 和 log_interval 必须大于 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if self.content_weight < 0 or self.style_weight < 0:
            raise ValueError("损失权重不能小于 0")
