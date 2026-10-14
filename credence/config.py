from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    backbone: str = "Qwen/Qwen3-8B"
    device: str = "cpu"
    dtype: str = "float32"
    context_budget: int = 24
    store_capacity: int = 1000
    credit_horizon: int = 20
    restoration_horizon: int = 10
    group_size: int = 8
    max_policy_lag: int = 32
    embedding_dim: int = 3072
    predictor_hidden: int = 512
    adapter_rank: int = 16
    adapter_alpha: int = 32
    token_cost: float = 0.002
    encoder_weight: float = 0.2
    predictor_weight: float = 0.2
    contrastive_weight: float = 0.2
    contrastive_temperature: float = 0.1
    clip_ratio: float = 0.2
    kl_weight: float = 0.01
    advantage_epsilon: float = 1e-8
    learning_rate_scale: float = 0.01
    learning_rate_offset: int = 1000
    gradient_norm: float = 1.0
    calibration_quantile: float = 0.90
    calibration_stride: int = 5
    reuse_margin: float = 0.25
    warmup_episodes: int = 100
    token_budget: int = 28000
    max_steps: int = 50
    max_subtasks: int = 10
    encoder_tokens: int = 512
    encoder_batch: int = 8
    card_tokens: int = 1024
    policy_tokens: int = 512
    planner_tokens: int = 256
    actor_tokens: int = 1024
    seed: int = 149
    order_seeds: tuple = (41, 42, 43, 44)
    store_seeds: tuple = (141, 142, 143, 144, 145, 146, 147, 148)
    grammar_mask: bool = True
    protect_recent: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        positive = ("context_budget", "store_capacity", "credit_horizon",
                    "restoration_horizon", "embedding_dim", "predictor_hidden",
                    "adapter_rank", "adapter_alpha", "calibration_stride",
                    "token_budget", "max_steps", "max_subtasks", "encoder_tokens",
                    "encoder_batch", "card_tokens", "policy_tokens", "planner_tokens", "actor_tokens",
                    "learning_rate_offset")
        if any(getattr(self, name) < 1 for name in positive):
            raise ValueError("Counts and dimensions require positive values")
        if self.group_size < 2 or self.max_policy_lag < 0 or self.warmup_episodes < 0:
            raise ValueError("Invalid group, lag, or warm-up setting")
        if not 0 < self.calibration_quantile < 1:
            raise ValueError("Calibration quantile lies outside (0, 1)")
        if not 0 < self.clip_ratio < 1 or self.advantage_epsilon <= 0:
            raise ValueError("Invalid policy objective setting")
        if self.contrastive_temperature <= 0 or self.learning_rate_scale <= 0:
            raise ValueError("Temperature and learning-rate scale require positive values")
        if len(self.order_seeds) != 4 or len(self.store_seeds) != 8:
            raise ValueError("Stream construction requires four order seeds and eight store seeds")
        for name in ("token_cost", "encoder_weight", "predictor_weight",
                     "contrastive_weight", "kl_weight", "reuse_margin"):
            if getattr(self, name) < 0:
                raise ValueError("Loss weights and costs require nonnegative values")

    @property
    def summary_dim(self):
        return 2 * self.embedding_dim + 6 + self.context_budget

    def learning_rate(self, step):
        return self.learning_rate_scale / (step + self.learning_rate_offset)

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        allowed = {item.name for item in fields(cls)}
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"Unknown configuration fields {sorted(extra)}")
        return cls(**values)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(asdict(self)), encoding="utf-8")
