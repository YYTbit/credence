import math
from dataclasses import dataclass

import torch


@dataclass
class CreditStats:
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0

    def update(self, value):
        if not math.isfinite(value):
            raise ValueError("Credit requires a finite value")
        old_count = self.count
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.variance = (max(0, old_count - 1) * self.variance
                         + delta * (value - self.mean)) / max(1, old_count)
        self.variance = max(0.0, self.variance)

    @classmethod
    def merge(cls, left, right):
        n = left.count + right.count
        if n == 0:
            return cls()
        mean = (left.count * left.mean + right.count * right.mean) / n
        spread = max(0, left.count - 1) * left.variance + max(0, right.count - 1) * right.variance
        spread += left.count * right.count * (left.mean - right.mean) ** 2 / n
        return cls(n, mean, spread / max(1, n - 1))


def residual_bounds(values, quantile):
    values = sorted(float(value) for value in values)
    if not values:
        return math.inf, math.inf
    if any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("Calibration residuals require finite nonnegative values")
    n = len(values)
    rank = math.ceil(n * quantile)
    radius = math.ceil(math.sqrt(n * quantile * (1 - quantile)))
    estimate = values[min(n, rank) - 1]
    uncertainty = (values[min(n, rank + radius) - 1]
                   - values[max(1, rank - radius) - 1]) / 2
    return estimate, uncertainty


class Calibrator:
    def __init__(self, quantile=0.90, reuse_weight=0.25):
        self.quantile = quantile
        self.reuse_weight = reuse_weight
        self.residual = math.inf
        self.uncertainty = math.inf
        self.lower = None
        self.upper = None

    def fit(self, observed, removals, supported_summaries):
        first = residual_bounds(observed, self.quantile)
        second = residual_bounds(removals, self.quantile)
        self.residual, self.uncertainty = max((first, second), key=lambda value: value[0])
        if len(supported_summaries):
            matrix = torch.stack([value.detach().cpu() for value in supported_summaries])
            self.lower, self.upper = matrix.min(dim=0).values, matrix.max(dim=0).values

    def margin(self, count):
        if count <= 0:
            return math.inf
        return self.residual + self.uncertainty + self.reuse_weight / math.sqrt(count)

    def eligible(self, stats):
        return abs(stats.mean) > self.margin(stats.count)

    def supported(self, summary):
        if self.lower is None:
            return False
        value = summary.detach().cpu()
        return bool(((value >= self.lower) & (value <= self.upper)).all())

    def state_dict(self):
        return {"residual": self.residual, "uncertainty": self.uncertainty,
                "lower": self.lower, "upper": self.upper}

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)
