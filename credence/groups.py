from dataclasses import dataclass, field

import torch


@dataclass
class Sample:
    prompt: str
    prompt_ids: list
    output_ids: list
    text: str
    behavior_logprobs: torch.Tensor
    schema: dict | None = None


@dataclass
class Candidate:
    sample: Sample
    store: object
    asset_id: str | None = None
    payload_tokens: int = 0
    valid: bool = True
    scores: list = field(default_factory=list)
    reward: float | None = None


@dataclass
class CandidateGroup:
    group_id: str
    head: str
    stream: int
    generated_update: int
    born: int
    candidates: list
    reference_store: object | None = None
    reference_scores: list = field(default_factory=list)

    def mature(self, episode, horizon, cost):
        elapsed = episode - self.born
        for candidate in self.candidates:
            if candidate.reward is not None:
                continue
            if self.head == "curator":
                if elapsed >= horizon:
                    if len(candidate.scores) != len(self.reference_scores):
                        raise ValueError("Candidate and reference continuations differ in length")
                    # Matched reference continuations cancel task difficulty when scoring curator choices.
                    candidate.reward = sum(a - b for a, b in zip(candidate.scores, self.reference_scores)) / max(1, len(candidate.scores))
            elif not candidate.valid or candidate.asset_id is None:
                candidate.reward = -cost * candidate.payload_tokens / 1000
            else:
                record = candidate.store.creators[candidate.asset_id]
                if record.retired is not None or elapsed >= horizon:
                    candidate.reward = record.reward(cost)
        return all(candidate.reward is not None for candidate in self.candidates)


class GroupBuffer:
    def __init__(self, max_lag=32):
        self.max_lag = max_lag
        self.pending = []
        self.generated = 0
        self.matured = 0
        self.excluded = 0
        self.lags = []
        self.unavailable = 0

    def add(self, group):
        prompts = {candidate.sample.prompt for candidate in group.candidates}
        if len(prompts) != 1 or len(group.candidates) < 2:
            raise ValueError("Candidate groups require a shared prompt and multiple outputs")
        self.pending.append(group)
        self.generated += 1

    def release(self, head_updates, horizon, cost):
        ready, waiting = [], []
        for group in self.pending:
            lag = head_updates[group.head] - group.generated_update
            if lag > self.max_lag:
                self.excluded += 1
                self.lags.append(lag)
                continue
            episode = max(candidate.store.episode for candidate in group.candidates)
            if group.mature(episode, horizon, cost):
                ready.append(group)
                self.matured += 1
                self.lags.append(lag)
            else:
                waiting.append(group)
        self.pending = waiting
        return ready

    def statistics(self):
        return {"generated_groups": self.generated, "matured_groups": self.matured,
                "excluded_groups": self.excluded, "pending_groups": len(self.pending),
                "policy_lags": list(self.lags), "unavailable_groups": self.unavailable}

    def discard_unavailable(self, group):
        if group in self.pending:
            self.pending.remove(group)
            self.unavailable += 1


def group_advantages(rewards, epsilon=1e-8):
    values = torch.as_tensor(rewards)
    if not values.is_floating_point():
        values = values.float()
    if values.ndim != 1 or len(values) < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("A group requires multiple finite scalar rewards")
    centered = values - values.mean()
    return centered / (torch.sqrt(centered.square().mean()) + epsilon)


def grpo_loss(current, behavior, reference, advantage, clip_ratio=0.2, kl_weight=0.01):
    if current.shape != behavior.shape or current.shape != reference.shape or not current.numel():
        raise ValueError("Token likelihood sequences require matching nonempty shapes")
    ratio = (current - behavior.detach()).exp()
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
    gain = torch.minimum(ratio * advantage, clipped * advantage)
    difference = reference.detach() - current
    divergence = difference.exp() - difference - 1
    return -(gain - kl_weight * divergence).mean()
