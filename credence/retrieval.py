import torch
import torch.nn.functional as F


def similarities(subtasks, assets):
    if subtasks.ndim != 2 or assets.ndim != 2 or subtasks.shape[1] != assets.shape[1]:
        raise ValueError("Embedding matrices require a shared dimension")
    return (F.normalize(subtasks, dim=-1) @ F.normalize(assets, dim=-1).T).clamp(0, 1)


def coverage(subtasks, assets):
    if assets.shape[0] == 0 or subtasks.shape[0] == 0:
        return subtasks.sum() * 0 + assets.sum() * 0
    return similarities(subtasks, assets).max(dim=1).values.sum()


def greedy_select(subtasks, assets, budget):
    if budget < 0:
        raise ValueError("Context budget requires a nonnegative value")
    sim = similarities(subtasks, assets)
    best = sim.new_zeros(sim.shape[0])
    available = torch.ones(sim.shape[1], dtype=torch.bool, device=sim.device)
    indices = []
    for _ in range(min(budget, assets.shape[0])):
        gains = (sim - best[:, None]).clamp_min(0).sum(dim=0)
        gains = gains.masked_fill(~available, -1)
        index = int(gains.argmax())
        if float(gains[index]) <= 0:
            break
        indices.append(index)
        available[index] = False
        best = torch.maximum(best, sim[:, index])
    return indices, best.sum()


def pool_summary(subtasks, embeddings, assets, budget):
    if len(assets) != embeddings.shape[0]:
        raise ValueError("Asset records and embeddings differ in length")
    dim = subtasks.shape[1]
    mean_task = F.normalize(subtasks.mean(dim=0), dim=0) if len(subtasks) else subtasks.new_zeros(dim)
    mean_asset = F.normalize(embeddings.mean(dim=0), dim=0) if assets else subtasks.new_zeros(dim)
    values = similarities(subtasks, embeddings).max(dim=1).values if assets and len(subtasks) else subtasks.new_zeros(0)
    values = values.sort(descending=True).values[:budget]
    values = F.pad(values, (0, budget - len(values)))
    count = max(1, len(assets))
    stats = subtasks.new_tensor([
        len(assets), sum(a.age for a in assets) / count,
        max((a.age for a in assets), default=0),
        sum(a.kind == "tool" for a in assets),
        sum(a.credit.mean for a in assets) / count,
        sum(a.credit.variance for a in assets) / count,
    ])
    return torch.cat((mean_asset, mean_task, stats, values))


def keyword_partition(embeddings):
    if len(embeddings) < 2:
        raise ValueError("Memory splitting requires at least two keywords")
    points = F.normalize(embeddings.detach(), dim=-1)
    second = int(((points - points[0]) ** 2).sum(dim=1).argmax())
    centers = torch.stack((points[0], points[second]))
    labels = torch.zeros(len(points), dtype=torch.long, device=points.device)
    for _ in range(30):
        current = torch.cdist(points, centers).argmin(dim=1)
        if current.min() == current.max():
            current[len(points) // 2:] = 1
        updated = torch.stack([points[current == k].mean(dim=0) for k in range(2)])
        if torch.equal(current, labels) and torch.allclose(updated, centers):
            break
        labels, centers = current, updated
    return [[i for i in range(len(points)) if int(labels[i]) == k] for k in range(2)]
