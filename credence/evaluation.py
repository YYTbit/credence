import random
from pathlib import Path

import torch

from .checkpoint import file_digest, resolve_checkpoint
from .credit import Calibrator
from .environments import BENCHMARKS
from .records import Collection
from .runtime import EpisodeUnavailable, Runtime
from .store import Store
from .training import initial_store


def evaluate(models, environments, config, tasks, checkpoint, output, seed=149,
             input_identity=None):
    root = resolve_checkpoint(checkpoint)
    if (root / "evaluation.pt").is_file():
        state = torch.load(root / "evaluation.pt", map_location="cpu", weights_only=False)
        stores = {name: Store.from_state_dict(value) for name, value in state["stores"].items()}
    else:
        # Earlier checkpoints remain usable without allocating a training optimizer.
        state = torch.load(root / "training.pt", map_location="cpu", weights_only=False)
        stores = {stream.benchmark: stream.store for stream in reversed(state["streams"])}
    calibrator = Calibrator(config.calibration_quantile, config.reuse_margin)
    calibrator.load_state_dict(state["calibrator"])
    runtime = Runtime(models, calibrator, config)
    plan = []
    for index, benchmark in enumerate(BENCHMARKS):
        selected = [task for task in tasks if task["benchmark"] == benchmark and task["split"] != "train"]
        if benchmark == "appworld" and any(task["split"] != "test_normal" for task in selected):
            raise ValueError("AppWorld transfer uses Test-Normal tasks")
        random.Random(seed + index).shuffle(selected)
        plan.extend((index, position, task) for position, task in enumerate(selected))
    if not plan:
        raise ValueError("Evaluation requires tasks from an evaluation split")
    identity = {"stage": "evaluation", "tasks": plan, "seed": seed, "config": vars(config),
                "checkpoint": file_digest(root / "complete.json") if (root / "complete.json").is_file()
                else file_digest(root / "auxiliary.pt"), "inputs": input_identity or {}}
    output = Path(output)
    with Collection(output, identity, ("episodes.jsonl", "diagnostics.jsonl")) as collection:
        for cursor in range(collection.cursor, len(plan)):
            index, position, task = plan[cursor]
            benchmark = task["benchmark"]
            base_store = stores.get(benchmark)
            store = (base_store.clone() if base_store is not None
                     else initial_store(models, config, seed + index))
            store.advance()
            row = {"benchmark": benchmark, "task_id": task["task_id"], "seed": seed + position,
                   "backbone": config.backbone}
            try:
                episode = runtime.run_episode(task, store, environments[benchmark],
                                              output / "workspaces" / benchmark / str(position), seed + position)
            except EpisodeUnavailable as exc:
                row.update(status="unavailable", score=None, reason=str(exc))
            else:
                runtime.credit(episode, store)
                store = runtime.evolve(episode, store, seed + 50000 + position)
                row.update(status="scored", score=episode.score, tokens=episode.tokens,
                           store_size=len(store.assets))
            collection.commit({"episodes.jsonl": [row], "diagnostics.jsonl": runtime.diagnostics})
            runtime.diagnostics.clear()
        if not collection.state["complete"]:
            collection.commit({}, complete=True)
    from .artifacts import summarize_episodes
    return summarize_episodes(output / "episodes.jsonl", output)
