import json
import random
import subprocess
from pathlib import Path

from .credit import Calibrator
from .decoding import DECODING_PROTOCOL
from .checkpoint import file_digest, write_json
from .records import Collection
from .runtime import EpisodeUnavailable, Runtime, TokenBudget, policy_schema
from .training import TRAIN_BENCHMARKS, initial_store


HEADS = ("writer", "builder", "curator")


def _round_robin(tasks, count, seed):
    if count < 1:
        raise ValueError("Trajectory count must be positive")
    grouped = {name: [] for name in TRAIN_BENCHMARKS}
    nontrain = []
    for task in tasks:
        if task.get("split") != "train":
            nontrain.append(task)
            continue
        benchmark = task.get("benchmark")
        if benchmark not in grouped:
            raise ValueError("Warm-start training data must use the four training benchmarks")
        grouped[benchmark].append(task)
    for benchmark, rows in grouped.items():
        if not rows:
            raise ValueError(f"Training tasks missing for {benchmark}")
        train_ids = {row["task_id"] for row in rows}
        overlap = {row["task_id"] for row in nontrain if row.get("benchmark") == benchmark} & train_ids
        if overlap:
            raise ValueError(f"Training and evaluation task identifiers overlap for {benchmark}")
        random.Random(seed + TRAIN_BENCHMARKS.index(benchmark)).shuffle(rows)
    plan = []
    positions = dict.fromkeys(TRAIN_BENCHMARKS, 0)
    for index in range(count):
        benchmark = TRAIN_BENCHMARKS[index % len(TRAIN_BENCHMARKS)]
        rows = grouped[benchmark]
        position = positions[benchmark]
        plan.append(rows[position % len(rows)])
        positions[benchmark] = position + 1
    return plan, len(nontrain)


def _response(sample):
    value = json.loads(sample.text)
    if not isinstance(value, dict):
        raise ValueError("Head targets require a JSON object")
    return value


def _target(runtime, head, episode, store, seed, asset_id=None):
    prompt = runtime.prompt(head, episode, store, asset_id)
    budget = TokenBudget(runtime.config.token_budget)
    sample = runtime.complete("agent", prompt, runtime.config.policy_tokens, budget,
                              seed, policy_schema(head, store, asset_id), sample=True)
    response = _response(sample)
    trial = store.clone()
    runtime.apply(head, sample, episode, trial, asset_id)
    return {"head": head, "prompt": prompt,
            "response": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            "generation_seed": seed}


def collect_warm_start(models, environments, config, tasks, output, count=1000, seed=None,
                       runtime_factory=Runtime, input_hashes=None):
    """Collect base-model rollouts and schema-validated targets from training tasks."""
    seed = config.seed if seed is None else seed
    plan, excluded = _round_robin(tasks, count, seed)
    missing = set(TRAIN_BENCHMARKS) - set(environments)
    if missing:
        raise ValueError(f"Training environments missing for {sorted(missing)}")
    output = Path(output)
    runtime = runtime_factory(models, Calibrator(config.calibration_quantile, config.reuse_margin), config)
    model_config = getattr(models.model, "config", None)
    identity = {"stage": "warm-start", "decoding_protocol": DECODING_PROTOCOL,
                "plan": plan, "seed": seed, "config": vars(config),
                "base_revision": getattr(model_config, "_commit_hash", None), "inputs": input_hashes or {}}
    with Collection(output, identity, ("trajectories.jsonl", "head_examples.jsonl")) as collection:
        return _collect(models, environments, config, plan, seed, runtime, output,
                        collection, count, excluded, input_hashes)


def _collect(models, environments, config, plan, seed, runtime, output, collection,
             count, excluded, input_hashes):
    metadata = collection.state["metadata"]
    counts = metadata.get("head_counts", dict.fromkeys(HEADS, 0))
    rejected = metadata.get("rejected", dict.fromkeys(HEADS, 0))
    for index in range(collection.cursor, len(plan)):
        task = plan[index]
        examples = []
        row_seed = seed + index
        store = initial_store(models, config, row_seed)
        try:
            episode = runtime.run_episode(task, store, environments[task["benchmark"]],
                                          output / "workspaces" / task["benchmark"] / str(index), row_seed)
        except EpisodeUnavailable as exc:
            raise RuntimeError(f"Base-model rollout unavailable for {task['benchmark']}:{task['task_id']}") from exc
        trajectory_id = f"warm-{index:06d}"
        trajectory = {"trajectory_id": trajectory_id, "benchmark": task["benchmark"],
                             "task_id": task["task_id"], "split": "train", "seed": row_seed,
                             "task": task, "score": episode.score, "subtasks": episode.subtasks,
                             "assembled": episode.assembled, "actions": episode.trajectory,
                             "evidence": episode.evidence, "tokens": episode.tokens,
                             "completed": episode.completed, "initial_store": store.state_dict()}
        for head in ("writer", "builder"):
            try:
                example = _target(runtime, head, episode, store, row_seed + 1000 + HEADS.index(head))
            except (ValueError, TypeError, KeyError, AttributeError, subprocess.TimeoutExpired):
                rejected[head] += 1
                continue
            example.update({"trajectory_id": trajectory_id, "benchmark": task["benchmark"],
                            "task_id": task["task_id"], "split": "train", "rollout_seed": row_seed})
            examples.append(example)
        for offset, asset_id in enumerate(episode.assembled):
            if asset_id not in store.assets:
                continue
            try:
                example = _target(runtime, "curator", episode, store,
                                  row_seed + 2000 + offset, asset_id)
            except (ValueError, TypeError, KeyError, AttributeError, subprocess.TimeoutExpired):
                rejected["curator"] += 1
                continue
            example.update({"trajectory_id": trajectory_id, "benchmark": task["benchmark"],
                            "task_id": task["task_id"], "split": "train", "rollout_seed": row_seed})
            examples.append(example)
        for row in examples:
            counts[row["head"]] += 1
        collection.commit({"trajectories.jsonl": [trajectory], "head_examples.jsonl": examples},
                          {"head_counts": counts, "rejected": rejected})
    if any(counts[head] == 0 for head in HEADS):
        raise ValueError(f"No validated warm-start targets for every head: {counts}")
    trajectory_path = output / "trajectories.jsonl"
    example_path = output / "head_examples.jsonl"
    model_config = getattr(models.model, "config", None)
    manifest = {
        "protocol": "CREDENCE Appendix B.10 base-model rollout warm start",
        "requested_trajectories": count,
        "collected_trajectories": collection.cursor,
        "round_robin_benchmarks": list(TRAIN_BENCHMARKS),
        "trajectories_per_benchmark": {name: sum(row["benchmark"] == name for row in plan)
                                       for name in TRAIN_BENCHMARKS},
        "split": "train",
        "excluded_nontrain_records": excluded,
        "head_example_counts": counts,
        "rejected_targets": rejected,
        "base_policy": {"backbone": config.backbone,
                        "resolved_revision": getattr(model_config, "_commit_hash", None),
                        "adapters_disabled": True,
                        "rollout_decoding": "native_generation_config",
                        "generation_overrides": config.generation,
                        "chat_template_kwargs": config.chat_template_kwargs},
        "seed": seed,
        "input_sha256": input_hashes or {},
        "files": {
            "trajectories.jsonl": file_digest(trajectory_path),
            "head_examples.jsonl": file_digest(example_path),
        },
    }
    write_json(output / "manifest.json", manifest)
    if not collection.state["complete"]:
        collection.commit({}, complete=True)
    return manifest
