import hashlib
import json
import random
from pathlib import Path

from .credit import Calibrator
from .runtime import EpisodeUnavailable, Runtime, SCHEMAS, TokenBudget
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


def _write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _response(sample):
    value = json.loads(sample.text)
    if not isinstance(value, dict):
        raise ValueError("Head targets require a JSON object")
    return value


def _target(runtime, head, episode, store, seed, asset_id=None):
    prompt = runtime.prompt(head, episode, store, asset_id)
    budget = TokenBudget(runtime.config.token_budget)
    sample = runtime.complete("agent", prompt, runtime.config.policy_tokens, budget,
                              seed, SCHEMAS[head], sample=True)
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
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Warm-start output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_factory(models, Calibrator(config.calibration_quantile, config.reuse_margin), config)
    trajectories, examples, rejected = [], [], dict.fromkeys(HEADS, 0)
    for index, task in enumerate(plan):
        row_seed = seed + index
        store = initial_store(models, config, row_seed)
        try:
            episode = runtime.run_episode(task, store, environments[task["benchmark"]],
                                          output / "workspaces" / task["benchmark"] / str(index), row_seed)
        except EpisodeUnavailable as exc:
            raise RuntimeError(f"Base-model rollout unavailable for {task['benchmark']}:{task['task_id']}") from exc
        trajectory_id = f"warm-{index:06d}"
        trajectories.append({"trajectory_id": trajectory_id, "benchmark": task["benchmark"],
                             "task_id": task["task_id"], "split": "train", "seed": row_seed,
                             "task": task, "score": episode.score, "subtasks": episode.subtasks,
                             "assembled": episode.assembled, "actions": episode.trajectory,
                             "evidence": episode.evidence, "tokens": episode.tokens,
                             "completed": episode.completed, "initial_store": store.state_dict()})
        for head in ("writer", "builder"):
            try:
                example = _target(runtime, head, episode, store, row_seed + 1000 + HEADS.index(head))
            except (ValueError, TypeError, KeyError, AttributeError):
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
            except (ValueError, TypeError, KeyError, AttributeError):
                rejected["curator"] += 1
                continue
            example.update({"trajectory_id": trajectory_id, "benchmark": task["benchmark"],
                            "task_id": task["task_id"], "split": "train", "rollout_seed": row_seed})
            examples.append(example)
    counts = {head: sum(row["head"] == head for row in examples) for head in HEADS}
    if any(counts[head] == 0 for head in HEADS):
        raise ValueError(f"No validated warm-start targets for every head: {counts}")
    trajectory_path = output / "trajectories.jsonl"
    example_path = output / "head_examples.jsonl"
    _write_jsonl(trajectory_path, trajectories)
    _write_jsonl(example_path, examples)
    model_config = getattr(models.model, "config", None)
    manifest = {
        "protocol": "CREDENCE Appendix B.10 base-model rollout warm start",
        "requested_trajectories": count,
        "collected_trajectories": len(trajectories),
        "round_robin_benchmarks": list(TRAIN_BENCHMARKS),
        "trajectories_per_benchmark": {name: sum(row["benchmark"] == name for row in trajectories)
                                       for name in TRAIN_BENCHMARKS},
        "split": "train",
        "excluded_nontrain_records": excluded,
        "head_example_counts": counts,
        "rejected_targets": rejected,
        "base_policy": {"backbone": config.backbone,
                        "resolved_revision": getattr(model_config, "_commit_hash", None),
                        "adapters_disabled": True,
                        "rollout_decoding": "greedy runtime policy; the paper does not specify a decoding temperature"},
        "seed": seed,
        "input_sha256": input_hashes or {},
        "files": {
            "trajectories.jsonl": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
            "head_examples.jsonl": hashlib.sha256(example_path.read_bytes()).hexdigest(),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    return manifest
