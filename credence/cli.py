import argparse
import json
import random
from pathlib import Path

from .config import Config
from .environments import BENCHMARKS, load_environments, load_tasks


def parser():
    root = argparse.ArgumentParser(prog="credence")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate runtime inputs without loading a model")
    validate.add_argument("--config", required=True)
    validate.add_argument("--tasks", required=True)
    validate.add_argument("--environments", required=True)
    sft = commands.add_parser("sft", help="Train the three warm-start adapters from recorded head examples")
    sft.add_argument("--config", required=True)
    sft.add_argument("--examples", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--epochs", type=int, default=1)
    sft.add_argument("--learning-rate", type=float, default=5e-5)
    train = commands.add_parser("train", help="Train CREDENCE with matched continuations and delayed rewards")
    train.add_argument("--config", required=True)
    train.add_argument("--tasks", required=True)
    train.add_argument("--environments", required=True)
    starts = train.add_mutually_exclusive_group(required=True)
    starts.add_argument("--warm-start")
    starts.add_argument("--resume")
    train.add_argument("--output", required=True)
    train.add_argument("--episodes-per-stream", type=int, default=160)
    train.add_argument("--checkpoint-every", type=int, default=10)
    run = commands.add_parser("run", help="Execute official evaluation tasks using a trained checkpoint")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--tasks", required=True)
    run.add_argument("--environments", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--seed", type=int, default=149)
    for command in (sft, train, run):
        command.add_argument("--device", default=None)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    config_path = Path(args.checkpoint) / "config.yaml" if args.command == "run" else args.config
    config = Config.load(config_path)
    if getattr(args, "device", None):
        config.device = args.device
    if args.command == "validate":
        tasks = load_tasks(args.tasks)
        manifest = json.loads(Path(args.environments).read_text(encoding="utf-8"))
        missing = {task["benchmark"] for task in tasks} - set(manifest)
        if missing:
            raise ValueError(f"Environment definitions missing for {sorted(missing)}")
        for spec in manifest.values():
            if not isinstance(spec.get("factory"), str) or ":" not in spec["factory"]:
                raise ValueError("Environment factories use module:function notation")
        print(json.dumps({"tasks": len(tasks), "benchmarks": sorted({task['benchmark'] for task in tasks}),
                          "summary_dimension": config.summary_dim}))
        return
    import torch
    from .models import LanguageModels
    from .training import Trainer, supervised_warm_start
    torch.manual_seed(config.seed)
    models = LanguageModels.load(config)
    if args.command == "sft":
        if args.epochs < 1 or args.learning_rate <= 0:
            raise ValueError("Warm-start epochs and learning rate require positive values")
        examples = [json.loads(line) for line in Path(args.examples).read_text(encoding="utf-8").splitlines() if line.strip()]
        supervised_warm_start(models, examples, args.output, args.epochs, args.learning_rate)
        return
    tasks = load_tasks(args.tasks)
    environments = load_environments(args.environments)
    if args.command == "train":
        if args.episodes_per_stream < 1 or args.checkpoint_every < 1:
            raise ValueError("Training and checkpoint counts require positive values")
        trainer = Trainer(models, environments, config, args.output)
        if args.resume:
            trainer.restore(args.resume)
        else:
            models.restore(args.warm_start)
        statistics = trainer.train(tasks, args.episodes_per_stream, args.checkpoint_every)
        print(json.dumps(statistics))
        return
    from .runtime import EpisodeUnavailable, Runtime
    trainer = Trainer(models, environments, config, args.output)
    trainer.restore(args.checkpoint)
    runtime = Runtime(models, trainer.calibrator, config)
    from .training import initial_store
    output = Path(args.output)
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for index, benchmark in enumerate(BENCHMARKS):
            selected = [task for task in tasks if task["benchmark"] == benchmark and task["split"] != "train"]
            if benchmark == "appworld" and any(task["split"] != "test_normal" for task in selected):
                raise ValueError("AppWorld transfer uses Test-Normal tasks")
            random.Random(args.seed + index).shuffle(selected)
            trained = [stream.store for stream in trainer.streams if stream.benchmark == benchmark]
            base_store = trained[0].clone() if trained else initial_store(models, config, args.seed + index)
            for position, task in enumerate(selected):
                store = base_store.clone()
                store.advance()
                try:
                    episode = runtime.run_episode(task, store, environments[benchmark],
                                                  output / "workspaces" / benchmark / str(position), args.seed + position)
                except EpisodeUnavailable as exc:
                    handle.write(json.dumps({"benchmark": benchmark, "task_id": task["task_id"],
                                             "status": "unavailable", "score": None, "reason": str(exc)}) + "\n")
                    continue
                runtime.credit(episode, store)
                store = runtime.evolve(episode, store, args.seed + 50000 + position)
                handle.write(json.dumps({"benchmark": benchmark, "task_id": task["task_id"],
                                         "status": "scored", "score": episode.score, "tokens": episode.tokens,
                                         "store_size": len(store.assets)}, allow_nan=False) + "\n")
    (output / "diagnostics.jsonl").write_text("\n".join(json.dumps(row) for row in runtime.diagnostics), encoding="utf-8")


if __name__ == "__main__":
    main()
