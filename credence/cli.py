import argparse
import json
import math
from pathlib import Path

from .config import Config
from .environments import load_environments, load_tasks
from .checkpoint import compatible_checkpoint, digest, file_digest, resolve_checkpoint


def parser():
    root = argparse.ArgumentParser(prog="credence")
    commands = root.add_subparsers(dest="command", required=True)
    sft = commands.add_parser("sft", help="Train the three warm-start adapters from recorded head examples")
    sft.add_argument("--config", required=True)
    sft.add_argument("--examples", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--epochs", type=int, default=1)
    sft.add_argument("--learning-rate", type=float, default=5e-5)
    sft.add_argument("--resume")
    sft.add_argument("--checkpoint-every", type=int, default=100)
    collect = commands.add_parser("collect-warm-start", help="Collect base-model rollouts and validated head targets")
    collect.add_argument("--config", required=True)
    collect.add_argument("--tasks", required=True)
    collect.add_argument("--environments", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--trajectories", type=int, default=1000)
    collect.add_argument("--seed", type=int, default=None)
    train = commands.add_parser("train", help="Train CREDENCE with matched continuations and delayed rewards")
    train.add_argument("--config", required=True)
    train.add_argument("--tasks", required=True)
    train.add_argument("--environments", required=True)
    starts = train.add_mutually_exclusive_group(required=True)
    starts.add_argument("--warm-start")
    starts.add_argument("--resume")
    train.add_argument("--output", required=True)
    train.add_argument("--episodes-per-stream", type=int, default=None,
                       help="Override the configured epoch schedule with a fixed stream length")
    train.add_argument("--checkpoint-every", type=int, default=10)
    run = commands.add_parser("run", help="Execute official evaluation tasks using a trained checkpoint")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--config")
    run.add_argument("--tasks", required=True)
    run.add_argument("--environments", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--seed", type=int, default=149)
    for command in (sft, collect, train, run):
        command.add_argument("--device", default=None)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    checkpoint = (getattr(args, "checkpoint", None) or getattr(args, "resume", None)
                  or getattr(args, "warm_start", None))
    config_path = (args.config or resolve_checkpoint(checkpoint) / "config.yaml")
    config = Config.load(config_path)
    if getattr(args, "device", None):
        config.device = args.device
    if checkpoint is not None:
        compatible_checkpoint(config, checkpoint, resume=bool(getattr(args, "resume", None)))
    tasks, environments, examples = None, None, None
    input_identity = None
    if args.command == "sft":
        if args.epochs < 1 or args.checkpoint_every < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("SFT epochs, learning rate and checkpoint interval require positive values")
        examples = [json.loads(line) for line in Path(args.examples).read_text(encoding="utf-8").splitlines() if line.strip()]
        if {row.get("head") for row in examples} != {"writer", "builder", "curator"}:
            raise ValueError("SFT examples require all three heads")
        if args.resume is None and (Path(args.output) / "checkpoint.json").exists():
            raise FileExistsError("SFT output already contains a checkpoint; use --resume")
        if args.resume:
            header = resolve_checkpoint(args.resume) / "complete.json"
            if header.is_file():
                metadata = json.loads(header.read_text(encoding="utf-8"))
                identity = {"records": digest(examples), "epochs": args.epochs, "learning_rate": args.learning_rate}
                if metadata["stage"] != "sft" or metadata["sft_identity"] != identity:
                    raise ValueError("SFT resume requires its original examples and training settings")
    else:
        tasks = load_tasks(args.tasks)
        environments = load_environments(args.environments)
        missing = {task["benchmark"] for task in tasks} - set(environments)
        if missing:
            raise ValueError(f"Environment definitions missing for {sorted(missing)}")
        input_identity = {"tasks": file_digest(args.tasks), "environments": file_digest(args.environments)}
    if args.command == "collect-warm-start":
        from .warm_start import _round_robin
        _round_robin(tasks, args.trajectories,
                     config.seed if args.seed is None else args.seed)
    if args.command == "train":
        from .training import TRAIN_BENCHMARKS
        available = {task["benchmark"] for task in tasks if task["split"] == "train"}
        if set(TRAIN_BENCHMARKS) - available:
            raise ValueError("Joint training requires all four training distributions")
        if ((args.episodes_per_stream is not None and args.episodes_per_stream < 1)
                or args.checkpoint_every < 1):
            raise ValueError("Training and checkpoint counts require positive values")
        if args.resume is None and (Path(args.output) / "checkpoint").exists():
            raise FileExistsError("Training output already contains a checkpoint; use --resume")
        if args.warm_start:
            header = resolve_checkpoint(args.warm_start) / "complete.json"
            if header.is_file():
                metadata = json.loads(header.read_text(encoding="utf-8"))
                if metadata["stage"] != "sft" or not metadata["training_complete"]:
                    raise ValueError("Joint training starts from a completed SFT checkpoint")
        if args.resume:
            header = resolve_checkpoint(args.resume) / "complete.json"
            if header.is_file():
                metadata = json.loads(header.read_text(encoding="utf-8"))
                if args.episodes_per_stream is None and metadata.get("run_identity") is not None:
                    args.episodes_per_stream = metadata["run_identity"]["episodes_per_stream"]
                identity = {"tasks": digest(tasks), "episodes_per_stream": args.episodes_per_stream,
                            "inputs": input_identity}
                if metadata["stage"] != "joint" or metadata["run_identity"] != identity:
                    raise ValueError("Joint resume requires its original tasks, environment inputs and schedule")
    if args.command == "run":
        if not any(task["split"] != "train" for task in tasks):
            raise ValueError("Evaluation requires tasks from an evaluation split")
        if any(task["benchmark"] == "appworld" and task["split"] != "test_normal" for task in tasks):
            raise ValueError("AppWorld transfer uses Test-Normal tasks")
    from .models import LanguageModels
    from .training import Trainer, supervised_warm_start
    models = LanguageModels.load(config, checkpoint=checkpoint) if checkpoint is not None else LanguageModels.load(config)
    if args.command == "collect-warm-start":
        from .warm_start import collect_warm_start
        manifest = collect_warm_start(models, environments, config, tasks, args.output,
                                      args.trajectories, args.seed, input_hashes=input_identity)
        print(json.dumps(manifest, ensure_ascii=False))
        return
    if args.command == "sft":
        supervised_warm_start(models, examples, args.output, args.epochs, args.learning_rate,
                              args.resume, args.checkpoint_every)
        return
    if args.command == "train":
        trainer = Trainer(models, environments, config, args.output)
        if args.resume:
            trainer.restore(args.resume)
        statistics = trainer.train(tasks, args.episodes_per_stream, args.checkpoint_every, input_identity)
        print(json.dumps(statistics))
        return
    from .evaluation import evaluate
    print(json.dumps(evaluate(models, environments, config, tasks, args.checkpoint,
                              args.output, args.seed, input_identity), allow_nan=False))


if __name__ == "__main__":
    main()
