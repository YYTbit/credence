import hashlib
import json
import os
import random
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .decoding import DECODING_PROTOCOL


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolve_checkpoint(path):
    root = Path(path)
    pointer = root / "checkpoint.json"
    if not pointer.is_file():
        return root
    name = json.loads(pointer.read_text(encoding="utf-8"))["generation"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Invalid checkpoint generation")
    selected = root / "generations" / name
    if not (selected / "complete.json").is_file():
        raise ValueError("Checkpoint generation is incomplete")
    return selected


def compatible_checkpoint(config, path, resume=False):
    path = resolve_checkpoint(path)
    saved = Config.load(path / "config.yaml")
    architecture = ("backbone", "embedding_dim", "predictor_hidden", "context_budget",
                    "adapter_rank", "adapter_alpha")
    for name in architecture:
        if getattr(config, name) != getattr(saved, name):
            raise ValueError(f"Requested {name} differs from the checkpoint")
    if config.revision is not None and saved.revision is not None and config.revision != saved.revision:
        raise ValueError("Requested backbone revision differs from the checkpoint")
    if resume:
        current, previous = asdict(config), asdict(saved)
        for name in ("device", "revision"):
            current.pop(name)
            previous.pop(name)
        if current != previous:
            raise ValueError("Training resume requires the saved training configuration")
        if (path / "training.pt").is_file():
            header = path / "complete.json"
            if not header.is_file() or json.loads(header.read_text()).get("decoding_protocol") != DECODING_PROTOCOL:
                raise ValueError("Training resume requires the matching decoding protocol")
    if config.revision is None:
        config.revision = saved.revision
    return path


def save_bundle(models, output, training_state=None, evaluation_state=None):
    """Commit weights and their matching state through one atomic pointer."""
    root = Path(output)
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex
    staging = generations / (name + ".tmp")
    models.save(staging)
    if training_state is not None:
        torch.save(training_state, staging / "training.pt")
    if evaluation_state is not None:
        torch.save(evaluation_state, staging / "evaluation.pt")
    state = training_state or {}
    write_json(staging / "complete.json", {"generation": name, "stage": state.get("stage", "joint"),
                                           "run_identity": state.get("run_identity"),
                                           "sft_identity": state.get("identity"),
                                           "training_complete": state.get("complete"),
                                           "decoding_protocol": DECODING_PROTOCOL})
    os.replace(staging, generations / name)
    previous = resolve_checkpoint(root)
    write_json(root / "checkpoint.json", {"generation": name})
    # Retain the previous committed generation until the new pointer is durable.
    for item in generations.iterdir():
        if item.name != name and item != previous:
            shutil.rmtree(item)
    return generations / name


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
