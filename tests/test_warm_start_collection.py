import json
import hashlib
from types import SimpleNamespace

import pytest

from credence.cli import main
from credence.config import Config
from credence.artifacts import summarize_episodes
from credence.runtime import Episode
from credence.training import TRAIN_BENCHMARKS
from credence.warm_start import collect_warm_start, _round_robin


class FakeModels:
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision-1"))

    def token_count(self, text):
        return len(text.split())


class FakeRuntime:
    def __init__(self, models, calibrator, config):
        self.config = config
        self.calls = []

    def run_episode(self, task, store, environment, workspace, seed):
        assert task["split"] == "train"
        assert environment is not None
        return Episode(task, store.clone(), ["solve task"], ["memory-00000001"],
                       0.5, [{"action": "final", "content": "done"}], ["observed"], 12, seed, True)

    def prompt(self, head, episode, store, asset_id=None):
        return f"{head}:{asset_id or ''}"

    def complete(self, head, prompt, limit, budget, seed, schema=None, sample=True):
        self.calls.append((head, prompt, schema, seed))
        responses = {
            "writer": {"title": "Procedure", "text": "Read the task carefully."},
            "builder": {"name": "count_items", "code": "def count_items(items):\n    return len(items)"},
            "curator": {"asset_id": "memory-00000001", "action": "keep", "target": None},
        }
        return SimpleNamespace(text=json.dumps(responses["writer" if prompt.startswith("writer")
                                                        else "builder" if prompt.startswith("builder")
                                                        else "curator"]))

    def apply(self, head, sample, episode, store, asset_id=None):
        row = json.loads(sample.text)
        assert set(row) == set({"writer": ("title", "text"), "builder": ("name", "code"),
                                "curator": ("asset_id", "action", "target")}[head])


def test_collection_uses_four_train_distributions_and_writes_provenance(tmp_path):
    tasks = [{"benchmark": name, "task_id": f"{name}-train", "split": "train", "text": "Task"}
             for name in TRAIN_BENCHMARKS]
    tasks.append({"benchmark": "scienceworld", "task_id": "scienceworld-test", "split": "test", "text": "Test"})
    config = SimpleNamespace(seed=3, calibration_quantile=0.9, reuse_margin=0.25,
                             token_budget=1000, policy_tokens=32, backbone="base/model",
                             store_capacity=10, restoration_horizon=10)
    manifest = collect_warm_start(FakeModels(), dict.fromkeys(TRAIN_BENCHMARKS, object()), config,
                                  tasks, tmp_path / "warm", count=8, runtime_factory=FakeRuntime)
    trajectories = [json.loads(line) for line in (tmp_path / "warm" / "trajectories.jsonl").read_text().splitlines()]
    examples = [json.loads(line) for line in (tmp_path / "warm" / "head_examples.jsonl").read_text().splitlines()]
    assert [row["benchmark"] for row in trajectories] == list(TRAIN_BENCHMARKS) * 2
    assert all(row["split"] == "train" for row in trajectories + examples)
    assert manifest["collected_trajectories"] == 8
    assert manifest["excluded_nontrain_records"] == 1
    assert manifest["head_example_counts"] == {"writer": 8, "builder": 8, "curator": 8}
    assert manifest["base_policy"] == {"backbone": "base/model", "resolved_revision": "revision-1",
                                       "adapters_disabled": True,
                                       "rollout_decoding": "greedy runtime policy; the paper does not specify a decoding temperature"}
    for head in ("writer", "builder", "curator"):
        schemas = [row for row in examples if row["head"] == head]
        assert all(isinstance(json.loads(row["response"]), dict) for row in schemas)


def test_round_robin_rejects_missing_train_distribution_and_split_overlap():
    tasks = [{"benchmark": name, "task_id": f"{name}-0", "split": "train"} for name in TRAIN_BENCHMARKS]
    with pytest.raises(ValueError, match="missing"):
        _round_robin(tasks[:-1], 4, 0)
    tasks.append({"benchmark": "scienceworld", "task_id": "scienceworld-0", "split": "test"})
    with pytest.raises(ValueError, match="overlap"):
        _round_robin(tasks, 4, 0)


def test_collection_cli_wires_configured_count_and_input_records(tmp_path, monkeypatch, capsys):
    from credence.models import LanguageModels
    import credence.warm_start

    config_path = tmp_path / "config.yaml"
    Config(device="cpu", dtype="float32", embedding_dim=4, predictor_hidden=8).save(config_path)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"benchmark": name, "task_id": name,
                                               "split": "train", "text": "Task"})
                                 for name in TRAIN_BENCHMARKS))
    environments = tmp_path / "environments.json"
    environments.write_text("{}")
    calls = []
    monkeypatch.setattr(LanguageModels, "load", classmethod(lambda cls, config: FakeModels()))
    monkeypatch.setattr("credence.cli.load_environments", lambda path: {name: object() for name in TRAIN_BENCHMARKS})
    def collect(models, envs, config, records, output, count, seed, input_hashes=None):
        calls.append((count, seed, len(records), sorted(envs), input_hashes))
        return {"collected_trajectories": count}
    monkeypatch.setattr(credence.warm_start, "collect_warm_start", collect)
    main(["collect-warm-start", "--config", str(config_path), "--tasks", str(tasks),
          "--environments", str(environments), "--output", str(tmp_path / "out"),
          "--trajectories", "8", "--seed", "9"])
    assert calls == [(8, 9, 4, sorted(TRAIN_BENCHMARKS),
                      {"tasks": hashlib.sha256(tasks.read_bytes()).hexdigest(),
                       "environments": hashlib.sha256(environments.read_bytes()).hexdigest()})]
    assert json.loads(capsys.readouterr().out)["collected_trajectories"] == 8


def test_episode_artifacts_use_recorded_scores_and_report_unavailable_rows(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(json.dumps(row) for row in [
        {"benchmark": "scienceworld", "status": "scored", "score": 0.25, "tokens": 10, "store_size": 2},
        {"benchmark": "scienceworld", "status": "scored", "score": 0.75, "tokens": 30, "store_size": 4},
        {"benchmark": "webshop", "status": "unavailable", "score": None},
    ]))
    summary = summarize_episodes(episodes, tmp_path / "plots")
    assert summary["scienceworld"]["mean_normalized_score"] == pytest.approx(0.5)
    assert summary["scienceworld"]["sample_std_normalized_score"] == pytest.approx(0.353553, rel=1e-5)
    assert summary["webshop"]["unavailable"] == 1
    assert "0.50" in (tmp_path / "plots" / "episode_scores.svg").read_text()
    main(["artifacts", "--episodes", str(episodes), "--output", str(tmp_path / "cli-plots")])
    assert (tmp_path / "cli-plots" / "episode_summary.json").is_file()
