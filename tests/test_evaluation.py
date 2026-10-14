import json
from types import SimpleNamespace

from credence import models, runtime, training
from credence.cli import main
from credence.config import Config
from credence.credit import Calibrator
from credence.store import Store


def test_evaluation_isolates_each_task_from_previous_test_feedback(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    Config().save(checkpoint / "config.yaml")
    initial = Store()
    initial.add("memory", "Training memory", "Memory", [], 2, "writer")
    model = SimpleNamespace(token_count=len)
    monkeypatch.setattr(models.LanguageModels, "load", lambda config: model)
    class Trainer:
        def __init__(self, models, environments, config, output):
            self.streams = [SimpleNamespace(benchmark="scienceworld", store=initial)]
            self.calibrator = Calibrator()
        def restore(self, path):
            pass
    seen = []
    class Runtime:
        def __init__(self, models, calibrator, config):
            self.diagnostics = []
        def run_episode(self, task, store, environment, workspace, seed):
            seen.append([asset.text for asset in store.assets.values()])
            return SimpleNamespace(score=0.5, tokens=5)
        def credit(self, episode, store):
            pass
        def evolve(self, episode, store, seed):
            store.add("memory", "Test feedback", "Feedback", [], 2, "writer")
            return store
    monkeypatch.setattr(training, "Trainer", Trainer)
    monkeypatch.setattr(runtime, "Runtime", Runtime)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"benchmark": "scienceworld", "task_id": str(i),
                                        "split": "test", "text": "Task"}) for i in range(2)))
    environments = tmp_path / "environments.json"
    environments.write_text(json.dumps({"scienceworld": {"factory": "credence.environments:WorkspaceEnvironment"}}))
    output = tmp_path / "output"
    output.mkdir()
    main(["run", "--checkpoint", str(checkpoint), "--tasks", str(tasks),
          "--environments", str(environments), "--output", str(output)])
    assert seen == [["Training memory"], ["Training memory"]]
    assert len(initial.assets) == 1
