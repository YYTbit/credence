import json

import torch
import pytest

from credence.config import Config
from credence.credit import Calibrator
from credence.environments import WorkspaceEnvironment
from credence.groups import Sample
from credence.runtime import EpisodeUnavailable, Runtime
from credence.store import Store


class RecordedResponses:
    def __init__(self, responses):
        self.responses = list(responses)

    def prompt_ids(self, prompt):
        return list(prompt.encode())

    def generate(self, head, prompt, limit, seed, schema, sample):
        response = self.responses.pop(0)
        return Sample(prompt, self.prompt_ids(prompt), list(response.encode()), response,
                      torch.zeros(len(response)), schema)

    def encode(self, texts, grad=False):
        return torch.tensor([[1.0, 0.0] for _ in texts]).reshape(-1, 2)


def test_episode_reuses_executable_asset_and_official_session_grade(tmp_path):
    store = Store()
    aid = store.add("tool", "def count(items):\n    return len(items)", "Count", ["count items"], 10, "builder")
    filename = aid.replace("-", "_")
    models = RecordedResponses([
        "Count items",
        json.dumps({"action": "py", "content": f"import {filename}; print({filename}.count([2, 3]))"}),
        json.dumps({"action": "final", "content": "2"}),
    ])
    runtime = Runtime(models, Calibrator(), Config(embedding_dim=2, max_steps=3))
    task = {"benchmark": "scienceworld", "task_id": "software-fixture", "text": "Count two items", "expected": "2"}
    record = runtime.run_episode(task, store, WorkspaceEnvironment(), tmp_path, 11)
    assert record.completed
    assert record.score == 1
    assert record.assembled == [aid]
    assert record.evidence == ["2\n"]
    assert store.assets[aid].credit.count == 0


def test_episode_budget_exhaustion_preserves_partial_grade(tmp_path):
    models = RecordedResponses([])
    config = Config(embedding_dim=2, token_budget=1)
    runtime = Runtime(models, Calibrator(), config)
    record = runtime.run_episode({"text": "Task", "expected": "answer"}, Store(), WorkspaceEnvironment(), tmp_path, 0)
    assert not record.completed
    assert record.score == 0
    assert record.tokens <= config.token_budget


def test_execution_timeout_allows_an_official_grade(tmp_path):
    class Session:
        initial_observation = ""
        def step(self, action):
            raise TimeoutError("Timeout")
        def grade(self, answer):
            return float(answer == "2")
        def close(self):
            pass
    class Environment:
        def start(self, task, workspace, seed):
            return Session()
    models = RecordedResponses(["Compute", json.dumps({"action": "py", "content": "print(2)"}),
                                json.dumps({"action": "final", "content": "2"})])
    runtime = Runtime(models, Calibrator(), Config(embedding_dim=2))
    record = runtime.run_episode({"text": "Compute"}, Store(), Environment(), tmp_path, 1)
    assert record.completed and record.score == 1
    assert "time limit" in record.evidence[0]


def test_unavailable_grade_never_becomes_a_numeric_score(tmp_path):
    closed = []
    class Session:
        initial_observation = ""
        def grade(self, answer):
            raise TimeoutError("Grader unavailable")
        def close(self):
            closed.append(True)
    class Environment:
        def start(self, task, workspace, seed):
            return Session()
    models = RecordedResponses(["Compute", json.dumps({"action": "final", "content": "2"})])
    runtime = Runtime(models, Calibrator(), Config(embedding_dim=2))
    with pytest.raises(EpisodeUnavailable):
        runtime.run_episode({"text": "Compute"}, Store(), Environment(), tmp_path, 1)
    assert closed == [True]
