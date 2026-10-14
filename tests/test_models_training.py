import copy
import json

import pytest
import torch

from credence.groups import Candidate, CandidateGroup
from credence.runtime import Episode, EpisodeUnavailable, TokenBudget
from credence.store import Store
from credence.training import Trainer, Stream, make_streams, supervised_warm_start


def task(benchmark="scienceworld", number=0):
    return {"benchmark": benchmark, "task_id": f"task-{number}", "split": "train", "text": f"Task {number}"}


def episode(models, benchmark="scienceworld", number=0):
    store = Store()
    aid = store.add("memory", "Read the task carefully", "Read", ["read task"], 8, "writer")
    return Episode(task(benchmark, number), store, ["read task"], [aid], 0.5)


def test_behavior_probabilities_match_replayed_sampling_policy(language_models):
    sample = language_models.generate("writer", "Write a memory", 5, seed=24)
    replay = language_models.logprobs("writer", sample, grad=False).cpu()
    assert torch.allclose(sample.behavior_logprobs, replay, atol=1e-5)


def test_masked_behavior_probabilities_match_replayed_policy(language_models):
    schema = {"type": "object", "properties": {"choice": {"enum": ["a", "b"]}},
              "required": ["choice"], "additionalProperties": False}
    sample = language_models.generate("writer", "Choose", 20, seed=3, schema=schema)
    replay = language_models.logprobs("writer", sample, grad=False).cpu()
    assert torch.allclose(sample.behavior_logprobs, replay, atol=1e-5)


def test_causal_encoder_representation_uses_complete_text(language_models):
    values = language_models.encode(["Read first", "Read second"])
    assert not torch.allclose(values[0], values[1])
    assert torch.allclose(values.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_three_heads_and_auxiliary_models_receive_updates(language_models, tmp_path):
    trainer = Trainer(language_models, {}, language_models.config, tmp_path)
    trainer.config.credit_horizon = 1
    before = {name: value.detach().clone() for name, value in language_models.model.named_parameters() if "lora_B" in name}
    projection = copy.deepcopy(language_models.projection.state_dict())
    predictor = copy.deepcopy(language_models.outcome.state_dict())
    for head in language_models.heads:
        candidates = []
        for i, reward in enumerate((0.4, -0.2)):
            sample = language_models.generate(head, "Shared decision", 3, seed=100 + i)
            store = Store()
            aid = store.add("memory", "Procedure", "Procedure", [], 10, head)
            store.observe(aid, reward)
            store.advance()
            candidates.append(Candidate(sample, store, aid, 10))
        group = CandidateGroup(head, head, 0, 0, 0, candidates)
        if head == "curator":
            group.reference_scores = [0.2]
            candidates[0].scores, candidates[1].scores = [0.6], [0.1]
        trainer.buffer.add(group)
    trainer.update([episode(language_models), episode(language_models, "webshop", 1)])
    after = dict(language_models.model.named_parameters())
    for head in language_models.heads:
        names = [name for name in before if f".{head}." in name]
        assert any(not torch.equal(before[name], after[name]) for name in names)
        assert trainer.head_updates[head] == 1
        refs = [name for name in before if f".{head}_ref." in name]
        assert all(torch.equal(before[name], after[name]) for name in refs)
    assert any(not torch.equal(value, language_models.projection.state_dict()[key]) for key, value in projection.items())
    assert any(not torch.equal(value, language_models.outcome.state_dict()[key]) for key, value in predictor.items())


def test_warmup_updates_predictor_with_encoder_preserved(language_models, tmp_path):
    trainer = Trainer(language_models, {}, language_models.config, tmp_path)
    before = copy.deepcopy(language_models.projection.state_dict())
    trainer.update([episode(language_models)], warmup=True)
    assert all(torch.equal(value, language_models.projection.state_dict()[key]) for key, value in before.items())
    assert trainer.head_updates == {"writer": 0, "builder": 0, "curator": 0}


def test_checkpoint_preserves_pending_groups_and_behavior_data(language_models, tmp_path):
    trainer = Trainer(language_models, {}, language_models.config, tmp_path)
    sample = language_models.generate("writer", "Decision", 2, seed=11)
    candidates = []
    for _ in range(2):
        record = episode(language_models)
        aid = record.assembled[0]
        candidates.append(Candidate(sample, record.store, aid, 8))
    trainer.buffer.add(CandidateGroup("pending", "writer", 0, 0, 0, candidates))
    trainer.streams = [Stream(0, "scienceworld", episode(language_models).store, [task()])]
    trainer.save(tmp_path / "checkpoint")
    original = copy.deepcopy(language_models.outcome.state_dict())
    with torch.no_grad():
        for parameter in language_models.outcome.parameters():
            parameter.zero_()
    resumed = Trainer(language_models, {}, language_models.config, tmp_path / "resumed")
    resumed.restore(tmp_path / "checkpoint")
    assert resumed.buffer.statistics()["pending_groups"] == 1
    assert torch.equal(resumed.buffer.pending[0].candidates[0].sample.behavior_logprobs, sample.behavior_logprobs)
    assert resumed.streams[0].order == [task()]
    assert all(torch.equal(value, language_models.outcome.state_dict()[key]) for key, value in original.items())


def test_warm_start_reference_matches_recorded_training_result(language_models, tmp_path):
    examples = [
        {"head": "writer", "prompt": "Write", "response": {"title": "Read", "text": "Read the task"}},
        {"head": "builder", "prompt": "Build", "response": {"name": "count", "code": "def count(items): return len(items)"}},
        {"head": "curator", "prompt": "Curate", "response": {"asset_id": "a", "action": "keep", "target": None}},
    ]
    supervised_warm_start(language_models, examples, tmp_path / "warm")
    sample = language_models.generate("writer", "Write", 2, seed=17)
    current = language_models.logprobs("writer", sample, grad=False)
    reference = language_models.logprobs("writer_ref", sample, grad=False)
    assert torch.allclose(current, reference)


def test_stream_pairs_share_tasks_with_separate_stores(language_models):
    tasks = [task(name, i) for name in ("scienceworld", "webshop", "api_bank", "toolbench") for i in range(3)]
    streams = make_streams(tasks, language_models, language_models.config, 5)
    assert len(streams) == 8
    for i in range(0, 8, 2):
        assert streams[i].order == streams[i + 1].order
        assert streams[i].store.assets != streams[i + 1].store.assets


def test_candidate_selection_precedes_matched_continuations(language_models, tmp_path, monkeypatch):
    trainer = Trainer(language_models, {"scienceworld": object()}, language_models.config, tmp_path)
    trainer.config.credit_horizon = 2
    record = episode(language_models)
    samples = [language_models.generate("writer", "prompt", 2, seed=20 + i) for i in range(2)]
    for i, sample in enumerate(samples):
        sample.text = json.dumps({"title": f"Candidate {i}", "text": f"Procedure {i}"})
    def complete(head, prompt, limit, budget, seed, schema=None):
        sample = samples.pop(0)
        sample.prompt = prompt
        return sample
    monkeypatch.setattr(trainer.runtime, "complete", complete)
    selected = trainer.create_group("writer", record, record.store, 0, 51, TokenBudget(100000))
    group = trainer.buffer.pending[0]
    assert len(selected.assets) == 2
    seen = []
    def collect(task, store, environment, workspace, seed):
        seen.append((task["task_id"], seed, [a.text for a in store.assets.values()]))
        return Episode(task, store.clone(), ["read task"], list(store.assets), 0.5)
    monkeypatch.setattr(trainer.runtime, "run_episode", collect)
    monkeypatch.setattr(trainer.runtime, "evolve", lambda record, store, seed: store)
    stream = Stream(0, "scienceworld", selected, [task()])
    trainer.advance_groups(stream, task(number=1), seed=123)
    trainer.advance_groups(stream, task(number=2), seed=124)
    assert [row[:2] for row in seen] == [("task-1", 123)] * 2 + [("task-2", 124)] * 2
    assert seen[0][2] != seen[1][2]
    assert all(candidate.reward is not None for candidate in group.candidates)


def test_unavailable_continuation_excludes_original_comparison(language_models, tmp_path, monkeypatch):
    trainer = Trainer(language_models, {"scienceworld": object()}, language_models.config, tmp_path)
    candidates = []
    for index in range(2):
        record = episode(language_models)
        sample = language_models.generate("writer", "Decision", 1, seed=index)
        candidates.append(Candidate(sample, record.store, record.assembled[0], 8))
    trainer.buffer.add(CandidateGroup("unavailable", "writer", 0, 0, 0, candidates))
    def unavailable(*args, **kwargs):
        raise EpisodeUnavailable("Grader unavailable")
    monkeypatch.setattr(trainer.runtime, "run_episode", unavailable)
    trainer.advance_groups(Stream(0, "scienceworld", Store(), []), task(), 11)
    assert trainer.buffer.pending == []
    assert trainer.buffer.unavailable == 1
    assert all(candidate.reward is None for candidate in candidates)


def test_unavailable_main_episodes_leave_training_labels_empty(language_models, tmp_path, monkeypatch):
    trainer = Trainer(language_models, dict.fromkeys(("scienceworld", "webshop", "api_bank", "toolbench"), object()),
                      language_models.config, tmp_path)
    def unavailable(*args, **kwargs):
        raise EpisodeUnavailable("Grader unavailable")
    monkeypatch.setattr(trainer.runtime, "run_episode", unavailable)
    tasks = [task(name) for name in ("scienceworld", "webshop", "api_bank", "toolbench")]
    trainer.train(tasks, episodes_per_stream=1)
    assert trainer.fitting == []
    assert trainer.calibration == []
    assert trainer.head_updates == {"writer": 0, "builder": 0, "curator": 0}
    assert [stream.position for stream in trainer.streams] == [1] * 8
    records = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert len(records) == 8 and all(record["event"] == "episode_unavailable" for record in records)
