import json
import sys

import pytest

from credence.cli import main
from credence.environments import CommandEnvironment, WorkspaceEnvironment, load_tasks, normalized_score


def test_workspace_fixture_executes_and_grades(tmp_path):
    task = {"expected": "5", "files": {"input.txt": "2 3"}}
    session = WorkspaceEnvironment().start(task, tmp_path, seed=11)
    observation = session.step({"action": "py", "content": "print(2 + 3)"})
    assert observation.strip() == "5"
    assert session.grade("5") == 1
    session.close()


def test_workspace_paths_remain_relative(tmp_path):
    with pytest.raises(ValueError):
        WorkspaceEnvironment().start({"files": {"../escape.txt": "x"}}, tmp_path, 0)


def test_command_environment_routes_task_seed_and_score(tmp_path):
    script = tmp_path / "bridge.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request['method'] == 'close': break\n"
        "    if request['method'] == 'reset':\n"
        "        result = {'observation': str(request['seed'])}\n"
        "    elif request['method'] == 'step':\n"
        "        result = {'observation': request['action']['content']}\n"
        "    else:\n"
        "        result = {'score': 0.5}\n"
        "    print(json.dumps(result), flush=True)\n"
    )
    session = CommandEnvironment([sys.executable, str(script)]).start({"task_id": "x"}, tmp_path, 42)
    assert session.initial_observation == "42"
    assert session.step({"action": "py", "content": "value"}) == "value"
    assert session.grade("answer") == 0.5
    session.close()


@pytest.mark.parametrize("value", [1.1, -0.1, float("nan"), float("inf")])
def test_grade_requires_normalized_official_value(value):
    with pytest.raises(ValueError):
        normalized_score(value)


def test_validate_runs_without_loading_backbone(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("backbone: unused-model\n")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"benchmark": "scienceworld", "task_id": "x", "split": "train", "text": "Task"}) + "\n")
    environments = tmp_path / "environments.json"
    environments.write_text(json.dumps({"scienceworld": {"factory": "credence.environments:WorkspaceEnvironment"}}))
    main(["validate", "--config", str(config), "--tasks", str(tasks), "--environments", str(environments)])
    result = json.loads(capsys.readouterr().out)
    assert result["tasks"] == 1
    assert result["summary_dimension"] == 6174


def test_task_loader_requires_explicit_split(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"benchmark": "scienceworld", "task_id": "x", "text": "Task"}) + "\n")
    with pytest.raises(ValueError):
        load_tasks(path)
