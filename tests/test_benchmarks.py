import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from credence import benchmarks


def module(monkeypatch, name, **attributes):
    value = ModuleType(name)
    value.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, value)
    return value


def test_scienceworld_grade_uses_cumulative_score(monkeypatch, tmp_path):
    class Environment:
        def __init__(self, **options):
            self.loaded = None
        def load(self, name, variation, simplification):
            self.loaded = (name, variation, simplification)
        def reset(self):
            return "initial", {"score": 0}
        def step(self, action):
            return "observation", 5, False, {"score": 40}
        def close(self):
            pass
    module(monkeypatch, "scienceworld", ScienceWorldEnv=Environment)
    backend = benchmarks.ScienceWorldBackend({"parameters": {"task_name": "test", "variation": 3}}, tmp_path, 1, {})
    assert backend.env.loaded == ("test", 3, "")
    backend.action("step", {"action": "look around"})
    assert backend.grade("") == 0.4


def test_webshop_preserves_released_goal_and_terminal_score(monkeypatch, tmp_path):
    class Environment:
        def __init__(self, **options):
            pass
        def reset(self, session):
            self.goal = session
            return "shop", None
        def step(self, action):
            return "purchased", 0.75, True, None
        def close(self):
            pass
    module(monkeypatch, "web_agent_site.envs.web_agent_text_env", WebAgentTextEnv=Environment)
    backend = benchmarks.WebShopBackend({"parameters": {"goal_id": 42}}, tmp_path, 1, {})
    assert backend.env.goal == 42
    backend.action("step", {"action": "click[Buy Now]"})
    assert backend.grade("") == 0.75


def test_api_bank_delegates_call_comparison_to_released_checker(monkeypatch, tmp_path):
    class API:
        def check_api_call_correctness(self, result, expected):
            return result == expected
    class Manager:
        def get_api_description(self, name):
            return name
        def api_call(self, name, **parameters):
            return parameters
        def init_tool(self, name):
            return API()
    module(monkeypatch, "fixture_manager", ToolManager=Manager)
    task = {"parameters": {"ground_truth": {"api_name": "Add", "result": {"x": 2}}}}
    backend = benchmarks.APIBankBackend(task, tmp_path, 1, {"manager_module": "fixture_manager"})
    backend.action("call", {"name": "Add", "parameters": {"x": 2}})
    assert backend.grade("") == 1


def test_api_bank_counts_incorrect_and_extra_call_attempts(monkeypatch, tmp_path):
    class Manager:
        def get_api_description(self, name):
            return name
        def api_call(self, name, **parameters):
            if name != "Add":
                raise KeyError(name)
            return parameters
        def init_tool(self, name):
            return SimpleNamespace(check_api_call_correctness=lambda actual, expected: actual == expected)
    module(monkeypatch, "fixture_manager", ToolManager=Manager)
    task = {"parameters": {"ground_truth": {"api_name": "Add", "result": {"x": 2}}}}
    backend = benchmarks.APIBankBackend(task, tmp_path, 1, {"manager_module": "fixture_manager"})
    response = backend.action("call", {"name": "Wrong", "parameters": {}})
    assert response["exception"] == "KeyError"
    backend.action("call", {"name": "Add", "parameters": {"x": 2}})
    assert len(backend.calls) == 2
    assert backend.grade("") == 0
    backend.calls = []
    backend.action("call", {"name": "Add", "parameters": {"x": 2}})
    backend.action("call", {"name": "Add", "parameters": {"x": 2}})
    assert backend.grade("") == 0.5


def test_appworld_grade_uses_task_success(monkeypatch, tmp_path):
    class World:
        def __init__(self, **options):
            self.options = options
            self.task = SimpleNamespace(instruction="Task")
        def execute(self, code):
            return code
        def evaluate(self, suppress_errors):
            assert not suppress_errors
            return SimpleNamespace(success=True, pass_percentage=100)
        def close(self):
            pass
    module(monkeypatch, "appworld", AppWorld=World)
    backend = benchmarks.AppWorldBackend({"task_id": "task"}, tmp_path, 17, {})
    assert backend.world.options["random_seed"] == 17
    assert backend.grade("") == 1


def test_native_workspace_executes_sdk_calls_in_owner_thread(monkeypatch, tmp_path):
    owner = threading.get_ident()
    class Backend:
        def __init__(self, task, workspace, seed, options):
            self.initial_observation = "Fixture environment"
            self.actions = []
        def action(self, name, arguments):
            assert threading.get_ident() == owner
            self.actions.append((name, arguments))
            return "value"
        def grade(self, answer):
            return 0.5
        def close(self):
            pass
    monkeypatch.setitem(benchmarks.BACKENDS, "scienceworld", Backend)
    session = benchmarks.NativeSession("scienceworld", {}, tmp_path, 3, {})
    try:
        observation = session.step({"action": "py", "content": "import benchmark; print(benchmark.step('inspect'))"})
        assert observation.strip() == "value"
        assert session.backend.actions == [("step", {"action": "inspect"})]
        assert session.grade("answer") == 0.5
    finally:
        session.close()


def test_native_environment_factory_selects_isolated_worker():
    env = benchmarks.ScienceWorldEnvironment(step_limit=100)
    assert env.command[1:3] == ["-m", "credence.benchmarks"]
    assert "scienceworld" in env.command


def test_toolbench_uses_official_tool_eval_contract(monkeypatch, tmp_path):
    from enum import Enum
    class Status(Enum):
        Solved = "Solved"
        Unsolved = "Unsolved"
    class Passed(Enum):
        Passed = "Passed"
        Failed = "Failed"
        Unsure = "Unsure"
    class Environment:
        def __init__(self, query, descriptions, retriever, arguments, process_id):
            self.functions = [{"name": "Finish", "description": "Finish task"}]
        def step(self, **arguments):
            return "result", 3
    class Evaluator:
        def check_is_solved(self, query, answer, return_reason):
            assert answer["answer_details"][0]["message"].startswith("{'name': 'Finish'")
            return Status.Solved, "reason"
        def check_task_solvable(self, query, has_been_solved, return_reason):
            return "Solvable", "reason"
        def is_passed(self, query, answer, answer_status, task_status):
            assert answer_status == Status.Solved
            return Passed.Passed
    module(monkeypatch, "toolbench.inference.Downstream_tasks.rapidapi", rapidapi_wrapper=Environment)
    module(monkeypatch, "evaluators", load_registered_automatic_evaluator=lambda **options: Evaluator())
    module(monkeypatch, "evaluators.registered_cls.rtl", AnswerStatus=Status, AnswerPass=Passed)
    backend = benchmarks.ToolBenchBackend({"parameters": {"query_json": {"query": "Task"}}}, tmp_path, 1,
                                         {"sdk_path": str(tmp_path), "tool_root_dir": str(tmp_path)})
    assert backend.grade("") == 0
    backend.action("call", {"name": "Finish", "parameters": {"return_type": "give_answer", "final_answer": "done"}})
    assert backend.grade("done") == 1


def test_native_api_failure_remains_an_observation_with_a_grade(monkeypatch, tmp_path):
    class Manager:
        def get_api_description(self, name):
            return name
        def api_call(self, name, **parameters):
            raise KeyError(name)
        def init_tool(self, name):
            return SimpleNamespace(check_api_call_correctness=lambda actual, expected: actual == expected)
    module(monkeypatch, "fixture_manager", ToolManager=Manager)
    task = {"parameters": {"ground_truth": {"api_name": "Add", "result": {"x": 2}}}}
    session = benchmarks.NativeSession("api_bank", task, tmp_path, 1, {"manager_module": "fixture_manager"})
    try:
        observation = session.step({"action": "py", "content": "import benchmark; print(benchmark.call('Wrong'))"})
        assert "KeyError" in observation
        assert session.grade("done") == 0
        assert session.backend.calls == [("Wrong", None)]
    finally:
        session.close()
