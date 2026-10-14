import argparse
import contextlib
import importlib
import json
import os
import queue
import random
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from .environments import CommandEnvironment, normalized_score


class NativeEnvironment(CommandEnvironment):
    benchmark = None

    def __init__(self, sdk_python=None, timeout=120, **options):
        command = [sdk_python or sys.executable, "-m", "credence.benchmarks",
                   "--benchmark", self.benchmark, "--options", json.dumps(options)]
        super().__init__(command, timeout)


class ScienceWorldEnvironment(NativeEnvironment):
    benchmark = "scienceworld"


class WebShopEnvironment(NativeEnvironment):
    benchmark = "webshop"


class APIBankEnvironment(NativeEnvironment):
    benchmark = "api_bank"


class ToolBenchEnvironment(NativeEnvironment):
    benchmark = "toolbench"


class AppWorldEnvironment(NativeEnvironment):
    benchmark = "appworld"


class ScienceWorldBackend:
    def __init__(self, task, workspace, seed, options):
        from scienceworld import ScienceWorldEnv
        parameters = task["parameters"]
        self.env = ScienceWorldEnv(serverPath=options.get("jar_path"), envStepLimit=options.get("step_limit", 100))
        self.env.load(parameters["task_name"], parameters["variation"], parameters.get("simplification", ""))
        self.initial_observation, info = self.env.reset()
        self.score = max(0, info["score"]) / 100
        self.done = False

    def action(self, name, arguments):
        if name != "step":
            raise ValueError("ScienceWorld interface uses step(action)")
        if self.done:
            return "The environment has reached a terminal state."
        observation, _, self.done, info = self.env.step(arguments["action"])
        self.score = max(0, info["score"]) / 100
        return observation

    def grade(self, answer):
        return normalized_score(self.score)

    def close(self):
        self.env.close()


class WebShopBackend:
    def __init__(self, task, workspace, seed, options):
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        arguments = dict(options.get("environment", {}))
        arguments.setdefault("observation_mode", "text")
        arguments.setdefault("human_goals", True)
        self.env = WebAgentTextEnv(**arguments)
        self.initial_observation, _ = self.env.reset(session=int(task["parameters"]["goal_id"]))
        self.score, self.done = 0.0, False

    def action(self, name, arguments):
        if name != "step":
            raise ValueError("WebShop interface uses step(action)")
        if self.done:
            return "The environment has reached a terminal state."
        observation, reward, self.done, _ = self.env.step(arguments["action"])
        if self.done:
            self.score = normalized_score(reward)
        return observation

    def grade(self, answer):
        return self.score

    def close(self):
        self.env.close()


class APIBankBackend:
    def __init__(self, task, workspace, seed, options):
        module = importlib.import_module(options.get("manager_module", "tool_manager"))
        self.manager = module.ToolManager()
        self.calls = []
        self.expected = task["parameters"]["ground_truth"]
        if isinstance(self.expected, dict):
            self.expected = [self.expected]
        if not self.expected:
            raise ValueError("API-Bank grading requires released API-call labels")
        names = task["parameters"].get("apis", [row["api_name"] for row in self.expected])
        self.initial_observation = "\n".join(self.manager.get_api_description(name) for name in names)

    def action(self, name, arguments):
        if name != "call":
            raise ValueError("API-Bank interface uses call(name, **parameters)")
        api_name = arguments["name"]
        index = len(self.calls)
        self.calls.append((api_name, None))
        try:
            result = self.manager.api_call(api_name, **arguments.get("parameters", {}))
        except Exception as exc:
            return {"api_name": api_name, "exception": type(exc).__name__, "message": str(exc)}
        self.calls[index] = (api_name, result)
        return result

    def grade(self, answer):
        correct = 0
        for index, expected in enumerate(self.expected):
            if index >= len(self.calls) or self.calls[index][0] != expected["api_name"] or self.calls[index][1] is None:
                continue
            api = self.manager.init_tool(expected["api_name"])
            correct += bool(api.check_api_call_correctness(self.calls[index][1], expected["result"]))
        return correct / max(len(self.calls), len(self.expected))

    def close(self):
        pass


class AppWorldBackend:
    def __init__(self, task, workspace, seed, options):
        from appworld import AppWorld
        self.world = AppWorld(task_id=task["task_id"], experiment_name=workspace.name + "-" + secrets.token_hex(6),
                              random_seed=seed, **options.get("environment", {}))
        self.initial_observation = str(self.world.task.instruction)

    def action(self, name, arguments):
        if name != "execute":
            raise ValueError("AppWorld interface uses execute(code)")
        return self.world.execute(arguments["code"])

    def grade(self, answer):
        return float(self.world.evaluate(suppress_errors=False).success)

    def close(self):
        self.world.close()


class ToolBenchBackend:
    def __init__(self, task, workspace, seed, options):
        from toolbench.inference.Downstream_tasks.rapidapi import rapidapi_wrapper
        settings = dict(options.get("environment", {}))
        settings.setdefault("tool_root_dir", options["tool_root_dir"])
        settings.setdefault("toolbench_key", os.environ.get(options.get("toolbench_key_env", "TOOLBENCH_KEY"), ""))
        settings.setdefault("rapidapi_key", os.environ.get(options.get("rapidapi_key_env", "RAPIDAPI_KEY"), ""))
        settings.setdefault("use_rapidapi_key", False)
        settings.setdefault("api_customization", False)
        settings.setdefault("max_observation_length", 4096)
        settings.setdefault("observ_compress_method", "truncate")
        settings.setdefault("retrieved_api_nums", 5)
        parameters = task["parameters"]
        self.env = rapidapi_wrapper(parameters["query_json"], parameters.get("tool_descriptions", []),
                                   None, SimpleNamespace(**settings), process_id=seed)
        from evaluators import load_registered_automatic_evaluator
        self.evaluator = load_registered_automatic_evaluator(
            evaluator_name=options.get("evaluator", "tooleval_gpt-3.5-turbo_default"),
            evaluators_cfg_path=str(Path(options["sdk_path"]) / "toolbench" / "tooleval" / "evaluators"))
        self.repeats = options.get("evaluate_times", 4)
        if not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError("ToolEval repetitions require a positive integer")
        self.rng = random.Random(seed)
        self.query = parameters["query_json"]["query"]
        self.steps = []
        self.initial_observation = json.dumps(self.env.functions)

    def action(self, name, arguments):
        if name != "call":
            raise ValueError("ToolBench interface uses call(name, **parameters)")
        api_name = arguments["name"]
        payload = json.dumps(arguments.get("parameters", {}))
        observation, status = self.env.step(action_name=api_name, action_input=payload)
        self.steps.append({"role": "tool", "name": api_name, "input": payload,
                           "message": str({"name": api_name, "arguments": payload}),
                           "content": str(observation), "next": []})
        return {"observation": observation, "status": status}

    def grade(self, answer):
        from evaluators.registered_cls.rtl import AnswerStatus, AnswerPass
        if not answer or not any(step["name"] == "Finish" for step in self.steps):
            return 0.0
        available = [{"name": row["name"], "description": row.get("description", "")} for row in self.env.functions]
        details = []
        for step in reversed(self.steps):
            details = [{**step, "next": details}]
        candidate = {"method": "CREDENCE", "total_steps": len(self.steps),
                     "final_answer": answer, "answer_details": details}
        query = {"query": self.query, "available_tools": available}
        passed = 0
        for _ in range(self.repeats):
            solved, _ = self.evaluator.check_is_solved(query, candidate, return_reason=True)
            solvable, _ = self.evaluator.check_task_solvable(query, has_been_solved=solved == AnswerStatus.Solved,
                                                            return_reason=True)
            result = self.evaluator.is_passed(query, candidate, answer_status=solved, task_status=solvable)
            passed += result == AnswerPass.Passed or (result == AnswerPass.Unsure and self.rng.randrange(2) == 1)
        if passed * 2 == self.repeats:
            return float(self.rng.randrange(2))
        return float(passed * 2 > self.repeats)

    def close(self):
        pass


BACKENDS = {"scienceworld": ScienceWorldBackend, "webshop": WebShopBackend,
            "api_bank": APIBankBackend, "toolbench": ToolBenchBackend, "appworld": AppWorldBackend}


class NativeSession:
    def __init__(self, benchmark, task, workspace, seed, options):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        sdk_path = options.get("sdk_path")
        if sdk_path:
            sdk_path = str(Path(sdk_path).resolve())
            options = {**options, "sdk_path": sdk_path}
            sys.path.insert(0, sdk_path)
            if benchmark == "toolbench":
                sys.path.insert(0, str(Path(sdk_path) / "toolbench" / "tooleval"))
            os.chdir(sdk_path)
        random.seed(seed)
        self.backend = BACKENDS[benchmark](task, self.workspace, seed, options)
        self.token = secrets.token_hex(24)
        self.requests = queue.Queue()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.headers.get("Authorization") != owner.token:
                    self.send_error(403)
                    return
                try:
                    request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    response = queue.Queue(maxsize=1)
                    owner.requests.put((request, response))
                    payload = json.dumps(response.get(timeout=90)).encode()
                except Exception as exc:
                    payload = json.dumps({"error": str(exc)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        endpoint = f"http://127.0.0.1:{self.server.server_port}"
        client = (
            "import json, urllib.request\n"
            "def request(method, **arguments):\n"
            "    body = json.dumps({'method': method, 'arguments': arguments}).encode()\n"
            f"    query = urllib.request.Request({endpoint!r}, data=body, headers={{'Authorization': {self.token!r}}})\n"
            "    with urllib.request.urlopen(query, timeout=90) as response:\n"
            "        value = json.load(response)\n"
            "    if 'error' in value: raise RuntimeError(value['error'])\n"
            "    return value['result']\n"
            "def step(action): return request('step', action=action)\n"
            "def call(name, **parameters): return request('call', name=name, parameters=parameters)\n"
            "def execute(code): return request('execute', code=code)\n"
        )
        (self.workspace / "benchmark.py").write_text(client, encoding="utf-8")
        self.initial_observation = (str(self.backend.initial_observation)
                                    + "\nNative environment access is available in benchmark.py. "
                                    + "Use python3 to import benchmark and call step(action), call(name, **parameters), or execute(code) for this environment.")

    def step(self, action):
        if action["action"] == "py":
            command, shell = [sys.executable, "-c", action["content"]], False
        elif action["action"] == "shell":
            command, shell = action["content"], True
        else:
            raise ValueError("Native steps require shell or py actions")
        process = subprocess.Popen(command, cwd=self.workspace, shell=shell, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 90
        try:
            while True:
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(command, 90)
                try:
                    output, error = process.communicate(timeout=0.02)
                    return (output + error)[:8000]
                except subprocess.TimeoutExpired:
                    pass
                try:
                    request, response = self.requests.get_nowait()
                except queue.Empty:
                    continue
                try:
                    value = self.backend.action(request["method"], request["arguments"])
                    response.put({"result": value})
                except Exception as exc:
                    response.put({"error": str(exc)})
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def grade(self, answer):
        return normalized_score(self.backend.grade(answer))

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=BACKENDS, required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args()
    options = json.loads(args.options)
    protocol_output = sys.stdout
    session = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    method = request["method"]
                    if method == "reset":
                        if session is not None:
                            session.close()
                        session = NativeSession(args.benchmark, request["task"], request["workspace"], request["seed"], options)
                        response = {"observation": session.initial_observation}
                    elif method == "step":
                        response = {"observation": session.step(request["action"])}
                    elif method == "grade":
                        response = {"score": session.grade(request["answer"])}
                    elif method == "close":
                        if session is not None:
                            session.close()
                            session = None
                        response = {}
                    else:
                        raise ValueError("Unknown benchmark request")
            except Exception as exc:
                response = {"error": str(exc)}
            protocol_output.write(json.dumps(response) + "\n")
            protocol_output.flush()
            if request.get("method") == "close":
                break
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
