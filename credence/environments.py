import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


BENCHMARKS = ("scienceworld", "webshop", "api_bank", "toolbench", "appworld")


def load_tasks(path):
    tasks = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            if task.get("benchmark") not in BENCHMARKS or not isinstance(task.get("text"), str) or not task["text"].strip():
                raise ValueError("Tasks require a benchmark and nonempty text")
            if not isinstance(task.get("task_id"), str) or not task["task_id"]:
                raise ValueError("Tasks require a nonempty task identifier")
            if task.get("split") not in ("train", "test", "test_normal"):
                raise ValueError("Tasks require an explicit released split")
            key = (task["benchmark"], task["split"], task["task_id"])
            if key in seen:
                raise ValueError("Duplicate task identifier within a split")
            seen.add(key)
            tasks.append(task)
    if not tasks:
        raise ValueError("Task input is empty")
    return tasks


def load_environments(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    environments = {}
    for benchmark, spec in manifest.items():
        if benchmark not in BENCHMARKS:
            raise ValueError("Unknown environment benchmark")
        module, name = spec["factory"].split(":", 1)
        environments[benchmark] = getattr(importlib.import_module(module), name)(**spec.get("kwargs", {}))
    return environments


def normalized_score(value):
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError("Official scores require normalization to [0, 1]")
    return value


class WorkspaceSession:
    def __init__(self, task, workspace, seed):
        self.task, self.workspace = task, Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        for name, text in task.get("files", {}).items():
            path = (self.workspace / name).resolve()
            if self.workspace not in path.parents:
                raise ValueError("Workspace files require relative paths")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.initial_observation = ""

    def step(self, action):
        if action["action"] == "py":
            command = [sys.executable, "-c", action["content"]]
            shell = False
        elif action["action"] == "shell":
            command, shell = action["content"], True
        else:
            raise ValueError("Workspace steps require shell or py actions")
        result = subprocess.run(command, cwd=self.workspace, shell=shell, capture_output=True,
                                text=True, timeout=30, env={"PATH": os.defpath})
        return (result.stdout + result.stderr)[:8000]

    def grade(self, answer):
        if "expected" not in self.task:
            raise ValueError("Workspace fixtures require an expected answer")
        return float(answer.strip() == str(self.task["expected"]).strip())

    def close(self):
        pass


class WorkspaceEnvironment:
    def start(self, task, workspace, seed):
        return WorkspaceSession(task, workspace, seed)


class CommandSession:
    def __init__(self, command, task, workspace, seed, timeout):
        self.timeout = timeout
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            self.initial_observation = self.request("reset", task=task, workspace=str(workspace), seed=seed).get("observation", "")
        except Exception:
            self.close()
            raise

    def request(self, method, **parameters):
        import selectors
        self.process.stdin.write(json.dumps({"method": method, **parameters}) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                self.close()
                raise TimeoutError("Benchmark interface timed out")
        reply = self.process.stdout.readline()
        if not reply:
            raise RuntimeError("Benchmark interface terminated")
        response = json.loads(reply)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def step(self, action):
        return self.request("step", action=action)["observation"]

    def grade(self, answer):
        return normalized_score(self.request("grade", answer=answer)["score"])

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"method": "close"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class CommandEnvironment:
    def __init__(self, command, timeout=60):
        if not isinstance(command, list) or not command:
            raise ValueError("Benchmark command requires a nonempty argument list")
        self.command, self.timeout = command, timeout

    def start(self, task, workspace, seed):
        return CommandSession(self.command, task, workspace, seed, self.timeout)
