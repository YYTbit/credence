import importlib
import json
import math
import os
import subprocess
import sys
import selectors
import time
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
            if task["benchmark"] == "toolbench" and task["split"] == "train":
                from .rewards import reference_calls
                reference_calls(task.get("parameters", {}))
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
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Environment configuration requires named factories")
    environments = {}
    for benchmark, spec in manifest.items():
        if benchmark not in BENCHMARKS:
            raise ValueError("Unknown environment benchmark")
        if not isinstance(spec, dict) or not isinstance(spec.get("factory"), str) or ":" not in spec["factory"]:
            raise ValueError("Environment factories use module:function notation")
        module, name = spec["factory"].split(":", 1)
        environments[benchmark] = getattr(importlib.import_module(module), name)(**spec.get("kwargs", {}))
        if not callable(getattr(environments[benchmark], "start", None)):
            raise ValueError("Environment factories require a start(task, workspace, seed) method")
    return environments


def normalized_score(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Official scores require a normalized numeric value") from exc
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError("Official scores require normalization to [0, 1]")
    return value


class CommandSession:
    def __init__(self, command, task, workspace, seed, timeout):
        self.timeout = timeout
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=None, bufsize=0, env=environment)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self._pending = bytearray()
        self._failed = False
        try:
            reply = self.request("reset", task=task, workspace=str(Path(workspace).resolve()), seed=seed)
            self.initial_observation = reply["observation"]
        except Exception:
            self.close()
            raise

    def request(self, method, **values):
        if self._failed:
            raise RuntimeError("The benchmark interface is unavailable")
        try:
            deadline = time.monotonic() + self.timeout
            outgoing = memoryview((json.dumps({"method": method, **values}, allow_nan=False) + "\n").encode("utf-8"))
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                # Readiness covers available bytes, so retain partial lines under the same deadline.
                while outgoing or b"\n" not in self._pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("The benchmark interface timed out")
                    events = selector.select(remaining)
                    if not events:
                        raise TimeoutError("The benchmark interface timed out")
                    for key, _ in events:
                        try:
                            if key.fileobj is self.process.stdin:
                                written = os.write(key.fd, outgoing[:65536])
                                outgoing = outgoing[written:]
                                if not outgoing:
                                    selector.unregister(self.process.stdin)
                            else:
                                content = os.read(key.fd, 65536)
                                if not content:
                                    raise RuntimeError("The benchmark interface terminated before a complete reply")
                                self._pending.extend(content)
                        except BlockingIOError:
                            continue
            line, _, remaining = self._pending.partition(b"\n")
            self._pending = bytearray(remaining)
            try:
                reply = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError("The benchmark interface returned invalid JSON") from error
            if not isinstance(reply, dict):
                raise RuntimeError("The benchmark interface requires a JSON object")
        except TimeoutError:
            self._failed = True
            raise
        except OSError as exc:
            self._failed = True
            raise RuntimeError("The benchmark interface connection failed") from exc
        except BaseException:
            self._failed = True
            raise
        if "error" in reply:
            raise RuntimeError(reply["error"])
        required = "score" if method == "grade" else "observation" if method in ("reset", "step") else None
        if required is not None and required not in reply:
            self._failed = True
            raise RuntimeError(f"The benchmark response requires {required}")
        return reply

    def step(self, action):
        result = self.request("step", action=action)
        return result["observation"]

    def grade(self, answer):
        return normalized_score(self.request("grade", answer=answer)["score"])

    def close(self):
        self._failed = True
        if self.process.poll() is None:
            # EOF lets an external worker release its native resources.
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class CommandEnvironment:
    def __init__(self, command, timeout=120):
        if not isinstance(command, list) or not command or any(not isinstance(value, str) for value in command):
            raise ValueError("An environment command requires an argument list")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("An environment timeout requires a positive finite value")
        self.command, self.timeout = command, timeout

    def start(self, task, workspace, seed):
        return CommandSession(self.command, task, workspace, seed, self.timeout)
