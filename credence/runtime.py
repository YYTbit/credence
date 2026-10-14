import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import prompts
from .environments import normalized_score
from .retrieval import greedy_select, keyword_partition, pool_summary
from .store import Store
from .tools import tool_functions, validate_tool


SCHEMAS = {
    "writer": {"type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string"}},
               "required": ["title", "text"], "additionalProperties": False},
    "builder": {"type": "object", "properties": {"name": {"type": "string"}, "code": {"type": "string"}},
                "required": ["name", "code"], "additionalProperties": False},
    "curator": {"type": "object", "properties": {"asset_id": {"type": "string"},
                "action": {"enum": ["keep", "merge", "split", "remove"]},
                "target": {"type": ["string", "null"]}},
                "required": ["asset_id", "action", "target"], "additionalProperties": False},
    "actor": {"type": "object", "properties": {"action": {"enum": ["shell", "py", "final"]},
              "content": {"type": "string"}}, "required": ["action", "content"], "additionalProperties": False},
}


class BudgetExhausted(Exception):
    pass


class EpisodeUnavailable(RuntimeError):
    pass


@dataclass
class TokenBudget:
    limit: int
    used: int = 0

    def reserve(self, amount):
        if self.used + amount > self.limit:
            raise BudgetExhausted("Episode token budget exhausted")

    def consume(self, amount):
        self.reserve(amount)
        self.used += amount


@dataclass
class Episode:
    task: dict
    store: Store
    subtasks: list
    assembled: list
    score: float
    trajectory: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    tokens: int = 0
    seed: int = 0
    completed: bool = False


class Runtime:
    def __init__(self, models, calibrator, config):
        self.models, self.calibrator, self.config = models, calibrator, config
        self.diagnostics = []

    def embeddings(self, store, subtasks, exclude=None, insert=None, grad=False):
        assets = [asset for aid, asset in store.assets.items() if aid != exclude]
        if insert is not None:
            assets.append(insert)
        encoded = self.models.encode(subtasks + [asset.text for asset in assets], grad=grad)
        return encoded[:len(subtasks)], encoded[len(subtasks):], assets

    def summary(self, store, subtasks, exclude=None, insert=None, grad=False):
        tasks, embeddings, assets = self.embeddings(store, subtasks, exclude, insert, grad)
        return pool_summary(tasks, embeddings, assets, self.config.context_budget)

    def prediction(self, summary):
        with torch.no_grad():
            return float(self.models.outcome(summary).item())

    def selection(self, store, subtasks, exclude=None, insert=None):
        tasks, embeddings, assets = self.embeddings(store, subtasks, exclude, insert)
        indices, _ = greedy_select(tasks, embeddings, self.config.context_budget)
        return [assets[i].asset_id for i in indices]

    def complete(self, head, prompt, limit, budget, seed, schema=None, sample=True):
        prompt_tokens = len(self.models.prompt_ids(prompt))
        budget.reserve(prompt_tokens + limit)
        result = self.models.generate(head, prompt, limit, seed, schema, sample)
        budget.consume(len(result.prompt_ids) + len(result.output_ids))
        return result

    def run_episode(self, task, store, environment, workspace, seed):
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        start = store.clone()
        budget = TokenBudget(self.config.token_budget)
        try:
            session = environment.start(task, workspace, seed)
        except (TimeoutError, RuntimeError, subprocess.TimeoutExpired) as exc:
            raise EpisodeUnavailable(str(exc)) from exc
        trajectory, evidence, assembled = [], [], []
        answer, completed = "", False
        subtasks = [task["text"]]
        try:
            plan = self.complete("agent", prompts.PLANNER.format(task=task["text"]),
                                 self.config.planner_tokens, budget, seed, sample=False)
            subtasks = [line.strip() for line in plan.text.splitlines() if line.strip()][:self.config.max_subtasks]
            if not subtasks:
                subtasks = [task["text"]]
            task_embeddings, asset_embeddings, assets = self.embeddings(start, subtasks)
            chosen, _ = greedy_select(task_embeddings, asset_embeddings, self.config.context_budget)
            assembled = [assets[i].asset_id for i in chosen]
            context = "\n\n".join(
                f"{assets[i].kind} {assets[i].asset_id}"
                + (f" module {assets[i].asset_id.replace('-', '_')}.py" if assets[i].kind == "tool" else "")
                + f"\n{assets[i].text}" for i in chosen)
            for i in chosen:
                if assets[i].kind == "tool":
                    (workspace / f"{assets[i].asset_id.replace('-', '_')}.py").write_text(assets[i].text, encoding="utf-8")
            previous = session.initial_observation
            for step in range(self.config.max_steps):
                current = subtasks[min(step, len(subtasks) - 1)]
                prompt = prompts.ACTOR.format(task=task["text"], subtasks="\n".join(subtasks),
                                              subtask=current, context=context, prev=previous)
                output = self.complete("agent", prompt, self.config.actor_tokens, budget,
                                       seed + step + 1, SCHEMAS["actor"], sample=False)
                try:
                    action = json.loads(output.text)
                    if action.get("action") not in ("shell", "py", "final") or not isinstance(action.get("content"), str):
                        raise ValueError("Invalid actor action")
                except (ValueError, TypeError, AttributeError):
                    previous += "\nAction format requires a JSON object with action and content."
                    continue
                trajectory.append(action)
                if action["action"] == "final":
                    answer, completed = action["content"], True
                    break
                try:
                    observation = session.step(action)
                except (TimeoutError, subprocess.TimeoutExpired):
                    observation = "The execution exceeded its time limit."
                except RuntimeError as exc:
                    raise EpisodeUnavailable(str(exc)) from exc
                evidence.append(observation)
                previous += "\n" + json.dumps(action) + "\n" + str(observation)
        except BudgetExhausted:
            pass
        finally:
            try:
                try:
                    score = normalized_score(session.grade(answer))
                except (TimeoutError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                    raise EpisodeUnavailable(str(exc)) from exc
            finally:
                session.close()
        return Episode(task, start, subtasks, list(dict.fromkeys(assembled)), score,
                       trajectory, evidence, budget.used, seed, completed)

    def credit(self, episode, store):
        values = {}
        for aid in episode.assembled:
            summary = self.summary(episode.store, episode.subtasks, exclude=aid)
            value = episode.score - self.prediction(summary)
            if aid in store.assets:
                store.observe(aid, value)
                values[aid] = value
            self.diagnostics.append({"event": "credit", "asset_id": aid,
                                     "value": value, "supported": self.calibrator.supported(summary)})
        if store.episode % self.config.restoration_horizon == 0:
            for aid in store.assets.keys() - set(episode.assembled):
                value = episode.score - self.prediction(self.summary(episode.store, episode.subtasks, exclude=aid))
                self.diagnostics.append({"event": "idle_query", "asset_id": aid, "value": value})
        return values

    def restore(self, subtasks, store):
        base = self.summary(store, subtasks)
        baseline = self.prediction(base)
        current = self.selection(store, subtasks)
        for aid in list(store.buffer):
            asset = store.buffer[aid].asset
            summary = self.summary(store, subtasks, insert=asset)
            changed = self.selection(store, subtasks, insert=asset) != current
            supported = self.calibrator.supported(summary)
            gain = self.prediction(summary) - baseline
            restored = store.restore(aid, gain, supported, changed)
            self.diagnostics.append({"event": "restoration", "asset_id": aid, "new_version": restored,
                                     "gain": gain, "supported": supported, "context_changed": changed})
            if restored:
                base = self.summary(store, subtasks)
                baseline = self.prediction(base)
                current = self.selection(store, subtasks)

    def partition(self, keywords):
        return keyword_partition(self.models.encode(keywords))

    def split_memory(self, asset, groups):
        units = [unit.strip() for unit in re.split(r"(?<=[.!?;])\s+|\n+", asset.text) if unit.strip()]
        if len(units) < 2:
            raise ValueError("Memory splitting requires separable content units")
        keywords = self.models.encode(asset.keywords)
        centers = torch.stack([keywords[group].mean(dim=0) for group in groups])
        centers = torch.nn.functional.normalize(centers, dim=-1)
        labels = (self.models.encode(units) @ centers.T).argmax(dim=1)
        return [" ".join(unit for unit, label in zip(units, labels) if int(label) == k) for k in range(2)]

    def prompt(self, head, episode, store, asset_id=None):
        if head == "writer":
            return prompts.WRITER.format(task=episode.task["text"][:1200],
                                          subtasks="\n".join(episode.subtasks), score=f"{episode.score:.4f}",
                                          trajectory=json.dumps(episode.trajectory, ensure_ascii=False))
        if head == "builder":
            tools = []
            for asset in store.assets.values():
                if asset.kind == "tool":
                    tools.extend(f"{node.name} {ast.unparse(node.args)}" for node in tool_functions(asset.text))
            return prompts.BUILDER.format(task=episode.task["text"][:1200], score=f"{episode.score:.4f}",
                                           tools="\n".join(tools), trajectory=json.dumps(episode.trajectory, ensure_ascii=False),
                                           evidence=json.dumps(episode.evidence, ensure_ascii=False))
        card = store.assets[asset_id].card()
        card["compatible_targets"] = [aid for aid, asset in store.assets.items()
                                      if aid != asset_id and asset.kind == store.assets[asset_id].kind]
        return prompts.CURATOR.format(card=self.models.truncate(json.dumps(card), self.config.card_tokens))

    def apply(self, head, sample, episode, store, asset_id=None):
        proposal = json.loads(sample.text)
        if not isinstance(proposal, dict):
            raise ValueError("Policy output requires a JSON object")
        if head == "writer":
            if set(proposal) != {"title", "text"} or not all(isinstance(value, str) for value in proposal.values()):
                raise ValueError("Invalid memory format")
            if not proposal["title"].strip() or len(proposal["title"].split()) > 8 or not proposal["text"].strip() or len(proposal["text"].split()) > 120:
                raise ValueError("Memory format limits exceeded")
            return store.add("memory", proposal["text"], proposal["title"], episode.subtasks,
                             self.models.token_count(proposal["text"]), head)
        if head == "builder":
            if set(proposal) != {"name", "code"} or not all(isinstance(value, str) for value in proposal.values()):
                raise ValueError("Invalid tool format")
            functions = validate_tool(proposal["code"], episode.task.get("tool_tests", []))
            existing = [node.name for asset in store.assets.values() if asset.kind == "tool" for node in tool_functions(asset.text)]
            if len(functions) != 1 or proposal["name"] != functions[0] or proposal["name"] in existing:
                raise ValueError("Tool proposal requires one distinct named function")
            return store.add("tool", proposal["code"], proposal["name"], episode.subtasks,
                             self.models.token_count(proposal["code"]), head)
        if set(proposal) != {"asset_id", "action", "target"} or proposal["asset_id"] != asset_id:
            raise ValueError("Curator output requires the decision asset")
        if proposal["action"] != "merge" and proposal["target"] is not None:
            raise ValueError("Curator target is null for keep, split, and remove")
        if proposal["action"] != "keep" and not self.calibrator.eligible(store.assets[asset_id].credit):
            raise ValueError("Curation requires credit beyond the calibrated margin")
        store.apply(proposal, self.models.token_count, self.partition, self.config.protect_recent, self.split_memory)
        return None

    def evolve(self, episode, store, seed):
        budget = TokenBudget(self.config.token_budget, episode.tokens)
        for offset, head in enumerate(("writer", "builder")):
            prompt = self.prompt(head, episode, store)
            try:
                sample = self.complete(head, prompt, self.config.policy_tokens, budget,
                                       seed + offset, SCHEMAS[head])
                trial = store.clone()
                self.apply(head, sample, episode, trial)
                store = trial
            except (BudgetExhausted, ValueError, TypeError, KeyError, subprocess.TimeoutExpired):
                continue
        for offset, aid in enumerate(episode.assembled):
            if aid not in store.assets or not self.calibrator.eligible(store.assets[aid].credit):
                continue
            try:
                sample = self.complete("curator", self.prompt("curator", episode, store, aid),
                                       self.config.policy_tokens, budget, seed + 2 + offset, SCHEMAS["curator"])
                trial = store.clone()
                self.apply("curator", sample, episode, trial, aid)
                store = trial
            except (BudgetExhausted, ValueError, TypeError, KeyError, subprocess.TimeoutExpired):
                continue
        self.restore(episode.subtasks, store)
        episode.tokens = budget.used
        return store
