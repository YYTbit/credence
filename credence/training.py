import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .credit import Calibrator
from .curriculum import Curriculum
from .checkpoint import (compatible_checkpoint, digest, restore_rng,
                         rng_state, save_bundle)
from .groups import Candidate, CandidateGroup, GroupBuffer, group_advantages, grpo_loss
from .retrieval import coverage
from .runtime import BudgetExhausted, EpisodeUnavailable, Runtime, SCHEMAS, TokenBudget, policy_schema
from .store import Store


TRAIN_BENCHMARKS = ("scienceworld", "webshop", "api_bank", "toolbench")


@dataclass
class Stream:
    stream_id: int
    benchmark: str
    store: Store
    order: list
    position: int = 0


def initial_store(models, config, seed):
    rng = random.Random(seed)
    store = Store(config.store_capacity, config.restoration_horizon)
    memory = f"Verify intermediate results and preserve reusable procedures. Stream marker {rng.randrange(100000)}."
    store.add("memory", memory, "Verification procedure", ["verify intermediate results"],
              models.token_count(memory), "seed")
    code = f"def count_items_{rng.randrange(100000)}(items):\n    return len(items)"
    store.add("tool", code, "Item counter", ["count items"], models.token_count(code), "seed")
    return store


def make_streams(tasks, models, config, length=None):
    streams = []
    for pair, benchmark in enumerate(TRAIN_BENCHMARKS):
        available = [task for task in tasks if task["benchmark"] == benchmark and task["split"] == "train"]
        if not available:
            raise ValueError(f"Training tasks missing for {benchmark}")
        stream_length = config.training_epochs * len(available) if length is None else length
        order = []
        rng = random.Random(config.order_seeds[pair])
        while len(order) < stream_length:
            chunk = list(available)
            rng.shuffle(chunk)
            order.extend(chunk)
        for stream in range(2):
            sid = 2 * pair + stream
            streams.append(Stream(sid, benchmark, initial_store(models, config, config.store_seeds[sid]), order[:stream_length]))
    return streams


class Trainer:
    def __init__(self, models, environments, config, output):
        self.models, self.environments, self.config = models, environments, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.calibrator = Calibrator(config.calibration_quantile, config.reuse_margin)
        self.runtime = Runtime(models, self.calibrator, config)
        self.optimizer = torch.optim.AdamW(models.parameters(), lr=config.learning_rate(0))
        self.buffer = GroupBuffer(config.max_policy_lag)
        self.head_updates = dict.fromkeys(models.heads, 0)
        self.step = 0
        self.serial = 0
        self.fitting = []
        self.calibration = []
        self.streams = []
        self.run_identity = None
        self.curriculum = Curriculum(config)
        self.pending_records = {"warm": [], "joint": []}

    def create_group(self, head, episode, store, stream_id, seed, budget, asset_id=None):
        prompt = self.runtime.prompt(head, episode, store, asset_id)
        required = self.config.group_size * (len(self.models.prompt_ids(prompt)) + self.config.policy_tokens)
        budget.reserve(required)
        self.serial += 1
        group_id = f"group-{self.serial:08d}"
        candidates = []
        for index in range(self.config.group_size):
            sample = self.runtime.complete(head, prompt, self.config.policy_tokens, budget,
                                           seed + index, policy_schema(head, store, asset_id))
            trial = store.clone()
            valid, created = True, None
            payload_tokens = self.models.token_count(sample.text)
            try:
                proposal = json.loads(sample.text)
                payload = proposal.get("text" if head == "writer" else "code", sample.text)
                payload_tokens = self.models.token_count(payload) if isinstance(payload, str) else payload_tokens
                created = self.runtime.apply(head, sample, episode, trial, asset_id)
            except (ValueError, TypeError, KeyError, AttributeError, TimeoutError, subprocess.TimeoutExpired):
                trial, valid = store.clone(), False
            candidates.append(Candidate(sample, trial, created, payload_tokens, valid))
        group = CandidateGroup(group_id, head, stream_id, self.head_updates[head], store.episode,
                               candidates, store.clone() if head == "curator" else None)
        self.buffer.add(group)
        selected = random.Random(seed).randrange(len(candidates))
        return candidates[selected].store.clone()

    def decisions(self, episode, store, stream_id, seed):
        budget = TokenBudget(self.config.token_budget, episode.tokens)
        for offset, head in enumerate(("writer", "builder")):
            try:
                store = self.create_group(head, episode, store, stream_id, seed + 10000 * offset, budget)
            except BudgetExhausted:
                continue
        for offset, aid in enumerate(episode.assembled):
            if aid not in store.assets or not self.calibrator.eligible(store.assets[aid].credit):
                continue
            try:
                store = self.create_group("curator", episode, store, stream_id, seed + 20000 + offset * 100,
                                          budget, aid)
            except BudgetExhausted:
                continue
        self.runtime.restore(episode.subtasks, store, budget)
        episode.tokens = budget.used
        return store

    def advance_groups(self, stream, task, seed):
        for group in list(self.buffer.pending):
            if group.stream != stream.stream_id:
                continue
            try:
                self.advance_group(group, stream, task, seed)
            except EpisodeUnavailable as exc:
                self.buffer.discard_unavailable(group)
                self.log({"event": "continuation_unavailable", "group": group.group_id,
                          "task_id": task["task_id"], "reason": str(exc)})

    def advance_group(self, group, stream, task, seed):
        if self.head_updates[group.head] - group.generated_update > self.config.max_policy_lag:
            return
        reference = group.reference_store
        if reference is not None:
            reference.advance()
            episode = self.runtime.run_episode(task, reference, self.environments[stream.benchmark],
                                               self.output / "workspaces" / group.group_id / "reference" / str(reference.episode), seed)
            group.reference_scores.append(episode.score)
            self.runtime.credit(episode, reference)
            group.reference_store = self.runtime.evolve(episode, reference, seed + 70000)
            self.log({"event": "continuation", "group": group.group_id, "branch": "reference",
                      "benchmark": stream.benchmark, "task_id": task["task_id"],
                      "seed": seed, "score": episode.score, "tokens": episode.tokens})
        for index, candidate in enumerate(group.candidates):
            if candidate.reward is not None or (not candidate.valid and group.head != "curator"):
                continue
            candidate.store.advance()
            episode = self.runtime.run_episode(task, candidate.store, self.environments[stream.benchmark],
                                               self.output / "workspaces" / group.group_id / str(index) / str(candidate.store.episode), seed)
            candidate.scores.append(episode.score)
            self.runtime.credit(episode, candidate.store)
            candidate.store = self.runtime.evolve(episode, candidate.store, seed + 70000)
            self.log({"event": "continuation", "group": group.group_id, "branch": index,
                      "benchmark": stream.benchmark, "task_id": task["task_id"],
                      "seed": seed, "score": episode.score, "tokens": episode.tokens})
        group.mature(max(candidate.store.episode for candidate in group.candidates),
                     self.config.credit_horizon, self.config.token_cost)

    def calibrate_episode(self, episode, environment, workspace):
        cases = [(episode.store.clone(), episode.score)]
        for aid in episode.assembled:
            removed = episode.store.clone()
            removed.retire(aid, buffered=False)
            paired = self.runtime.run_episode(episode.task, removed, environment, workspace / aid, episode.seed)
            cases.append((removed, paired.score))
            self.log({"event": "calibration_removal", "benchmark": episode.task["benchmark"],
                      "task_id": episode.task["task_id"], "asset_id": aid,
                      "seed": episode.seed, "score": paired.score, "tokens": paired.tokens})
        self.calibration.append((episode.subtasks, cases))

    def refresh_calibration(self):
        observed, removals = [], []
        for subtasks, cases in self.calibration:
            for index, (store, score) in enumerate(cases):
                residual = abs(self.runtime.prediction(self.runtime.summary(store, subtasks)) - score)
                (observed if index == 0 else removals).append(residual)
        summaries = [self.runtime.summary(record.store, record.subtasks) for record in self.fitting]
        self.calibrator.fit(observed, removals, summaries)

    def auxiliary_losses(self, records, warmup):
        predicted, regression, representations = [], [], []
        for record in records:
            summary = self.runtime.summary(record.store, record.subtasks)
            predicted.append((self.models.outcome(summary.detach()) - record.score).square())
            if warmup:
                continue
            selected = [record.store.assets[aid] for aid in record.assembled]
            encoded = self.models.encode(record.subtasks + [asset.text for asset in selected], grad=True)
            tasks, assets = encoded[:len(record.subtasks)], encoded[len(record.subtasks):]
            regression.append((coverage(tasks, assets) - record.score).square())
            representations.append((record.task["benchmark"], record.task["task_id"], F.normalize(tasks.mean(dim=0), dim=0), assets))
        zero = next(self.models.outcome.parameters()).sum() * 0
        contrastive = []
        for benchmark, task_id, query, assets in representations:
            negatives = [other for b, t, _, other in representations if (b, t) != (benchmark, task_id) and len(other)]
            if not len(assets) or not negatives:
                continue
            positive = query @ assets.T / self.config.contrastive_temperature
            negative = query @ torch.cat(negatives).T / self.config.contrastive_temperature
            contrastive.append((torch.logsumexp(torch.cat((positive, negative)), dim=0) - positive).mean())
        model_loss = torch.stack(predicted).mean()
        regression_loss = torch.stack(regression).mean() if regression else zero
        contrastive_loss = torch.stack(contrastive).mean() if contrastive else zero
        return regression_loss, contrastive_loss, model_loss

    def update(self, records, warmup=False):
        if not records:
            return
        # Policy gradients use candidate groups only after their delayed rewards mature.
        ready = [] if warmup else self.buffer.release(self.head_updates, self.config.credit_horizon, self.config.token_cost)
        self.optimizer.zero_grad(set_to_none=True)
        policy_value = 0.0
        active = set()
        for group in ready:
            advantages = group_advantages([candidate.reward for candidate in group.candidates], self.config.advantage_epsilon)
            terms = []
            for index, candidate in enumerate(group.candidates):
                reference = self.models.logprobs(group.head + "_ref", candidate.sample, grad=False)
                current = self.models.logprobs(group.head, candidate.sample)
                behavior = candidate.sample.behavior_logprobs.to(current.device)
                terms.append(grpo_loss(current, behavior, reference, advantages[index].to(current.device),
                                       self.config.clip_ratio, self.config.kl_weight))
            loss = torch.stack(terms).mean() / max(1, len(ready))
            self.models.select(group.head)
            loss.backward()
            policy_value += float(loss.detach())
            active.add(group.head)
        regression, contrastive, model = self.auxiliary_losses(records, warmup)
        auxiliary = self.config.predictor_weight * model
        if not warmup:
            auxiliary = auxiliary + self.config.encoder_weight * (regression + self.config.contrastive_weight * contrastive)
            self.models.select("encoder")
        auxiliary.backward()
        encoder_parameters = [parameter for name, parameter in self.models.model.named_parameters() if ".encoder." in name]
        torch.nn.utils.clip_grad_norm_(encoder_parameters + list(self.models.projection.parameters()), self.config.gradient_norm)
        torch.nn.utils.clip_grad_norm_(self.models.outcome.parameters(), self.config.gradient_norm)
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate(self.step)
        self.optimizer.step()
        if not warmup:
            self.models.invalidate_embeddings()
        self.step += 1
        for head in active:
            self.head_updates[head] += 1
        self.fitting.extend(records)
        self.refresh_calibration()
        self.log({"event": "update", "step": self.step, "warmup": warmup,
                  "policy_loss": policy_value, "coverage_loss": float(regression.detach()),
                  "contrastive_loss": float(contrastive.detach()), "predictor_loss": float(model.detach()),
                  "eligible_groups": len(ready), **self.buffer.statistics()})

    def train(self, tasks, episodes_per_stream=None, checkpoint_every=10, input_identity=None):
        if episodes_per_stream is None and self.run_identity is not None:
            episodes_per_stream = self.run_identity["episodes_per_stream"]
        identity = {"tasks": digest(tasks), "episodes_per_stream": episodes_per_stream,
                    "inputs": input_identity or {}}
        if self.run_identity is not None and identity != self.run_identity:
            raise ValueError("Training resume requires the original tasks and environment inputs")
        self.run_identity = identity
        if not self.streams:
            self.streams = make_streams(tasks, self.models, self.config, episodes_per_stream)
        last_saved_step = self.step
        while any(stream.position < len(stream.order) for stream in self.streams):
            warm, current = [], []
            before_stage = (self.curriculum.phase, self.curriculum.stage)
            active = self.curriculum.active(self.streams)
            if before_stage != (self.curriculum.phase, self.curriculum.stage):
                self.log({"event": "curriculum", "phase": self.curriculum.phase,
                          "stage": self.curriculum.stage, "order": self.curriculum.order})
            for stream in active:
                if stream.position >= len(stream.order):
                    continue
                task = stream.order[stream.position]
                seed = self.config.seed + (stream.stream_id // 2) * 100000 + stream.position
                calibration = (stream.position + 1) % self.config.calibration_stride == 0
                warmup = stream.position < self.config.warmup_episodes
                if not calibration:
                    stream.store.advance()
                    if not warmup:
                        self.advance_groups(stream, task, seed)
                workspace = self.output / "workspaces" / f"stream-{stream.stream_id}" / str(stream.position)
                try:
                    record = self.runtime.run_episode(task, stream.store, self.environments[stream.benchmark], workspace, seed)
                    if calibration:
                        self.calibrate_episode(record, self.environments[stream.benchmark], workspace / "removals")
                except EpisodeUnavailable as exc:
                    self.log({"event": "episode_unavailable", "stream": stream.stream_id,
                              "position": stream.position, "task_id": task["task_id"], "reason": str(exc)})
                    stream.position += 1
                    continue
                if not calibration:
                    self.runtime.credit(record, stream.store)
                    if warmup:
                        stream.store = self.runtime.evolve(record, stream.store, seed + 30000)
                    else:
                        stream.store = self.decisions(record, stream.store, stream.stream_id, seed + 30000)
                    (warm if warmup else current).append(record)
                self.log({"event": "episode", "stream": stream.stream_id, "position": stream.position,
                          "benchmark": stream.benchmark, "task_id": task["task_id"], "calibration": calibration,
                          "score": record.score, "tokens": record.tokens, "assembled": record.assembled,
                          "store_size": len(stream.store.assets)})
                stream.position += 1
            self.pending_records["warm"].extend(warm)
            self.pending_records["joint"].extend(current)
            phase_complete = (all(stream.position >= len(stream.order) for stream in active)
                              or (self.curriculum.phase == "warmup" and all(
                                  stream.position >= min(self.config.warmup_episodes, len(stream.order))
                                  for stream in self.streams)))
            observed_stage = self.curriculum.stage
            self.consume_records(force=phase_complete)
            if self.curriculum.stage != observed_stage:
                self.log({"event": "curriculum", "phase": self.curriculum.phase,
                          "stage": self.curriculum.stage, "order": self.curriculum.order,
                          "reason": "score_plateau"})
            if not warm and not current:
                self.refresh_calibration()
            self.flush_diagnostics()
            if self.step and self.step % checkpoint_every == 0 and self.step != last_saved_step:
                self.save(self.output / "checkpoint")
                last_saved_step = self.step
        self.consume_records(force=True)
        self.save(self.output / "checkpoint")
        return self.buffer.statistics()

    def consume_records(self, force=False):
        for name in ("warm", "joint"):
            records = self.pending_records[name]
            if records and (len(records) >= len(self.streams) or force):
                self.curriculum.observe(records)
                self.update(records, warmup=name == "warm")
                self.pending_records[name] = []

    def log(self, values):
        with (self.output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, allow_nan=False) + "\n")

    def flush_diagnostics(self):
        with (self.output / "diagnostics.jsonl").open("a", encoding="utf-8") as handle:
            for record in self.runtime.diagnostics:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.runtime.diagnostics.clear()

    def save(self, path):
        self.flush_diagnostics()
        state = {"optimizer": self.optimizer.state_dict(), "step": self.step,
                 "head_updates": self.head_updates, "serial": self.serial, "buffer": self.buffer,
                 "streams": self.streams, "fitting": self.fitting, "calibration": self.calibration,
                 "calibrator": self.calibrator.state_dict(), "rng": rng_state(),
                 "curriculum": self.curriculum.state_dict(),
                 "pending_records": self.pending_records,
                 "run_identity": self.run_identity, "log_positions": {}}
        for name in ("training.jsonl", "diagnostics.jsonl"):
            target = self.output / name
            if target.is_file():
                import hashlib
                body = target.read_bytes()
                state["log_positions"][name] = (len(body), hashlib.sha256(body).hexdigest())
        evaluation = {"calibrator": self.calibrator.state_dict(),
                      "stores": {stream.benchmark: stream.store.state_dict() for stream in reversed(self.streams)},
                      "source_stores": [{"stream": stream.stream_id, "benchmark": stream.benchmark,
                                         "store": stream.store.state_dict()} for stream in self.streams]}
        save_bundle(self.models, path, state, evaluation)

    def restore(self, path):
        path = compatible_checkpoint(self.config, path, resume=True)
        state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        if "rng" not in state:
            raise ValueError("Training resume requires a checkpoint with RNG state")
        if self.config.curriculum and "curriculum" not in state:
            raise ValueError("Training resume requires the saved curriculum state")
        self.models.restore(path)
        random_state = state.pop("rng")
        for name, (offset, checksum) in state.pop("log_positions", {}).items():
            target = self.output / name
            if target.is_file():
                import hashlib
                with target.open("r+b") as handle:
                    if hashlib.sha256(handle.read(offset)).hexdigest() != checksum:
                        raise ValueError("Existing training logs differ from the resumed checkpoint")
                    handle.truncate(offset)
        self.optimizer.load_state_dict(state.pop("optimizer"))
        self.calibrator.load_state_dict(state.pop("calibrator"))
        if "curriculum" in state:
            self.curriculum.load_state_dict(state.pop("curriculum"))
        for name, value in state.items():
            setattr(self, name, value)
        restore_rng(random_state)


def supervised_warm_start(models, records, output, epochs=1, learning_rate=5e-5,
                          resume=None, checkpoint_every=100):
    samples = []
    for row in records:
        if row.get("head") not in models.heads or not isinstance(row.get("prompt"), str) or not row["prompt"]:
            raise ValueError("Warm-start examples require a head and prompt")
        response = row.get("response")
        if isinstance(response, dict):
            response = json.dumps(response)
        if not isinstance(response, str) or not response:
            raise ValueError("Warm-start examples require a JSON response")
        parsed = json.loads(response)
        schema = SCHEMAS[row["head"]]
        if not isinstance(parsed, dict) or set(parsed) != set(schema["required"]):
            raise ValueError("Warm-start response differs from its head schema")
        from .groups import Sample
        ids = models.response_ids(row["prompt"], response)
        samples.append((row["head"], Sample(row["prompt"], models.prompt_ids(row["prompt"]), ids, response, torch.zeros(len(ids)))))
    if not samples or set(head for head, _ in samples) != set(models.heads):
        raise ValueError("Warm-start input requires examples for all three heads")
    optimizer = torch.optim.AdamW(models.parameters(), lr=learning_rate)
    identity = {"records": digest(records), "epochs": epochs, "learning_rate": learning_rate}
    cursor = 0
    if resume is not None:
        checkpoint = compatible_checkpoint(models.config, resume, resume=True)
        state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
        if state.get("stage") != "sft" or state["identity"] != identity:
            raise ValueError("SFT resume requires the same examples and training settings")
        models.restore(checkpoint)
        optimizer.load_state_dict(state["optimizer"])
        cursor = state["cursor"]
        restore_rng(state["rng"])
    def save(cursor, complete=False):
        save_bundle(models, output, {"stage": "sft", "optimizer": optimizer.state_dict(),
                                    "identity": identity, "cursor": cursor, "rng": rng_state(),
                                    "complete": complete})
    for epoch in range(epochs):
        order = list(samples)
        random.Random(models.config.seed + epoch).shuffle(order)
        for index, (head, sample) in enumerate(order):
            position = epoch * len(samples) + index
            if position < cursor:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = -models.logprobs(head, sample).mean()
            loss.backward()
            optimizer.step()
            cursor = position + 1
            if cursor % checkpoint_every == 0:
                save(cursor)
    models.refresh_references()
    models.invalidate_embeddings()
    save(cursor, complete=True)
