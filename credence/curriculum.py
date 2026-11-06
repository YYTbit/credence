from types import SimpleNamespace

from .environments import BENCHMARKS


class Curriculum:
    def __init__(self, config):
        from transformers import EarlyStoppingCallback
        self.config = config
        self.monitor = EarlyStoppingCallback(config.curriculum_patience, config.curriculum_threshold)
        self.phase = "warmup" if config.curriculum else "joint"
        self.density = {name: [0, 0] for name in BENCHMARKS[:-1]}
        self.order = []
        self.stage = 0
        self.scores = []
        self.best = None

    def advance(self):
        self.stage += 1
        self.scores = []
        self.best = None
        self.monitor.early_stopping_patience_counter = 0

    def active(self, streams):
        unfinished = [stream for stream in streams if stream.position < len(stream.order)]
        if self.phase == "warmup":
            warm = [stream for stream in unfinished
                    if stream.position < min(self.config.warmup_episodes, len(stream.order))]
            if warm:
                return warm
            self.order = sorted(self.density, key=lambda name: -self.density[name][0] / max(1, self.density[name][1]))
            self.phase = "curriculum"
        while self.phase == "curriculum" and self.stage < len(self.order):
            selected = [stream for stream in unfinished if stream.benchmark == self.order[self.stage]]
            if selected:
                return selected
            self.advance()
        if self.phase == "curriculum":
            # Joint continuation consumes each stream's remaining scheduled episodes.
            self.phase = "joint"
        return unfinished

    def observe(self, records):
        if self.phase == "warmup":
            for record in records:
                counters = self.density[record.task["benchmark"]]
                counters[0] += record.score > 0
                counters[1] += 1
        elif self.phase == "curriculum":
            from transformers import TrainerControl, TrainerState
            self.scores.extend(record.score for record in records)
            window = self.config.curriculum_window
            while len(self.scores) >= window:
                score = sum(self.scores[:window]) / window
                self.scores = self.scores[window:]
                control = TrainerControl()
                self.monitor.on_evaluate(SimpleNamespace(greater_is_better=True, metric_for_best_model="score"),
                                         TrainerState(best_metric=self.best), control, {"eval_score": score})
                self.best = score if self.best is None else max(self.best, score)
                if control.should_training_stop:
                    self.advance()
                    break

    def state_dict(self):
        return {"phase": self.phase, "density": self.density, "order": self.order,
                "stage": self.stage, "scores": self.scores, "best": self.best,
                "monitor": self.monitor.state()}

    def load_state_dict(self, state):
        from transformers import EarlyStoppingCallback
        for name in ("phase", "density", "order", "stage", "scores", "best"):
            setattr(self, name, state[name])
        self.monitor = EarlyStoppingCallback.from_state(state["monitor"])
