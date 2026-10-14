import copy
import re
from dataclasses import asdict, dataclass, field

from .credit import CreditStats
from .tools import merge_tools, split_tools, validate_tool


@dataclass
class Asset:
    asset_id: str
    kind: str
    text: str
    title: str = ""
    keywords: list = field(default_factory=list)
    tokens: int = 0
    age: int = 0
    credit: CreditStats = field(default_factory=CreditStats)
    sources: list = field(default_factory=list)

    def card(self):
        return {"asset_id": self.asset_id, "type": self.kind, "title": self.title,
                "keywords": self.keywords[:10], "credit_mean": self.credit.mean,
                "credit_variance": self.credit.variance, "retrieval_count": self.credit.count,
                "age": self.age, "tokens": self.tokens}


@dataclass
class CreatorRecord:
    policy: str
    born: int
    tokens: int
    retired: int | None = None
    credits: list = field(default_factory=list)

    def reward(self, cost):
        return sum(self.credits) / max(1, len(self.credits)) - cost * self.tokens / 1000


@dataclass
class BufferedAsset:
    asset: Asset
    removed_at: int
    gains: list = field(default_factory=list)


class Store:
    def __init__(self, capacity=1000, horizon=10):
        self.capacity = capacity
        self.horizon = horizon
        self.assets = {}
        self.creators = {}
        self.retired = {}
        self.buffer = {}
        self.counter = 0
        self.episode = 0

    def clone(self):
        return copy.deepcopy(self)

    def add(self, kind, text, title, keywords, tokens, policy, sources=(), credit=None):
        if kind not in ("memory", "tool") or not text.strip():
            raise ValueError("Invalid asset kind or empty payload")
        if kind == "tool":
            validate_tool(text)
        if len(self.assets) >= self.capacity:
            raise ValueError("Asset store capacity reached")
        self.counter += 1
        aid = f"{kind}-{self.counter:08d}"
        asset = Asset(aid, kind, text, title, list(dict.fromkeys(keywords)), tokens,
                      credit=copy.deepcopy(credit or CreditStats()), sources=list(sources))
        self.assets[aid] = asset
        self.creators[aid] = CreatorRecord(policy, self.episode, tokens)
        return aid

    def retire(self, aid, buffered=True):
        asset = self.assets.pop(aid)
        self.retired[aid] = copy.deepcopy(asset)
        self.creators[aid].retired = self.episode
        if buffered:
            self.buffer[aid] = BufferedAsset(copy.deepcopy(asset), self.episode)
        return asset

    def observe(self, aid, value):
        self.assets[aid].credit.update(value)
        self.creators[aid].credits.append(float(value))

    def advance(self):
        self.episode += 1
        for asset in self.assets.values():
            asset.age += 1
        for aid, entry in list(self.buffer.items()):
            if self.episode - entry.removed_at > self.horizon:
                del self.buffer[aid]

    def restore(self, aid, gain, supported, context_changed):
        entry = self.buffer[aid]
        if supported and context_changed:
            entry.gains.append(float(gain))
        if not entry.gains or sum(entry.gains) / len(entry.gains) <= 0:
            return None
        if len(self.assets) >= self.capacity:
            return None
        asset = entry.asset
        new_id = self.add(asset.kind, asset.text, asset.title, asset.keywords,
                          asset.tokens, "curator", (aid,), asset.credit)
        del self.buffer[aid]
        return new_id

    def apply(self, operation, tokens, partition=None, protect=True, content_partition=None):
        aid, action = operation.get("asset_id"), operation.get("action")
        if aid not in self.assets or action not in ("keep", "remove", "merge", "split"):
            raise ValueError("Invalid curator operation")
        asset = self.assets[aid]
        if action == "keep":
            return [aid]
        if action == "remove":
            if protect and asset.credit.mean > 0 and asset.credit.count <= 2:
                return [aid]
            self.retire(aid)
            return []
        if action == "merge":
            target = operation.get("target")
            if target == aid or target not in self.assets or self.assets[target].kind != asset.kind:
                raise ValueError("Merge requires a compatible target")
            other = self.assets[target]
            text = (asset.text + "\n" + other.text) if asset.kind == "memory" else merge_tools(
                asset.text, other.text, asset.credit.mean, other.credit.mean)
            merged = CreditStats.merge(asset.credit, other.credit)
            size = tokens(text)
            self.retire(aid)
            self.retire(target)
            new_id = self.add(asset.kind, text, asset.title, asset.keywords + other.keywords,
                              size, "curator", (aid, target), merged)
            self.assets[new_id].age = min(asset.age, other.age)
            return [new_id]
        if asset.kind == "memory":
            if partition is None or len(asset.keywords) < 2:
                raise ValueError("Memory splitting requires a keyword partition")
            groups = partition(asset.keywords)
            if len(groups) != 2 or not all(groups) or sorted(sum(groups, [])) != list(range(len(asset.keywords))):
                raise ValueError("Keyword partition requires two disjoint exhaustive groups")
            if content_partition is None:
                units = [unit.strip() for unit in re.split(r"(?<=[.!?;])\s+|\n+", asset.text) if unit.strip()]
                labels = []
                for unit in units:
                    matches = [sum(keyword.lower() in unit.lower() for keyword in (asset.keywords[i] for i in group)) for group in groups]
                    if max(matches) == 0 or matches[0] == matches[1]:
                        raise ValueError("Memory content requires an unambiguous partition")
                    labels.append(matches.index(max(matches)))
                texts = [" ".join(unit for unit, label in zip(units, labels) if label == k) for k in range(2)]
            else:
                texts = content_partition(asset, groups)
            if len(texts) != 2 or not all(text.strip() for text in texts) or texts[0] == texts[1]:
                raise ValueError("Memory content requires two distinct nonempty patterns")
            payloads = [(text, [asset.keywords[i] for i in group]) for text, group in zip(texts, groups)]
        else:
            payloads = [(code, list(asset.keywords)) for code in split_tools(asset.text)]
        if len(self.assets) - 1 + len(payloads) > self.capacity:
            raise ValueError("Split exceeds store capacity")
        sizes = [tokens(text) for text, _ in payloads]
        self.retire(aid)
        n, remainder = divmod(asset.credit.count, len(payloads))
        children = []
        for i, (text, keywords) in enumerate(payloads):
            inherited = CreditStats(n + (i < remainder), asset.credit.mean, asset.credit.variance)
            children.append(self.add(asset.kind, text, asset.title, keywords,
                                     sizes[i], "curator", (aid,), inherited))
        return children

    def state_dict(self):
        return {"capacity": self.capacity, "horizon": self.horizon, "counter": self.counter,
                "episode": self.episode, "assets": {key: asdict(value) for key, value in self.assets.items()},
                "creators": {key: asdict(value) for key, value in self.creators.items()},
                "retired": {key: asdict(value) for key, value in self.retired.items()},
                "buffer": {key: asdict(value) for key, value in self.buffer.items()}}

    @classmethod
    def from_state_dict(cls, state):
        store = cls(state["capacity"], state["horizon"])
        store.counter, store.episode = state["counter"], state["episode"]
        def asset_of(value):
            value = dict(value)
            value["credit"] = CreditStats(**value["credit"])
            return Asset(**value)
        store.assets = {key: asset_of(value) for key, value in state["assets"].items()}
        store.retired = {key: asset_of(value) for key, value in state["retired"].items()}
        store.creators = {key: CreatorRecord(**value) for key, value in state["creators"].items()}
        store.buffer = {key: BufferedAsset(asset_of(value["asset"]), value["removed_at"], value["gains"])
                        for key, value in state["buffer"].items()}
        return store
