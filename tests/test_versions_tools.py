import pytest

from credence.credit import CreditStats
from credence.store import Store
from credence.tools import merge_tools, split_tools, validate_tool


def memory(store, text="A useful procedure", keywords=("first", "second")):
    return store.add("memory", text, "Procedure", keywords, len(text), "writer")


def test_merge_closes_creators_and_preserves_evidence():
    store = Store()
    left, right = memory(store), memory(store, "Another procedure")
    store.observe(left, 0.2)
    store.observe(left, 0.4)
    store.observe(right, -0.2)
    children = store.apply({"asset_id": left, "action": "merge", "target": right}, len)
    merged = store.assets[children[0]]
    assert merged.credit.count == 3
    assert merged.credit.mean == pytest.approx(0.4 / 3)
    assert store.creators[left].retired == store.episode
    assert store.creators[right].retired == store.episode
    assert store.creators[children[0]].credits == []
    assert merged.sources == [left, right]


def test_split_preserves_total_count_and_creates_new_records():
    store = Store()
    parent = memory(store, "First inspect the input. Second compute the sum.")
    for credit in (0.1, 0.2, 0.3, 0.4, 0.5):
        store.observe(parent, credit)
    children = store.apply({"asset_id": parent, "action": "split"}, len, lambda words: [[0], [1]])
    assert sum(store.assets[child].credit.count for child in children) == 5
    assert all(store.assets[child].credit.variance == store.retired[parent].credit.variance for child in children)
    assert all(store.creators[child].credits == [] for child in children)
    assert len({store.assets[child].text for child in children}) == 2
    assert store.creators[parent].retired == store.episode


def test_retirement_stops_creator_reward_and_restoration_opens_version():
    store = Store(horizon=2)
    aid = memory(store)
    store.observe(aid, -0.3)
    original = store.creators[aid].reward(0)
    store.retire(aid)
    assert store.restore(aid, 0.2, supported=False, context_changed=True) is None
    assert store.restore(aid, 0.2, supported=True, context_changed=False) is None
    restored = store.restore(aid, 0.2, supported=True, context_changed=True)
    assert restored != aid
    store.observe(restored, 0.5)
    assert store.creators[aid].reward(0) == original
    assert store.creators[restored].credits == [0.5]


def test_restoration_buffer_expires():
    store = Store(horizon=2)
    aid = memory(store)
    store.retire(aid)
    store.advance()
    store.advance()
    assert aid in store.buffer
    store.advance()
    assert aid not in store.buffer


def test_recent_positive_asset_protection():
    store = Store()
    aid = memory(store)
    store.observe(aid, 0.3)
    store.apply({"asset_id": aid, "action": "remove"}, len)
    assert aid in store.assets
    store.apply({"asset_id": aid, "action": "remove"}, len, protect=False)
    assert aid in store.buffer


def test_store_round_trip_preserves_versions_and_credit():
    store = Store()
    aid = memory(store)
    store.observe(aid, 0.2)
    store.retire(aid)
    restored = Store.from_state_dict(store.state_dict())
    assert restored.state_dict() == store.state_dict()


def test_tool_validation_executes_supplied_tests():
    code = "def add(a, b):\n    return a + b"
    assert validate_tool(code, [{"function": "add", "args": [2, 3], "expected": 5}]) == ["add"]
    with pytest.raises(ValueError):
        validate_tool(code, [{"function": "add", "args": [2, 3], "expected": 4}])


@pytest.mark.parametrize("code", ["def f():\n    return open('secret').read()", "import os\ndef f():\n    return os.getenv('TOKEN')", "def f():\n    return eval('1')", "def f():\n    return (1).__class__"])
def test_tool_interface_rejects_external_access(code):
    with pytest.raises(ValueError):
        validate_tool(code)


def test_type_aware_tool_merge_and_split():
    first = "def f(a):\n    return a + 1"
    second = "def f(a):\n    return a + 2"
    other = "def g(a):\n    return a * 2"
    assert merge_tools(first, second, 0.5, 0.2) == first
    combined = merge_tools(first, other, 0.5, 0.2)
    assert len(split_tools(combined)) == 2
    with pytest.raises(ValueError):
        merge_tools(first, "def f(a, b):\n    return a + b", 0, 0)


def test_incompatible_merge_keeps_original_store():
    store = Store()
    aid = memory(store)
    tool = store.add("tool", "def count(items):\n    return len(items)", "Count", [], 5, "builder")
    before = store.state_dict()
    with pytest.raises(ValueError):
        store.apply({"asset_id": aid, "action": "merge", "target": tool}, len)
    assert store.state_dict() == before


def test_split_declines_inseparable_content():
    store = Store()
    aid = memory(store)
    before = store.state_dict()
    with pytest.raises(ValueError):
        store.apply({"asset_id": aid, "action": "split"}, len, lambda words: [[0], [1]])
    assert store.state_dict() == before


def test_dependent_tool_functions_remain_bundled():
    with pytest.raises(ValueError):
        split_tools("def f(x):\n    return x + 1\n\ndef g(x):\n    return f(x) * 2")


def test_malformed_tool_is_a_validation_failure():
    with pytest.raises(ValueError):
        validate_tool("def broken(:\n    pass")
