import pytest
import torch

from credence.groups import Candidate, CandidateGroup, GroupBuffer, Sample, group_advantages, grpo_loss
from credence.store import Store


def sample(prompt="shared decision"):
    return Sample(prompt, [1], [2], "{}", torch.tensor([-0.7]))


def creator_group():
    candidates = []
    for value in (0.4, -0.2):
        store = Store()
        aid = store.add("memory", "A reusable instruction", "Instruction", [], 100, "writer")
        store.observe(aid, value)
        candidates.append(Candidate(sample(), store, aid, 100))
    return CandidateGroup("g", "writer", 0, 0, 0, candidates)


def test_rewards_mature_after_window_or_version_retirement():
    group = creator_group()
    assert not group.mature(1, 2, 0.002)
    group.candidates[0].store.retire(group.candidates[0].asset_id)
    assert not group.mature(1, 2, 0.002)
    assert group.candidates[0].reward == pytest.approx(0.3998)
    assert group.mature(2, 2, 0.002)
    assert group.candidates[1].reward == pytest.approx(-0.2002)


def test_invalid_candidates_receive_proposed_payload_cost():
    group = creator_group()
    group.candidates[0].valid = False
    assert not group.mature(0, 2, 0.002)
    assert group.candidates[0].reward == pytest.approx(-0.0002)


def test_group_buffer_preserves_pending_groups_and_counts_policy_lag():
    buffer = GroupBuffer(2)
    group = creator_group()
    buffer.add(group)
    assert buffer.release({"writer": 1}, 2, 0.002) == []
    assert buffer.statistics()["pending_groups"] == 1
    assert buffer.release({"writer": 3}, 2, 0.002) == []
    assert buffer.statistics()["excluded_groups"] == 1


def test_invalid_first_candidate_does_not_delay_valid_reward():
    buffer = GroupBuffer()
    group = creator_group()
    group.candidates[0].valid = False
    group.candidates[1].store.advance()
    group.candidates[1].store.advance()
    buffer.add(group)
    assert len(buffer.release({"writer": 0}, 2, 0.002)) == 1


def test_curator_reward_uses_matched_score_differences():
    group = creator_group()
    group.head = "curator"
    group.reference_scores = [0.1, 0.4]
    group.candidates[0].scores = [0.4, 0.5]
    group.candidates[1].scores = [0.1, 0.3]
    assert group.mature(2, 2, 0)
    assert group.candidates[0].reward == pytest.approx(0.2)
    assert group.candidates[1].reward == pytest.approx(-0.05)


def test_candidate_prompt_mismatch_is_rejected():
    group = creator_group()
    group.candidates[1].sample.prompt = "another decision"
    with pytest.raises(ValueError):
        GroupBuffer().add(group)


def test_equal_rewards_have_zero_advantage():
    assert torch.equal(group_advantages([0.2, 0.2]), torch.zeros(2))
    values = group_advantages([0.4, -0.2, 0.1])
    assert float(values.mean()) == pytest.approx(0, abs=1e-6)
    assert float(values.square().mean()) == pytest.approx(1)


def test_advantage_epsilon_is_added_after_standard_deviation():
    assert torch.allclose(group_advantages([2.0, 0.0], epsilon=0.1), torch.tensor([1 / 1.1, -1 / 1.1]))


def test_token_ratios_and_clipping_affect_each_candidate_gradient():
    current = torch.tensor([0.4, -0.1], requires_grad=True)
    behavior = torch.zeros(2)
    reference = current.detach().clone()
    loss = grpo_loss(current, behavior, reference, 1, clip_ratio=0.2, kl_weight=0)
    loss.backward()
    assert current.grad[0] == 0
    assert current.grad[1] < 0
    assert float(loss.detach()) == pytest.approx(-(1.2 + torch.exp(torch.tensor(-0.1)).item()) / 2)


def test_reference_divergence_is_nonnegative():
    current = torch.tensor([-0.7, -1.2], requires_grad=True)
    loss = grpo_loss(current, current.detach(), torch.tensor([-0.4, -0.8]), 0, kl_weight=1)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(current.grad).all()
