import itertools
import math

import pytest
import torch

from credence.config import Config
from credence.credit import Calibrator, CreditStats, residual_bounds
from credence.models import OutcomeModel
from credence.retrieval import coverage, greedy_select, pool_summary
from credence.store import Asset


def test_streaming_statistics_and_pooled_variance():
    left, right = CreditStats(), CreditStats()
    values = [-0.4, 0.1, 0.6, -0.2, 0.3]
    for value in values[:3]:
        left.update(value)
    for value in values[3:]:
        right.update(value)
    merged = CreditStats.merge(left, right)
    assert merged.count == len(values)
    assert merged.mean == pytest.approx(torch.tensor(values).mean().item())
    assert merged.variance == pytest.approx(torch.tensor(values).var().item())


def test_statistics_reject_nonfinite_credit():
    with pytest.raises(ValueError):
        CreditStats().update(math.nan)


def test_greedy_coverage_against_exhaustive_optimum():
    generator = torch.Generator().manual_seed(20)
    for _ in range(8):
        tasks = torch.randn(3, 4, generator=generator)
        assets = torch.randn(5, 4, generator=generator)
        indices, value = greedy_select(tasks, assets, 2)
        optimum = max(float(coverage(tasks, assets[list(subset)]))
                      for subset in itertools.combinations(range(5), 2))
        assert len(indices) == len(set(indices)) <= 2
        assert float(value) + 1e-6 >= (1 - 1 / math.e) * optimum
        assert float(value) == pytest.approx(float(coverage(tasks, assets[indices])))


def test_empty_and_nonpositive_context():
    tasks = torch.eye(2)
    assert greedy_select(tasks, torch.empty(0, 2), 3)[0] == []
    assert greedy_select(tasks, -torch.eye(2), 3)[0] == []
    assert greedy_select(tasks, torch.eye(2), 0)[0] == []


def test_summary_preserves_task_identity_and_removal():
    tasks, embeddings = torch.eye(2), torch.eye(2)
    assets = [Asset("a", "memory", "A"), Asset("b", "tool", "B")]
    full = pool_summary(tasks, embeddings, assets, 3)
    removed = pool_summary(tasks, embeddings[:1], assets[:1], 3)
    changed_task = pool_summary(-tasks, embeddings, assets, 3)
    assert len(full) == 2 * 2 + 6 + 3
    assert not torch.equal(full, removed)
    assert not torch.equal(full, changed_task)
    assert removed[-3:].tolist() == [1, 0, 0]


def test_default_dimensions_match_paper():
    config = Config()
    model = OutcomeModel(config.summary_dim, config.predictor_hidden)
    assert config.summary_dim == 6174
    assert sum(parameter.numel() for parameter in model.parameters()) == 3162113


def test_order_statistic_and_decision_margin():
    estimate, uncertainty = residual_bounds([i / 100 for i in range(1, 101)], 0.9)
    assert estimate == pytest.approx(0.9)
    assert uncertainty == pytest.approx(0.03)
    calibrator = Calibrator()
    calibrator.fit([0.02, 0.03, 0.04], [0.05, 0.06, 0.07], [torch.zeros(2), torch.ones(2)])
    assert calibrator.residual == pytest.approx(0.07)
    assert calibrator.margin(4) > calibrator.margin(16)
    assert not calibrator.eligible(CreditStats())
    assert calibrator.eligible(CreditStats(100, -0.5, 0.01))
    assert calibrator.supported(torch.tensor([0.2, 0.8]))
    assert not calibrator.supported(torch.tensor([1.1, 0.8]))


def test_incomplete_calibration_gathers_evidence():
    calibrator = Calibrator()
    calibrator.fit([0.01], [], [torch.zeros(2)])
    assert not calibrator.eligible(CreditStats(100, 0.9, 0))


@pytest.mark.parametrize("values", [{"group_size": 1}, {"credit_horizon": 0}, {"calibration_quantile": 1}, {"token_cost": -1}])
def test_configuration_validation(values):
    with pytest.raises(ValueError):
        Config(**values)
