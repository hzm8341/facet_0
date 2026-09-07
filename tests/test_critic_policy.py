import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from facet0.models.action_wrench_critic import ActionWrenchCritic, CriticConfig
from facet0.policies.value_guided_policy import CandidateScores, rank_candidates
from facet0.training.train_critic import expected_return, reward_proxy, scalar_to_two_hot


def test_critic_has_distribution_and_four_auxiliary_heads():
    model = ActionWrenchCritic(CriticConfig(input_dim=8, width=16, num_return_bins=11), rngs=nnx.Rngs(0))
    output = model(jnp.zeros((3, 8)))
    assert output["return_logits"].shape == (3, 11)
    assert set(output) == {
        "return_logits",
        "success_logit",
        "efficiency",
        "violation_logit",
        "recovery_logit",
    }


def test_two_hot_projection_preserves_in_range_expectation():
    support = np.linspace(-1, 1, 5)
    values = np.asarray([-0.75, 0.0, 0.6])
    projected = scalar_to_two_hot(values, support)
    np.testing.assert_allclose(projected.sum(axis=-1), 1.0)
    np.testing.assert_allclose(projected @ support, values)
    with pytest.raises(ValueError):
        scalar_to_two_hot(values, support[::-1])


def test_high_wrench_risk_candidate_ranks_lower_and_disable_is_identity():
    scores = CandidateScores(
        expected_return=np.asarray([0.8, 0.9]),
        success_probability=np.asarray([0.8, 0.8]),
        violation_probability=np.asarray([0.05, 0.9]),
    )
    np.testing.assert_array_equal(rank_candidates(scores), [0, 1])
    np.testing.assert_array_equal(rank_candidates(scores, enabled=False), [0, 1])


def test_reward_proxy_penalizes_violation_and_expected_return_is_finite():
    reward = reward_proxy(
        success=np.asarray([1, 1]),
        duration_fraction=np.asarray([0.5, 0.5]),
        wrench_violation=np.asarray([0, 1]),
    )
    assert reward[0] > reward[1]
    assert np.isfinite(expected_return(np.zeros((2, 5)), np.linspace(-1, 1, 5))).all()
