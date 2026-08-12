"""Unit tests for the loss/cost functions in ``src/losses.py``.

Run just this file:
    pytest tests/unit/test_losses.py -v

Run every test in the project:
    pytest -v

To finish a skipped test:
    1. Remove the ``@pytest.mark.skip(...)`` decorator
    2. Replace the ``pass`` body with real assertions (read the hints)
    3. Run pytest again and watch that test go from SKIPPED to PASSED
"""
import numpy as np
import pytest

from src.losses import categorical_cross_entropy_cost, categorical_cross_entropy_loss
from src.activations import softmax

# ---------------------------------------------------------------------------
# categorical_cross_entropy_loss() — per-sample loss, returns (1, m)
# ---------------------------------------------------------------------------

def test_loss_shape_is_one_by_m():
    """The loss must return one value per sample, shaped (1, m), not a scalar."""
    z = np.random.randn(5, 5)
    a = softmax(z)
    y = np.eye(5)

    loss = categorical_cross_entropy_loss(a, y)

    expected_shape = (1, 5)

    assert loss.shape == expected_shape

def test_loss_perfect_predictions_is_zero():
    """Perfectly confident predictions must give a loss of zero."""
    a = np.eye(3)
    y = np.eye(3)

    loss = categorical_cross_entropy_loss(a, y)

    np.testing.assert_allclose(loss, np.zeros(loss.shape))


def test_loss_known_small_case():
    """A single uncertain correct prediction has a hand-computable loss."""
    a = np.array([
        [0.8],
        [0.2],
    ])

    y = np.array([
        [1],
        [0],
    ])

    loss = categorical_cross_entropy_loss(a, y)

    true_loss = -1 * np.log(0.8)
    np.testing.assert_allclose(loss, true_loss) 

# ---------------------------------------------------------------------------
# categorical_cross_entropy_cost() — batch-averaged cost, returns a scalar
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_cost_uniform_predictions_equals_ln_of_n_classes():
    """A uniform prediction over K classes must cost exactly ln(K)."""
    # Goal: the defining sanity check for any cross-entropy implementation.
    # Hint: a = np.full((10, m), 0.1) (every class probability 0.1), m = 10000
    # balanced labels: y[rng choice, column] = 1
    # np.testing.assert_allclose(cost, np.log(10), rtol=1e-6)
    # vs ln(10) = 2.302585...
    # TODO: replace `pass` with real assertions
    pass


@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_cost_perfect_predictions_is_zero():
    """A cost of zero when every prediction is perfectly confident."""
    # Hint: same inputs as the perfect-loss test: a = y = np.eye(3)
    # assert cost == 0.0 and isinstance(cost, float)
    # (the docstring promises a scalar, so check the type too)
    # TODO: replace `pass` with real assertions
    pass


# ---------------------------------------------------------------------------
# cost() and loss() together
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_cost_equals_mean_of_per_sample_losses():
    """The cost must be the mean of the per-sample losses — this is the whole split."""
    # Goal: verify the loss/cost relationship you designed.
    # Hint: build any valid a (shape (3, 5), columns sum to 1) and one-hot y
    # loss = categorical_cross_entropy_loss(a, y)       # shape (1, 5)
    # cost = categorical_cross_entropy_cost(a, y)       # scalar
    # np.testing.assert_allclose(cost, np.mean(loss))
    # TODO: replace `pass` with real assertions
    pass