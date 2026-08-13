"""Unit tests for the loss/cost functions in ``src/losses.py``.

Run just this file:
    pytest tests/unit/test_losses.py -v

Run every test in the project:
    pytest -v
"""
import numpy as np

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

def test_cost_uniform_predictions_equals_ln_of_n_classes():
    """A uniform prediction over K classes must cost exactly ln(K)."""
    K = 10

    # Initialize an activation matrix that has uniform probability for each class
    a = np.full((K, K), 1/K)

    y = np.eye(K)

    cost = categorical_cross_entropy_cost(a, y)

    np.testing.assert_allclose(cost, np.log(K))

def test_cost_perfect_predictions_is_zero():
    """A cost of zero when every prediction is perfectly confident."""
    a = y = np.eye(3)

    cost = categorical_cross_entropy_cost(a, y)

    np.testing.assert_allclose(cost, 0.0)

    assert isinstance(cost, float)
