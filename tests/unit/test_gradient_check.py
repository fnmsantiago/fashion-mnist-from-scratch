"""Unit tests for the gradient-checking utility in ``src/gradient_check.py``.

Run just this file:
    pytest tests/unit/test_gradient_check.py -v

Run every test in the project:
    pytest -v
"""
import numpy as np
import pytest

from src.gradient_check import gradient_check
from src.model import NeuralNetwork
from src.utils import one_hot_encode


# ---------------------------------------------------------------------------
# gradient_check() — the verification utility
# ---------------------------------------------------------------------------
# Learning goals: implement src/gradient_check.py, then un-skip each test
# (remove the @pytest.mark.skip decorator) and fill in the assertion.

@pytest.mark.skip(reason="TODO: implement src/gradient_check.py")
def test_gradient_check_reports_small_errors_for_correct_backward():
    """Correct backpropagation must yield a tiny max relative error."""
    np.random.seed(0)
    network = NeuralNetwork([4, 5, 3])
    m = 6
    x = np.random.randn(4, m)
    y = one_hot_encode(np.random.randint(0, 3, size=m), 3)

    # TODO: call gradient_check(network, x, y) and assert every reported
    # error is small — recall the course's gradient-checking threshold.
    pass


@pytest.mark.skip(reason="TODO: implement src/gradient_check.py")
def test_gradient_check_flags_a_broken_gradient():
    """A deliberately wrong gradient must produce a large reported error."""
    np.random.seed(0)
    network = NeuralNetwork([4, 5, 3])
    m = 6
    x = np.random.randn(4, m)
    y = one_hot_encode(np.random.randint(0, 3, size=m), 3)

    # TODO: make backward() return wrong gradients somehow, then assert
    # gradient_check reports large errors — the checker must be able to
    # detect a bug instead of always passing.
    pass


@pytest.mark.skip(reason="TODO: implement src/gradient_check.py")
def test_gradient_check_reports_one_error_per_parameter():
    """The result must map every parameter key to a non-negative float."""
    np.random.seed(0)
    network = NeuralNetwork([4, 5, 3])
    m = 6
    x = np.random.randn(4, m)
    y = one_hot_encode(np.random.randint(0, 3, size=m), 3)

    # TODO: assert set(errors.keys()) == set(network.parameters.keys()) and
    # that every reported value is a non-negative float.
    pass
