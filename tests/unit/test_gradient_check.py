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

def test_gradient_check_reports_small_errors_for_correct_backward():
    """Correct backpropagation must yield a tiny max relative error."""
    np.random.seed(0)
    network = NeuralNetwork([4, 5, 3])
    m = 6
    x = np.random.randn(4, m)
    y = one_hot_encode(np.random.randint(0, 3, size=m), 3)

    network_error, errors = gradient_check(network, x, y)

    very_small_number = 10e-7

    assert network_error <= very_small_number

    for error in errors.values():
        assert error <= very_small_number

def test_gradient_check_flags_a_broken_gradient():
    """A deliberately wrong gradient must produce a large reported error."""
    np.random.seed(0)
    network = NeuralNetwork([4, 5, 3])
    m = 6
    x = np.random.randn(4, m)
    y = one_hot_encode(np.random.randint(0, 3, size=m), 3)

    real_backward = network.backward

    def broken_backward(al, y, caches):
        grads = real_backward(al, y, caches)

        # corrupt the weight gradient of the first layer.
        grads[f"dW1"] += 12.34

        return grads

    # Use Python's feature to shadow an instance's function. 
    network.backward = broken_backward

    network_error, errors = gradient_check(network, x, y)

    very_small_number = 10e-7

    assert network_error > very_small_number

    for theta_name, error in errors.items():
        if theta_name == "W1":
            assert error > 10e-7
        else:
            assert error <= 10e-7


def test_gradient_check_reports_one_error_per_parameter():
    """The result must map every parameter key to a non-negative float."""
    np.random.seed(0)
    network = NeuralNetwork([4, 5, 3])
    m = 6
    x = np.random.randn(4, m)
    y = one_hot_encode(np.random.randint(0, 3, size=m), 3)

    network_error, errors = gradient_check(network, x, y)

    assert set(errors.keys()) == set(network.parameters.keys())

    assert network_error > 0.0

    for error in errors.values():
        assert error > 0.0
