"""Unit tests for the activation functions in ``src/activations.py``.

Run every test in the project:
    pytest -v

Run just this file:
    pytest tests/unit/test_activations.py -v

Run a single test by name:
    pytest tests/unit/test_activations.py::test_relu_positive_values -v
"""
import numpy as np
import pytest

from src.activations import relu, relu_backward

def test_relu_positive_values_pass_through_unchanged():
    """Positive inputs should be returned exactly as-is."""
    z = np.array([1.0, 2.5, 0.3])
    expected = np.array([1.0, 2.5, 0.3])
    np.testing.assert_allclose(relu(z), expected)


def test_relu_zeroes_out_negative_values():
    """Negative inputs should all become zero."""
    z = np.array([-1.0, -2.5, -0.3])
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(relu(z), expected)


def test_relu_handles_a_mix_of_values():
    """Positive, negative, and zero inputs in one array."""
    z = np.array([-1.0, 0.0, 2.0])
    expected = np.array([0.0, 0.0, 2.0])
    np.testing.assert_allclose(relu(z), expected)

def test_relu_preserves_input_shape():
    """The output must have the same shape as the input."""
    z = np.random.randn(3, 4)
    expected_shape = z.shape
    assert relu(z).shape == expected_shape

def test_relu_does_not_mutate_the_input():
    """relu() should be a pure function — it must not modify its argument."""
    z = np.random.rand(3, 4)
    z_copy = z.copy()
    relu(z)
    np.testing.assert_array_equal(z, z_copy)

def test_relu_backward_keeps_gradient_where_z_positive():
    """Where z > 0, the upstream gradient should pass through unchanged."""
    da = np.array([1.0, 2.0, 3.0])
    z = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(relu_backward(da, z), expected)


def test_relu_backward_zeroes_gradient_where_z_negative():
    """Where z < 0, the gradient should be completely zeroed out."""
    da = np.array([1.0, 2.0, 3.0])
    z = np.array([-1.0, -2.0, -3.0])
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(relu_backward(da, z), expected)


def test_relu_backward_zero_gradient_at_z_equals_zero():
    """At z == 0, the subgradient convention used here gives 0."""
    da = np.array([5.0])
    z = np.array([0.0])
    expected = np.array([0.0])
    np.testing.assert_allclose(relu_backward(da, z), expected)
