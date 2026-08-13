"""Unit tests for the activation functions in ``src/activations.py``.

Run every test in the project:
    pytest -v

Run just this file:
    pytest tests/unit/test_activations.py -v

Run a single test by name:
    pytest tests/unit/test_activations.py::test_relu_positive_values -v
"""
import numpy as np

from src.activations import relu, relu_backward, softmax

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


# ---------------------------------------------------------------------------
# softmax()
# ---------------------------------------------------------------------------

def test_softmax_columns_sum_to_one():
    """Every column of the output must sum to 1 (a valid probability distribution)."""
    z = np.random.randn(3, 5)
    a = softmax(z)

    expected = np.ones((1, 5))
    np.testing.assert_allclose(a.sum(axis=0, keepdims=True), expected)

def test_softmax_preserves_input_shape():
    """The output must have the same shape as the input."""
    z = np.random.rand(3, 5)
    a = softmax(z)

    expected_shape = (3, 5)
    assert a.shape == expected_shape

def test_softmax_outputs_are_positive_probabilities():
    """All outputs must be in (0, 1]: no negatives and nothing greater than 1."""
    z = np.random.randn(3, 5)
    a = softmax(z)

    # Lower bound: every probability is strictly positive.
    lower_limit = np.zeros(a.shape)
    np.testing.assert_array_less(lower_limit, a)

    # Upper bound: no probability exceeds 1.
    assert (a <= 1.0).all()

def test_softmax_is_numerically_stable_for_large_logits():
    """Huge logits (e.g. 1000) must produce finite probabilities, not NaN/inf."""
    huge_z = np.random.randn(3, 5) * 1000
    a = softmax(huge_z)

    np.testing.assert_allclose(np.isfinite(a), True)

def test_softmax_is_shift_invariant():
    """Adding a constant to every logit in a column must not change the result."""
    z = np.random.randn(3, 5)
    np.testing.assert_allclose(softmax(z), softmax(z+1000))

def test_softmax_known_small_case():
    """Two tied logits must give exactly [0.5, 0.5] in each column."""
    z = np.array([
        [1.0, 2.0],
        [1.0, 2.0],
    ])

    expected = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    np.testing.assert_allclose(softmax(z), expected)

def test_softmax_does_not_mutate_the_input():
    """softmax() should be a pure function — it must not modify its argument."""
    z = np.random.randn(3, 5)
    z_copy = z.copy()

    softmax(z)

    np.testing.assert_array_equal(z, z_copy)