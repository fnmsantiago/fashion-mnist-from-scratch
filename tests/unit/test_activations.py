"""Unit tests for the activation functions in ``src/activations.py``.

Run every test in the project:
    pytest -v

Run just this file:
    pytest tests/test_activations.py -v

Run a single test by name:
    pytest tests/test_activations.py::test_relu_positive_values -v

Some tests below are already implemented; a few are left for YOU to write.
They are marked with ``@pytest.mark.skip`` and a TODO. To see which ones,
run:

    pytest tests/test_activations.py -v -rs      # -rs shows skip reasons

To finish a skipped test:
    1. Remove the ``@pytest.mark.skip(...)`` decorator
    2. Replace the ``pass`` body with real assertions (read the hints)
    3. Run pytest again and watch that test go from SKIPPED to PASSED
"""
import numpy as np
import pytest

from src.activations import relu, relu_backward


# ---------------------------------------------------------------------------
# relu() — completed tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# relu() — tests for YOU to implement
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_relu_preserves_input_shape():
    """The output must have the same shape as the input."""
    # Hint: z = np.random.randn(3, 4)  ->  the output shape should be (3, 4)
    # TODO: replace `pass` with real assertions
    pass


@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_relu_does_not_mutate_the_input():
    """relu() should be a pure function — it must not modify its argument."""
    # Hint: save a copy of z BEFORE calling relu (z_copy = z.copy()),
    # then call relu(z) and use np.array_equal(z, z_copy) to check the
    # original is still intact.
    # TODO: replace `pass` with real assertions
    pass


# ---------------------------------------------------------------------------
# relu_backward() — completed tests
# ---------------------------------------------------------------------------

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
# relu_backward() — tests for YOU to implement
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_relu_backward_expression_matches_np_where():
    """relu_backward should equal np.where(z > 0, da, 0) elementwise."""
    # This tests the function against a one-line statement of its own
    # definition rule — a clean way to verify a vectorized implementation.
    # Hint: da = np.random.randn(10, 5); z = np.random.randn(10, 5)
    #       expected = np.where(z > 0, da, 0.0)
    # TODO: replace `pass` with real assertions
    pass


@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_relu_backward_works_on_2d_batch_arrays():
    """Backward must handle (n_neurons, m) batches like real training caches."""
    # Hint: the forward pass caches z with shape (n_neurons, m), e.g. (64, 32).
    # Check that: (1) the output shape matches the input shapes, and
    # (2) positions where z > 0 keep their `da` value while the rest become 0.
    # TODO: replace `pass` with real assertions
    pass