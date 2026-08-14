"""Unit tests for the helper functions in ``src/utils.py``.

Run just this file:
    pytest tests/unit/test_utils.py -v

Run every test in the project:
    pytest -v
"""
import numpy as np

from src.utils import one_hot_encode


# ---------------------------------------------------------------------------
# one_hot_encode() — integer labels to (n_classes, m) one-hot columns
# ---------------------------------------------------------------------------

def test_one_hot_encoded_shape_is_n_classes_by_m():
    """The encoded matrix must have shape (n_classes, m), not (m, n_classes)."""
    y = np.arange(5)
    encoded = one_hot_encode(y, 10)

    assert encoded.shape == (10, 5)

def test_each_column_has_exactly_one_one():
    """Every sample column must sum to 1 — exactly one class picked per sample."""
    m = 8
    y = np.random.randint(0, 10, size=m)
    encoded = one_hot_encode(y, 10)

    ones = np.ones((1, m))

    np.testing.assert_allclose(encoded.sum(axis=0, keepdims=True), ones)

def test_argmax_round_trips_to_original_labels():
    """argmax over the class axis must recover the original labels exactly."""
    m = 8
    y = np.random.randint(0, 10, size=m)
    encoded = one_hot_encode(y, 10)

    np.testing.assert_allclose(encoded.argmax(axis=0), y)

def test_known_small_case():
    """A tiny example where each 1 lands on a hand-checkable coordinate."""
    y = np.array([3, 0, 2])

    encoded = one_hot_encode(y, 4)

    expected = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])

    np.testing.assert_allclose(encoded, expected)

def test_consecutive_labels_form_the_identity_matrix():
    """Labels 0..K-1 in order must produce exactly np.eye(K)."""
    y = np.arange(5)
    encoded = one_hot_encode(y, 5)

    np.testing.assert_allclose(encoded, np.eye(5))