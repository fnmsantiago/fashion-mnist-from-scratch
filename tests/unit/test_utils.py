"""Unit tests for the helper functions in ``src/utils.py``.

Run just this file:
    pytest tests/unit/test_utils.py -v

Run every test in the project:
    pytest -v

To finish a skipped test:
    1. Remove the ``@pytest.mark.skip(...)`` decorator
    2. Replace the ``pass`` body with real assertions (read the hints)
    3. Run pytest again and watch that test go from SKIPPED to PASSED
"""
import numpy as np
import pytest

from src.utils import one_hot_encode, random_mini_batches


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


# ---------------------------------------------------------------------------
# random_mini_batches() — shuffled mini-batches that keep x and y paired
# ---------------------------------------------------------------------------

def test_full_batches_plus_one_partial_batch():
    """Samples must split into full-size batches, with leftovers as a final batch."""
    batch_size = 10
    m = 14

    x = np.random.randn(m, m)
    y = np.random.randn(10, m)

    mini_batches = random_mini_batches(x, y, batch_size)

    # One full batch of 10, then a final partial batch of the 4 leftovers.
    expected_widths = [10, 4]
    assert [xb.shape[1] for xb, yb in mini_batches] == expected_widths

def test_all_samples_survive_the_shuffle():
    """Every original sample must appear exactly once across all batches."""
    m = 10
    batch_size = 4
    seed = 1234

    # Brand each sample by replacing the first rows of samples/labels 
    # with the their original indexes.
    brands = np.arange(m)
    x = np.zeros((3, m))
    x[0] = brands
    y = np.zeros((1, m))
    y[0] = brands

    # Here, they are shuffled.
    mini_batches = random_mini_batches(x, y, batch_size, seed)

    brand_rows_x = []
    brand_rows_y = []

    for x_batch, y_batch in mini_batches:
        # Take only the brands, which should still be on the 1st row.
        brand_rows_x.append(x_batch[0])
        brand_rows_y.append(y_batch[0])

    # Stitch the per-batch id arrays back into one array of length m.
    ids_x = np.concatenate(brand_rows_x)
    ids_y = np.concatenate(brand_rows_y)

    # Sort back to canonical order: every id 0..m-1 appears once, no drops.
    np.testing.assert_allclose(np.sort(ids_x), brands)
    np.testing.assert_allclose(np.sort(ids_y), brands)

def test_x_and_y_columns_stay_paired():
    """Each sample's features and its labels must never be separated."""
    m = 10
    x = np.zeros((3, m))
    y = np.zeros((1, m))
    batch_size = 4

    # Brand the samples and labels with the same id, using their original index.
    brands = np.arange(m)
    x[0] = brands
    y[0] = brands

    mini_batches = random_mini_batches(x, y,  batch_size)

    # Retrieve only the 1st row from each minibatch, as they contain the ids.
    brand_rows_x = [x_batch[0] for x_batch, _ in mini_batches]
    brand_rows_y = [y_batch[0] for _, y_batch in mini_batches]

    # Concatenate each sub list to form a single Numpy.
    ids_x = np.concatenate(brand_rows_x)
    ids_y = np.concatenate(brand_rows_y)

    # If the sample and label pair were properly batched together,
    # then both Numpy arrays must be identical.
    np.testing.assert_allclose(ids_x, ids_y)

def test_same_seed_reproduces_the_same_batches():
    """Two calls with the same seed must return identical batches."""
    m = 10
    batch_size = 5

    x = np.random.randn(3, m)
    y = np.random.randn(1, m)

    batch_seed = 1234

    # The function returns a list of (x_batch, y_batch) tuples, not a single
    # array, so compare the two sequences element by element, in order.
    batches_1 = random_mini_batches(x, y, batch_size, batch_seed)
    batches_2 = random_mini_batches(x, y, batch_size, batch_seed)

    # Same seed means the two lists must have the same number of batches.
    assert len(batches_1) == len(batches_2)

    # Pair the elements with the same index together using zip and compare.
    for (x_batch_1, y_batch_1), (x_batch_2, y_batch_2) in zip(batches_1, batches_2):
        np.testing.assert_allclose(x_batch_1, x_batch_2)
        np.testing.assert_allclose(y_batch_1, y_batch_2)

def test_different_seeds_reshuffle_the_order():
    """A different seed must change the order while keeping every sample."""
    m = 10
    batch_size = 5

    x = np.random.randn(3, m)
    y = np.random.randn(1, m)

    brand = np.arange(m)
    x[0] = brand
    y[0] = brand

    batch_seed_1 = 1234
    batch_seed_2 = 5678

    batches_1 = random_mini_batches(x, y, batch_size, batch_seed_1)
    batches_2 = random_mini_batches(x, y, batch_size, batch_seed_2)

    ids_x_1 = np.concatenate([x_batch[0] for x_batch, _ in batches_1])
    ids_y_1 = np.concatenate([y_batch[0] for _, y_batch in batches_1])

    ids_x_2 = np.concatenate([x_batch[0] for x_batch, _ in batches_2])
    ids_y_2 = np.concatenate([y_batch[0] for _, y_batch in batches_2])

    # Different seeds should've produced different orders.
    assert not np.array_equal(ids_x_1, ids_x_2)
    assert not np.array_equal(ids_y_1, ids_y_2)

    # Retains all samples.
    assert ids_x_1.shape[0] == m
    assert ids_y_1.shape[0] == m
    assert ids_x_2.shape[0] == m
    assert ids_y_2.shape[0] == m

    # Retains sample-label pairings.
    np.testing.assert_allclose(ids_x_1, ids_y_1)
    np.testing.assert_allclose(ids_x_2, ids_y_2)

def test_large_batch_size_still_covers_all_samples():
    """A batch_size bigger than m must still return usable batches."""
    m = 10
    batch_size = 15

    # Brand each sample with its original index so the content check can
    # catch dropped samples, not just zero-filled placeholders.
    brands = np.arange(m)
    x = np.zeros((3, m))
    x[0] = brands
    y = np.zeros((1, m))
    y[0] = brands

    mini_batches = random_mini_batches(x, y, batch_size)

    # Only 1 batch should've been created.
    assert len(mini_batches) == 1

    x_batch = mini_batches[0][0]
    y_batch = mini_batches[0][1]

    # All samples/labels retained — every brand appears exactly once.
    np.testing.assert_allclose(np.sort(x_batch[0]), brands)
    np.testing.assert_allclose(np.sort(y_batch[0]), brands)

    # Sample-label pairings are still intact inside the batch.
    np.testing.assert_allclose(x_batch[0], y_batch[0])
