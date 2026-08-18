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
    # Goal: verify the contract's core promise about batch sizes.
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
    # Goal: shuffling is a permutation, not a filter. The data itself carries
    # each sample's identity, so the reassembled set can be checked in ANY
    # order — no need to know the internal permutation.
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

@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_x_and_y_columns_stay_paired():
    """Each sample's features and its labels must never be separated."""
    # Goal: this is the whole point of one shared shuffle.
    # Hint: make each sample's label a deterministic function of its own x
    # column (encode the column identity into both). After batching, derive
    # each x_batch's labels from x and compare against the matching y_batch.
    # TODO: replace `pass` with real assertions
    pass

@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_same_seed_reproduces_the_same_batches():
    """Two calls with the same seed must return identical batches."""
    # Goal: determinism is what makes seeded shuffling reproducible.
    # Hint: call the function twice with the same seed. What comparison would
    # show the two results are literally the same, batch by batch?
    # TODO: replace `pass` with real assertions
    pass


@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_different_seeds_reshuffle_the_order():
    """A different seed must change the order while keeping every sample."""
    # Goal: passing a fresh seed per epoch is what reshuffles the data.
    # Hint: compare batches produced from two different seeds. The column
    # order should differ — yet coverage and pairing must still hold.
    # TODO: replace `pass` with real assertions
    pass


@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_large_batch_size_still_covers_all_samples():
    """A batch_size bigger than m must still return usable batches."""
    # Goal: an edge case — nothing should break or go missing.
    # Hint: call with a batch_size larger than the number of samples. How
    # many batches come back, and does every sample still appear?
    # TODO: replace `pass` with real assertions
    pass
