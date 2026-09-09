"""Unit tests for the evaluation metrics in ``src/metrics.py``.

Run just this file:
    pytest tests/unit/test_metrics.py -v

Run every test in the project:
    pytest -v
"""
import numpy as np
import pytest

from src.metrics import confusion_matrix, precision_recall


# ---------------------------------------------------------------------------
# confusion_matrix() — a (n_classes, n_classes) count grid
# ---------------------------------------------------------------------------
# Learning goals: implement src/metrics.py, then un-skip each test (remove
# the @pytest.mark.skip decorator) and fill in the assertion.

def test_confusion_matrix_known_small_case():
    """A tiny example where every cell is hand-checkable."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 0, 0, 1, 2, 1, 2, 0])
    n_classes = 3

    matrix = confusion_matrix(y_true, y_pred, n_classes)

    expected = np.zeros((n_classes, n_classes), dtype=int)

    # Class 0: 3 cases in y_true
    # true positive cases: 2
    expected[0,0] = 2

    # misclassified as class 1 cases: 1
    expected[0,1] = 1

    # Class 1: 4 cases in y_true
    # true positive cases: 2
    expected[1,1] = 2

    # misclassified as class 0 cases: 1
    expected[1,0] = 1

    # misclassified as class 2 cases: 1
    expected[1,2] = 1

    # Class 2: 2 cases in y_true
    # true positive cases: 1
    expected[2,2] = 1

    # misclassified as class 0 cases: 1
    expected[2,0] = 1

    np.testing.assert_array_equal(matrix, expected)


def test_confusion_matrix_rows_sum_to_true_class_counts():
    """Each row i must sum to the number of samples whose true label is i."""
    np.random.seed(0)
    m = 500
    n_classes = 5
    y_true = np.random.randint(0, n_classes, size=m)
    y_pred = np.random.randint(0, n_classes, size=m)

    matrix = confusion_matrix(y_true, y_pred, n_classes)

    # Count each classes and place them in the appropriate bins. Ex: Class 0 in bin 0.
    # Ensure unrepresented classes are given a count of 0.
    expected = np.bincount(y_true, minlength=n_classes)

    np.testing.assert_array_equal(expected, matrix.sum(axis=1))


# ---------------------------------------------------------------------------
# precision_recall() — per-class precision and recall
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: implement src/metrics.py")
def test_precision_and_recall_known_case():
    """The worked teaching example: precision 0.8, recall 2/3 for class 0."""
    # 30 samples of class 0, 70 of class 1.
    # Class 0 is predicted 25 times: 20 correctly (TP), 5 falsely (FP),
    # and 10 real class-0 samples are missed (FN).
    y_true = np.array([0] * 30 + [1] * 70)
    y_pred = np.array([0] * 20 + [1] * 10 + [0] * 5 + [1] * 65)

    precision, recall = precision_recall(y_true, y_pred, n_classes=2)

    # TODO: assert precision[0] == 0.8 and recall[0] == 2/3.
    pass


@pytest.mark.skip(reason="TODO: implement src/metrics.py")
def test_perfect_predictions_give_perfect_scores():
    """Perfect predictions must give precision and recall of 1.0 for all."""
    np.random.seed(0)
    m = 200
    n_classes = 4
    y = np.random.randint(0, n_classes, size=m)

    precision, recall = precision_recall(y, y, n_classes)

    # TODO: assert precision and recall are all exactly 1.0.
    pass
