"""Evaluation metrics for the multi-class classifier.

Confusion matrix, precision, and recall computed from scratch.
"""
from typing import Tuple

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    """Builds the confusion matrix for integer labels.

    Args:
        y_true: True labels of shape (m,).
        y_pred: Predicted labels of shape (m,).
        n_classes: Number of classes.

    Returns:
        An (n_classes, n_classes) array where row i, column j is the number
        of samples whose true label is i and predicted label is j.
    """
    matrix = np.zeros((n_classes, n_classes))

    for c in range(n_classes):
        # Extract the indexes of class c as it appears on the true labels.
        true_indexes = np.flatnonzero(y_true == c)

        # count the correct/incorrect classifications
        counts = np.bincount(y_pred[true_indexes], minlength=n_classes)

        matrix[c] = counts

    return matrix


def precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes per-class precision and recall (one-vs-rest).

    Args:
        y_true: True labels of shape (m,).
        y_pred: Predicted labels of shape (m,).
        n_classes: Number of classes.

    Returns:
        Two arrays of shape (n_classes,): precision[c] and recall[c], where
        class c is treated as the positive class and all others as negative.
    """
    matrix = confusion_matrix(y_true, y_pred, n_classes)

    true_positives = matrix.diagonal()

    precision = true_positives / matrix.sum(axis=0)
    recall = true_positives / matrix.sum(axis=1)

    return (precision, recall)