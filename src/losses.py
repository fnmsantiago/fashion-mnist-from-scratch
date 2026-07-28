import numpy as np


def categorical_cross_entropy(a: np.ndarray, y: np.ndarray) -> float:
    """Computes the categorical cross-entropy cost.

    Args:
        a: Softmax output of shape (n_classes, m).
        y: One-hot true labels of shape (n_classes, m).

    Returns:
        The cost, averaged over the m examples.
    """
    # TODO: implement
    pass
