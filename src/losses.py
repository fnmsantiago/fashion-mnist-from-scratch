import numpy as np


def categorical_cross_entropy_cost(a: np.ndarray, y: np.ndarray) -> float:
    """Computes the categorical cross-entropy cost.

    Args:
        a: Softmax output of shape (n_classes, m).
        y: One-hot true labels of shape (n_classes, m).

    Returns:
        The cost, averaged over the m examples.
    """
    loss = categorical_cross_entropy_loss(a, y)
    # np.mean returns a numpy scalar; float() converts to a plain Python float
    return float(np.mean(loss))
 
def categorical_cross_entropy_loss(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the categorical cross-entropy loss.

    Args:
        a: Softmax output of shape (n_classes, m).
        y: One-hot true labels of shape (n_classes, m).

    Returns:
        A NumPy array of shape (1, m) of the losses for each sample.
    """
    # Goal: Guard against log(0), an infinite number.
    # A sufficiently small number to replace zeroes while preserving their low probabilities. 
    lower_limit = 1e-12

    # Softmax outputs (probabilites) are already <= 1. This preserves safe values.
    upper_limit = 1.0

    # np.clip clamps all activations into the range [lower_limit, upper_limit] 
    a_clipped = np.clip(a, lower_limit, upper_limit)

    return -np.sum(y * np.log(a_clipped), axis=0, keepdims=True)
