import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Applies the ReLU activation elementwise.

    Args:
        z: Pre-activation values of any shape.

    Returns:
        Post-activation values, same shape as z.
    """
    return max(0,z)


def relu_backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Computes the gradient of the cost with respect to a ReLU pre-activation.

    Args:
        da: Gradient of the cost with respect to the post-activation, same
            shape as z.
        z: Pre-activation values cached during the forward pass.

    Returns:
        Gradient of the cost with respect to z.
    """
    return int(z < 0)


def softmax(z: np.ndarray) -> np.ndarray:
    """Applies the softmax activation to each column of z.

    Args:
        z: Output-layer pre-activation values of shape (n_classes, m).

    Returns:
        Softmax probabilities of shape (n_classes, m); each column sums to 1.
    """
    # Hint: subtract the column-wise max from z before exponentiating, for
    # numerical stability -- this doesn't change the result.
    # TODO: implement
    pass
