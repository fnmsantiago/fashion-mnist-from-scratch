"""Gradient checking for the NeuralNetwork.

Verifies that the analytic gradients produced by ``backward()`` agree with
numeric gradients of the cost estimated by perturbing the parameters.
"""
from typing import Dict

import numpy as np

from .model import NeuralNetwork


def _cost_with_parameters(
    network: NeuralNetwork,
    x: np.ndarray,
    y: np.ndarray,
    parameters: Dict[str, np.ndarray],
) -> float:
    """Computes the cost that a given parameter set would produce.

    forward() and compute_cost() read ``self.parameters``, so evaluating an
    alternate parameter set requires swapping it in temporarily. A
    try/finally guarantees the network's real parameters are restored even
    if the forward pass raises.

    Args:
        network: The network to evaluate.
        x: Input data of shape (n_x, m).
        y: One-hot labels of shape (n_classes, m).
        parameters: The parameter dict to evaluate.

    Returns:
        The scalar cost for that parameter set.
    """
    saved_parameters = network.parameters

    # replace the current network's parameters with the one to evaluate.
    network.parameters = parameters

    try:
        al, _ = network.forward(x)
        return network.compute_cost(al, y)
    finally:
        # revert the network's parameters
        network.parameters = saved_parameters


def gradient_check(
    network: NeuralNetwork,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    """Compares backward()'s analytic gradients against numeric ones.

    Numeric gradients of the cost are estimated by perturbing each
    parameter; the comparison with backward()'s gradients uses a
    relative-error measure. The returned dict reports one error per
    parameter.

    Args:
        network: The network to check.
        x: Input data of shape (n_x, m).
        y: One-hot labels of shape (n_classes, m).
        epsilon: Perturbation size used for the finite difference.

    Returns:
        A dict mapping each parameter key ('W1', 'b1', ...) to an error
        measure for that parameter.
    """
    # 1. Analytic gradients from one forward + backward pass.
    al, caches = network.forward(x)
    analytic = network.backward(al, y, caches)

    errors: Dict[str, float] = {}

    # 2. Numeric gradients and comparison, one parameter at a time.
    # TODO: implement — apply the gradient-checking technique from the course.
    for key, param in network.parameters.items():
        errors[key] = np.float64(0.0)

    return errors
