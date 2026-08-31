"""Gradient checking for the NeuralNetwork.

Verifies that the analytic gradients produced by ``backward()`` agree with
numeric gradients of the cost estimated by perturbing the parameters.
"""
from typing import Dict, Tuple

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
) -> Tuple[float, Dict[str, float]]:
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
        A tuple that contains the Euclidean distance between the gradients and its approximation
        and the dict mapping each parameter key ('W1', 'b1', ...) to an error
        measure for that parameter.
    """
    # 1. Analytic gradients from one forward + backward pass.
    al, caches = network.forward(x)
    analytic = network.backward(al, y, caches)

    errors: Dict[str, float] = {}

    approximates = {}

    # 2. Numeric gradients and comparison, one parameter at a time.
    for theta_name, theta in network.parameters.items():
        # Pluck the gradient of the parameter.
        grad = analytic[f"d{theta_name}"]

        grad_approx = np.zeros_like(grad)

        # Loop through each element of theta.
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                # Copy the theta first to avoid mutating the original.
                plus = theta.copy()
                minus = theta.copy()

                # Add/subtract epsilon to the element.
                plus[i, j] += epsilon
                minus[i, j] -= epsilon

                # Copy the entire dictionary of parameters.
                plus_params = network.parameters.copy()
                minus_params = network.parameters.copy()

                # Replace the parameter with the modified one.
                plus_params[theta_name] = plus
                minus_params[theta_name] = minus
            
                # Compute the cost as a result of the addition/subtraction.
                J_plus = _cost_with_parameters(network, x, y, plus_params)
                J_minus = _cost_with_parameters(network, x, y, minus_params)

                # Note the approximate gradient using the definition of a derivative.
                grad_approx[i, j] = (J_plus - J_minus)/(2*epsilon)
    
        # Identify the "distance" of what you got v.s. what should be.
        errors[theta_name] = np.linalg.norm(grad - grad_approx)/(np.linalg.norm(grad) + np.linalg.norm(grad_approx))

        # Note the approximate gradient for theta.
        approximates[f"d{theta_name}"] = grad_approx

    # The elements of analytic and approximates have different shapes.
    # In order to take advantage of vectorization, we need to reshape them.
    # Flattening them should do.
    analytic_flattened = [a.reshape(-1) for a in analytic.values()]

    # The analytic order starts from the last layer,
    # whereas approximates starts from the 1st layer.
    # Fix this discrepancy by looping using analytic's order then use the keys to pluck from the approximates. 
    approximates_flattened = [approximates[theta_name].reshape(-1) for theta_name in analytic.keys()]

    # Given your list of 1D arrays, concatenate them to form a true NumPy array.
    grad_arr = np.concatenate(analytic_flattened)
    grad_approx_arr = np.concatenate(approximates_flattened)

    # compute the "distance" for the entire network
    network_error = np.linalg.norm(grad_arr - grad_approx_arr) / (np.linalg.norm(grad_arr) + np.linalg.norm(grad_approx_arr))

    return (network_error, errors)
