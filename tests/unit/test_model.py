"""Unit tests for the NeuralNetwork model in ``src/model.py``.

Run just this file:
    pytest tests/unit/test_model.py -v

Run every test in the project:
    pytest -v

To finish a skipped test:
    1. Remove the ``@pytest.mark.skip(...)`` decorator
    2. Replace the ``pass`` body with real assertions (read the hints)
    3. Run pytest again and watch that test go from SKIPPED to PASSED
"""
import numpy as np
import pytest

from src.model import NeuralNetwork


# ---------------------------------------------------------------------------
# Parameter initialization — checked through the public constructor
# ---------------------------------------------------------------------------
# The constructor calls the private _initialize_parameters(), so building a
# NeuralNetwork and inspecting .parameters is the contract under test.

def test_every_layer_has_weights_and_biases():
    """A network must have W and b for all layers."""
    layer_dimensions = [64, 32, 16, 10]
    nn = NeuralNetwork(layer_dimensions)

    for l in range(1, len(layer_dimensions)):
        assert "W"+str(l) in nn.parameters
        assert "b"+str(l) in nn.parameters

def test_parameter_shapes_match_layer_dims():
    """Each W must link its layer to the one before; each b must match its layer."""
    layer_dimensions = [64, 32, 16, 18]
    nn = NeuralNetwork(layer_dimensions)

    for l in range(1, len(layer_dimensions)):
        # Weight shape should be (node_count_current_layer, node_count_previous_layer)
        assert nn.parameters["W"+str(l)].shape == (layer_dimensions[l], layer_dimensions[l-1])

        # Bias shape should be (node_count_current_layer, node_count_previous_layer)
        assert nn.parameters["b"+str(l)].shape == (layer_dimensions[l], 1)

def test_biases_are_initialized_to_zero():
    """All biases must start at exactly zero so the first forward is unbiased."""
    layer_dimensions = [64, 32, 16, 18]
    nn = NeuralNetwork(layer_dimensions)

    for l in range(1, len(layer_dimensions)):
        # Bias should be zero.
        np.testing.assert_allclose(nn.parameters["b"+str(l)], np.zeros((layer_dimensions[l], 1)))

def test_weights_follow_he_scale():
    """Weight spreads must follow the He rule for ReLU layers."""
    layer_dimensions = [64, 32, 16, 18]

    # Seed the RNG so the random draws are reproducible and stable.
    np.random.seed(0)

    nn = NeuralNetwork(layer_dimensions)

    for l in range(1, len(layer_dimensions)):
        # He rule: the expected std is sqrt(2 / number of neurons it connects FROM).
        expected_std = np.sqrt(2 / layer_dimensions[l-1])

        # A single weight matrix is a sample, so allow a small tolerance.
        np.testing.assert_allclose(np.std(nn.parameters["W"+str(l)]), expected_std, rtol=0.1)
