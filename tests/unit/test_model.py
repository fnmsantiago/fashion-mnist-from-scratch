"""Unit tests for the NeuralNetwork model in ``src/model.py``.

Run just this file:
    pytest tests/unit/test_model.py -v

Run every test in the project:
    pytest -v
"""
import numpy as np

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

        # Bias shape should be (node_count_current_layer, 1)
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


# ---------------------------------------------------------------------------
# forward() — a pass turns (n_x, m) inputs into (n_classes, m) probabilities
# ---------------------------------------------------------------------------

def test_forward_output_shape():
    """Inputs of shape (n_x, m) must come out as (n_classes, m) predictions."""
    classes_count = 10
    layer_dims = [4, 3, classes_count]

    nn = NeuralNetwork(layer_dims)

    m = 7
    x = np.random.randn(4, m)

    (aL, caches) = nn.forward(x)

    assert aL.shape == (classes_count, m)

def test_forward_columns_are_probability_distributions():
    """Each column of al must sum to 1 with all values in (0, 1]."""
    n_classes = 10
    layer_dims = [4, 3, n_classes]

    nn = NeuralNetwork(layer_dims)

    m = 7
    x = np.random.randn(4, m)

    (aL, caches) = nn.forward(x)

    ones = np.ones((1, m))

    np.testing.assert_allclose(aL.sum(axis=0, keepdims=True), ones)

    lower_limit = np.zeros(aL.shape)

    np.testing.assert_array_less(lower_limit, aL)

    assert (aL <= 1.0).all()

def test_hidden_activations_are_non_negative():
    """ReLU in the hidden layers means activations can never be negative."""
    n_features = 20
    layer_dims = [n_features, 15, 4, 3, 10]

    nn = NeuralNetwork(layer_dims)

    m = 7
    x = np.random.randn(n_features, m)

    (aL, caches) = nn.forward(x)

    # For each hidden layer, assert that the activation it received is non-negative.
    for l in range(2, len(layer_dims)):
        a_prev = caches[l][0]

        assert (a_prev >= 0).all()

def test_caches_have_one_four_tuple_per_layer():
    """caches must hold exactly L entries, one (a_prev, W, b, z) per layer."""
    layer_dims = [40, 30, 10]

    nn = NeuralNetwork(layer_dims)

    m = 12
    x = np.random.randn(40, m)

    (aL, caches) = nn.forward(x)

    assert len(caches) == len(layer_dims)-1

    for l in range(1, len(layer_dims)):
        assert len(caches[l]) == 4

def test_cache_shapes_link_adjacent_layers():
    """Each layer's cached values must line up with its own and prior dims."""
    layer_dims = [40, 30, 10]

    nn = NeuralNetwork(layer_dims)

    m = 12
    x = np.random.randn(40, m)

    (aL, caches) = nn.forward(x)

    for l in range(1, len(layer_dims)):
        # a_prev has correct shape
        assert caches[l][0].shape == (layer_dims[l-1], m)

        # W has correct shape
        assert caches[l][1].shape == (layer_dims[l], layer_dims[l-1])

        # b has correct shape
        assert caches[l][2].shape == (layer_dims[l], 1)

        # z has correct shape
        assert caches[l][3].shape == (layer_dims[l], m)

def test_forward_is_deterministic():
    """Two calls with the same weights and input must give identical al."""
    layer_dims = [40, 30, 10]

    nn = NeuralNetwork(layer_dims)

    m = 12
    x = np.random.randn(40, m)

    (aL1, caches) = nn.forward(x)

    (aL2, caches) = nn.forward(x)

    np.testing.assert_allclose(aL1, aL2)