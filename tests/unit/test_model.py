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
    # Goal: no layer may be silently missing from the parameter dict.
    # Hint: which keys (W1, b1, W2, b2, W3, b3) SHOULD exist? How would you
    # assert each one is present inside the parameters dictionary?
    layer_dimensions = [64, 32, 16, 10]
    nn = NeuralNetwork(layer_dimensions)

    for l in range(1, len(layer_dimensions)):
        assert "W"+str(l) in nn.parameters
        assert "b"+str(l) in nn.parameters

@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_parameter_shapes_match_layer_dims():
    """Each W must link its layer to the one before; each b must match its layer."""
    # Goal: the shapes are the contract that forward() depends on.
    # Hint: for layer l, how many rows does W need (its own neurons) and how
    # many columns (the previous layer's neurons)? What about the bias shape?
    # TODO: replace `pass` with real assertions
    pass


@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_biases_are_initialized_to_zero():
    """All biases must start at exactly zero so the first forward is unbiased."""
    # Goal: biases begin neutral; only weights carry initial information.
    # Hint: what does an all-zeros bias array look like, and how would you
    # check every entry of every b[l] is 0?
    # TODO: replace `pass` with real assertions
    pass


@pytest.mark.skip(reason="TODO: implement this test (remove this decorator first)")
def test_weights_follow_he_scale():
    """Weight spreads must follow the He rule for ReLU layers."""
    # Goal: confirm initialization isn't just shaped right, but scaled right.
    # Hint: the He rule ties a weight matrix's spread to the size of the layer
    # it connects FROM. Build many samples of W (or seed first) and compare
    # the observed standard deviation to that theoretical value.
    # TODO: replace `pass` with real assertions
    pass