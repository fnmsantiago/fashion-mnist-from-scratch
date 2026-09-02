from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from .activations import relu, relu_backward, softmax
from .losses import categorical_cross_entropy_cost
from .utils import one_hot_encode, random_mini_batches

class LayerCache(NamedTuple):
    """The operands and output of a layer's activation."""
    a_prev: np.ndarray
    W: np.ndarray
    b: np.ndarray
    z: np.ndarray

class NeuralNetwork:
    """An L-layer fully-connected network for multi-class classification.

    Hidden layers use ReLU activations; the output layer uses softmax.
    """

    def __init__(self, layer_dims: List[int]) -> None:
        """Initializes the network.

        Args:
            layer_dims: Size of each layer, e.g. [784, 64, 32, 10]. The
                first entry is the input size and the last is the number
                of classes.
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self) -> Dict[str, np.ndarray]:
        """Initializes weights and biases for every layer.

        Returns:
            A dict with keys 'W1', 'b1', ..., 'WL', 'bL'.
        """

        self.parameters = {}

        for l in range(1, self.num_layers + 1):
            curr_neurons = self.layer_dims[l]
            prev_neurons = self.layer_dims[l-1]

            he_multiplier = np.sqrt(2/prev_neurons)

            self.parameters[f"W{l}"] = np.random.randn(curr_neurons, prev_neurons) * he_multiplier
            self.parameters[f"b{l}"] = np.zeros((curr_neurons, 1))

        return self.parameters

        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[int, LayerCache]]:
        """Runs a forward pass through the network.

        Args:
            x: Input data of shape (n_x, m).

        Returns:
            A tuple (al, caches) where al is the softmax output of shape
            (n_classes, m) and caches is a dict keyed by layer index 1..L,
            each value being that layer's a_prev, W, b, and z for the
            values backward() needs.
        """
        # Each layer's cache gets its own entry, keyed by layer index.
        caches: Dict[int, LayerCache] = {}

        # Hidden layers 1..L-1: linear -> relu
        # Output layer L: linear -> softmax
        a_prev = x

        for l in range(1, self.num_layers + 1):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]

            # Pre-activation
            z = W @ a_prev + b

            # Activation
            if l == self.num_layers:
                a = softmax(z)
            else:
                a = relu(z)

            # Cache necessities for backpropagation.
            caches[l] = LayerCache(a_prev, W, b, z)

            # Forward
            a_prev = a

        # Return the prediction and the caches backward will need.
        return (a, caches)

    def compute_cost(self, al: np.ndarray, y: np.ndarray) -> float:
        """Computes the cost for a batch of predictions.

        Args:
            al: Softmax output of shape (n_classes, m).
            y: One-hot labels of shape (n_classes, m).

        Returns:
            The cost, as a scalar.
        """
        return categorical_cross_entropy_cost(al, y)

    def backward(
        self, al: np.ndarray, y: np.ndarray, caches: Dict[int, LayerCache]
    ) -> Dict[str, np.ndarray]:
        """Runs backpropagation and returns the parameter gradients.

        Args:
            al: Softmax output of shape (n_classes, m).
            y: One-hot labels of shape (n_classes, m).
            caches: Dict keyed by layer index 1..L produced by forward();
                each value is that layer's a_prev, W, b, and z.

        Returns:
            A dict with keys 'dW1', 'db1', ..., 'dWL', 'dbL'.
        """
        gradients = {}

        m = al.shape[1]

        # Output layer L: softmax + cross-entropy combine into dZ_L = al - y
        # (the softmax Jacobian cancels the log-derivative), so its gradients
        # are computed explicitly before the hidden-layer loop.
        cache = caches[self.num_layers]

        error_signal = al - y

        gradients[f"dW{self.num_layers}"] = (error_signal @ cache.a_prev.T) / m
        gradients[f"db{self.num_layers}"] = np.mean(error_signal, axis=1, keepdims=True)
        dA_prev = cache.W.T @ error_signal

        # Hidden layers L-1..1: uniform linear -> relu backward.
        for l in range(self.num_layers - 1, 0, -1):
            cache = caches[l]
            error_signal = relu_backward(dA_prev, cache.z)

            gradients[f"dW{l}"] = (error_signal @ cache.a_prev.T) / m
            gradients[f"db{l}"] = np.mean(error_signal, axis=1, keepdims=True)

            # Calculate the error signal of the current layer.
            dA_prev = cache.W.T @ error_signal

        return gradients

    def update_parameters(
        self, grads: Dict[str, np.ndarray], learning_rate: float
    ) -> None:
        """Applies one gradient descent update to the network's parameters.

        Args:
            grads: Gradients produced by backward().
            learning_rate: Step size for the update.
        """
        for (parameter, _) in self.parameters.items():
            # sample parameter name is "W1".
            # sample gradient key is "dW1".

            self.parameters[parameter] -= learning_rate * grads[f"d{parameter}"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts class labels for a batch of inputs.

        Args:
            x: Input data of shape (n_x, m).

        Returns:
            Predicted integer class labels of shape (m,).
        """
        aL, _ = self.forward(x)

        return aL.argmax(axis=0)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        patience: int,
        n_classes: int,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.1,
        print_cost: bool = True,
    ) -> List[float]:
        """Trains the network with mini-batch gradient descent.

        Args:
            x_train: Input data of shape (n_x, train_size).
            y_train: Integer labels of shape (val_size,), not one-hot encoded.
            x_val: The validation data set of shape (n_x, val_size).
            y_val: Integer labels of shape (val_size,), not one-hot encoded.
            patience: The number of epochs that determines early stopping.
            n_classes: Number of classes.
            epochs: Number of passes over the full training set.
            batch_size: Number of examples per mini-batch.
            learning_rate: Step size for gradient descent.
            print_cost: Whether to print the cost periodically.

        Returns:
            The cost after each epoch, for plotting.
        """
        y = one_hot_encode(y_train, n_classes)
        costs = []

        # Record of the lowest validation cost.
        min_val_cost = float('inf')

        # Copy of the parameters that produced the lowest validation cost.
        best_parameters = {}

        # The number of epochs since the last seen improvement.
        epoch_since_improvement = 0

        for epoch in range(epochs):
            mini_batches = random_mini_batches(x_train, y, batch_size, seed=epoch)
            epoch_cost = 0.0

            for x_batch, y_batch in mini_batches:
                al, caches = self.forward(x_batch)
                epoch_cost += self.compute_cost(al, y_batch) * x_batch.shape[1]
                grads = self.backward(al, y_batch, caches)
                self.update_parameters(grads, learning_rate)

            epoch_cost /= x_train.shape[1]
            costs.append(epoch_cost)
            if print_cost and epoch % 5 == 0:
                print(f"Epoch {epoch}: cost = {epoch_cost:.4f}")

            # After each epoch, determine performance on the validation set.
            al, caches = self.forward(x_val)
            val_cost = self.compute_cost(al, y_val)

            if val_cost < min_val_cost:
                min_val_cost = val_cost

                best_parameters = {k:v.copy() for k, v in self.parameters.items()}
                epoch_since_improvement = 0
            else:
                epoch_since_improvement += 1

            # If patience is exhausted, stop early.
            if epoch_since_improvement >= patience:
                break

        # Restore the best parameters.
        self.parameters = best_parameters

        return costs
