from typing import Dict, List, Tuple

import numpy as np

from .activations import relu, relu_backward, softmax
from .losses import categorical_cross_entropy
from .utils import one_hot_encode, random_mini_batches


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

        for l in range(1, self.num_layers):
            curr_neurons = self.layer_dims[l]
            prev_neurons = self.layer_dims[l-1]

            he_multiplier = np.sqrt(2/prev_neurons)
            
            l = str(l)

            self.parameters["W"+l] = np.random.randn(curr_neurons, prev_neurons) * he_multiplier
            self.parameters["b"+l] = np.random.zeros(curr_neurons, 1)

        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, list]:
        """Runs a forward pass through the network.

        Args:
            x: Input data of shape (n_x, m).

        Returns:
            A tuple (al, caches) where al is the softmax output of shape
            (n_classes, m) and caches holds whatever intermediate values
            backward() needs.
        """
        # Hidden layers 1..L-1: linear -> relu
        # Output layer L: linear -> softmax
        # TODO: implement
        pass

    def compute_cost(self, al: np.ndarray, y: np.ndarray) -> float:
        """Computes the cost for a batch of predictions.

        Args:
            al: Softmax output of shape (n_classes, m).
            y: One-hot labels of shape (n_classes, m).

        Returns:
            The cost, as a scalar.
        """
        # TODO: implement
        pass

    def backward(
        self, al: np.ndarray, y: np.ndarray, caches: list
    ) -> Dict[str, np.ndarray]:
        """Runs backpropagation and returns the parameter gradients.

        Args:
            al: Softmax output of shape (n_classes, m).
            y: One-hot labels of shape (n_classes, m).
            caches: Intermediate values produced by forward().

        Returns:
            A dict with keys 'dW1', 'db1', ..., 'dWL', 'dbL'.
        """
        # Output layer L: dZL = al - y, then propagate to dWL, dbL, dA_prev
        # Hidden layers L-1..1: linear -> relu backward
        # TODO: implement
        pass

    def update_parameters(
        self, grads: Dict[str, np.ndarray], learning_rate: float
    ) -> None:
        """Applies one gradient descent update to the network's parameters.

        Args:
            grads: Gradients produced by backward().
            learning_rate: Step size for the update.
        """
        # TODO: implement
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts class labels for a batch of inputs.

        Args:
            x: Input data of shape (n_x, m).

        Returns:
            Predicted integer class labels of shape (m,).
        """
        # TODO: implement
        pass

    def train(
        self,
        x: np.ndarray,
        y_int: np.ndarray,
        n_classes: int,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.1,
        print_cost: bool = True,
    ) -> List[float]:
        """Trains the network with mini-batch gradient descent.

        Args:
            x: Input data of shape (n_x, m).
            y_int: Integer labels of shape (m,), not one-hot encoded.
            n_classes: Number of classes.
            epochs: Number of passes over the full training set.
            batch_size: Number of examples per mini-batch.
            learning_rate: Step size for gradient descent.
            print_cost: Whether to print the cost periodically.

        Returns:
            The cost after each epoch, for plotting.
        """
        y = one_hot_encode(y_int, n_classes)
        costs = []

        for epoch in range(epochs):
            mini_batches = random_mini_batches(x, y, batch_size, seed=epoch)
            epoch_cost = 0.0

            for x_batch, y_batch in mini_batches:
                al, caches = self.forward(x_batch)
                epoch_cost += self.compute_cost(al, y_batch) * x_batch.shape[1]
                grads = self.backward(al, y_batch, caches)
                self.update_parameters(grads, learning_rate)

            epoch_cost /= x.shape[1]
            costs.append(epoch_cost)
            if print_cost and epoch % 5 == 0:
                print(f"Epoch {epoch}: cost = {epoch_cost:.4f}")

        return costs
