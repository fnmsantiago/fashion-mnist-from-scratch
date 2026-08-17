from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import math

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def load_fashion_mnist(
    test_size: int = 10000, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads Fashion-MNIST as normalized, feature-major arrays.

    Downloads the dataset on first use and caches it locally afterward.

    Args:
        test_size: Number of examples to hold out for the test set.
        random_state: Seed for the train/test split.

    Returns:
        A tuple (x_train, x_test, y_train, y_test) where x_train and x_test
        have shape (784, m) with pixel values scaled to [0, 1], and y_train
        and y_test have shape (m,) with integer labels 0-9.
    """
    x, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    x = x / 255.0
    y = y.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return x_train.T, x_test.T, y_train, y_test


def plot_examples(x: np.ndarray, y: np.ndarray, n: int = 8) -> None:
    """Plots a random sample of images with their labels.

    Args:
        x: Image data of shape (784, m).
        y: Integer labels of shape (m,).
        n: Number of examples to plot.
    """
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    idx = np.random.choice(x.shape[1], n, replace=False)
    for ax, i in zip(axes, idx):
        ax.imshow(x[:, i].reshape(28, 28), cmap='gray')
        ax.set_title(CLASS_NAMES[y[i]], fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Converts integer labels to one-hot vectors.

    Args:
        y: Integer labels of shape (m,), in the range [0, n_classes).
        n_classes: Number of classes.

    Returns:
        One-hot encoded labels of shape (n_classes, m).
    """
    # m is the number of samples to encode.
    m = y.shape[0]

    encoded = np.zeros((n_classes, m))

    # For example: y = [3, 7, 5, 1, 0]
    rows = y

    # Makes: columns = [0, 1, ..., m-1]
    columns = np.arange(m)

    # Set all row-column coordinates dictated by `rows` and `columns` to 1.
    # For example: (3, 0) is set to 1. As is, (7, 1).  
    encoded[rows, columns] = 1

    return encoded

def random_mini_batches(
    x: np.ndarray, y: np.ndarray, batch_size: int = 64, seed: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Splits a dataset into shuffled mini-batches.

    Args:
        x: Input data of shape (n_x, m).
        y: One-hot labels of shape (n_classes, m).
        batch_size: Number of examples per batch.
        seed: Random seed; pass a different value each call (e.g. the
            epoch number) to reshuffle.

    Returns:
        A list of (x_batch, y_batch) tuples. The last batch may be smaller
        than batch_size.
    """
    np.random.seed(seed)

    mini_batches = []

    m = x.shape[1]

    shuffled_indexes = list(np.random.permutation(m))

    shuffled_x = x[:,shuffled_indexes]
    shuffled_y = y[:,shuffled_indexes]

    count_full_minibatches = math.floor(m/batch_size)

    for k in range(0, count_full_minibatches):

        start = k * batch_size
        end = start + batch_size

        mini_batch = (shuffled_x[:, start:end], shuffled_y[:, start:end])
        mini_batches.append(mini_batch)

    # Handle the remainder, partial mini_batch
    remainder = m % batch_size

    if remainder != 0:
        mini_batch = (shuffled_x[:, -remainder:], shuffled_y[:, -remainder:])
        mini_batches.append(mini_batch)

    return mini_batches
